#!/usr/bin/env python3
"""
Step 3: Build Vector Store - Aditya-L1 RAG Pipeline
=====================================================
Embeds all chunks from Steps 1 & 2 into a local ChromaDB vector store.
Also builds a BM25 keyword index for hybrid search in Step 4.

EMBEDDING MODEL: BAAI/bge-m3  (sentence-transformers, GPU-accelerated)
  - 1024-dim, best retrieval quality for technical identifier-heavy documents
  - auto-detects GPU; falls back to CPU if CUDA PyTorch not installed
  - fully local, no data leaves your machine

INPUTS  (from Steps 1 & 2):
  extracted/chunks.json          - text chunks (FCP, MOR, OPS_MANUAL, GUIDELINES)
  extracted/image_captions.json  - image caption chunks
  extracted/tch_records.json     - structured command records

OUTPUTS:
  vectorstore/                   - ChromaDB persistent store
    collection "al1_knowledge"   - procedure + guideline + caption chunks
    collection "al1_commands"    - TCH command log records
  vectorstore/bm25_knowledge.pkl - keyword index for hybrid search
  vectorstore/bm25_commands.pkl  - keyword index for command log

USAGE:
  python step3_build_vectorstore.py               # incremental (safe to re-run)
  python step3_build_vectorstore.py --reset       # wipe and rebuild from scratch

DEPENDENCIES:
  pip install sentence-transformers chromadb rank_bm25
  GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import torch

# sentence-transformers (BGE-M3)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit(
        "Missing dependency:\n"
        "  pip install sentence-transformers\n"
        "BGE-M3 (~570 MB) downloads automatically on first run."
    )

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("Missing dependency:\n  pip install chromadb")

# BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("INFO: rank_bm25 not installed - BM25 keyword index disabled.")
    print("      pip install rank_bm25\n")

# Paths
EXTRACTED_DIR   = Path("extracted")
VECTORSTORE_DIR = Path("vectorstore")
CHUNKS_FILE     = EXTRACTED_DIR / "chunks.json"
CAPTIONS_FILE   = EXTRACTED_DIR / "image_captions.json"
TCH_FILE        = EXTRACTED_DIR / "tch_records.json"

# Config
EMBED_MODEL          = "BAAI/bge-m3"
COLLECTION_KNOWLEDGE = "al1_knowledge"
COLLECTION_COMMANDS  = "al1_commands"

CPU_BATCH_SIZE  = 16     # BGE-M3 on CPU: keep small to avoid OOM
GPU_BATCH_SIZE  = 128    # BGE-M3 on GPU: large batches for speed

MAX_CHUNK_CHARS = 1500   # max characters per chunk before splitting
                         # BGE-M3 limit is ~8192 tokens (~6000 chars),
                         # but 1500 chars gives better retrieval granularity
CHUNK_OVERLAP   = 200    # character overlap between adjacent sub-chunks
                         # prevents losing context at split boundaries


# Embedder (GPU-aware, BGE-M3 only)

class Embedder:
    """
    BGE-M3 through sentence-transformers. Uses the GPU with float16 if CUDA is
    available (about 2x faster, half the VRAM, no real precision loss since the
    embeddings are normalised), otherwise falls back to CPU.
    """

    def __init__(self):
        if torch.cuda.is_available():
            self.device     = "cuda"
            gpu_name        = torch.cuda.get_device_name(0)
            vram_gb         = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.batch_size = GPU_BATCH_SIZE
            print(f"  Device        : GPU - {gpu_name} ({vram_gb:.1f} GB VRAM)")
            print(f"  Batch size    : {self.batch_size}  (GPU mode)")
        else:
            self.device     = "cpu"
            self.batch_size = CPU_BATCH_SIZE
            print(f"  Device        : CPU  (no CUDA-enabled PyTorch found)")
            print(f"  Batch size    : {self.batch_size}  (CPU mode)")
            print(f"  GPU fix       : pip install torch --index-url https://download.pytorch.org/whl/cu121")

        print(f"  Loading       : {EMBED_MODEL} ...")
        self._model = SentenceTransformer(EMBED_MODEL, device=self.device)

        if self.device == "cuda":
            self._model.half()
            print(f"  Precision     : float16  (GPU optimised)")
        else:
            print(f"  Precision     : float32  (CPU default)")

        dim = getattr(self._model, 'get_embedding_dimension',
              getattr(self._model, 'get_sentence_embedding_dimension', None))()
        print(f"  Embedding dim : {dim}")
        print()

    def embed_batch(self, texts: list) -> list:
        vecs = self._model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = False,
            normalize_embeddings = True,   # unit vectors -> cosine similarity = dot product
        )
        return vecs.tolist()


# Text cleaning

def clean_caption_text(text: str) -> str:
    """
    Remove leading caption prefix artifacts that Qwen2.5VL sometimes outputs.
    These add noise to embeddings without contributing semantic meaning.

    Examples removed:
      "Caption: The image shows..."       -> "The image shows..."
      "Image Caption: Flow diagram of..." -> "Flow diagram of..."
      "Figure 3: MRS recovery flow"       -> "MRS recovery flow"
    """
    patterns = [
        r'^Image\s+Caption\s*:?\s*',
        r'^Caption\s*:?\s*',
        r'^Figure\s+\d+[.:]\s*',
        r'^Fig\.\s*\d+[.:]\s*',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()


# Chunk splitting

def smart_split(text: str,
                max_chars: int = MAX_CHUNK_CHARS,
                overlap: int   = CHUNK_OVERLAP) -> list:
    """
    Split long text into overlapping chunks, preferring sentence boundaries.
    BGE-M3 truncates anything past ~8192 tokens, and a huge single chunk loses
    all its internal detail, so ~1500-char chunks keep retrieval fine-grained.
    Boundary preference: '. ', '.\\n', '\\n\\n', '\\n', then a hard cut.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end == len(text):
            chunks.append(text[start:].strip())
            break

        # Walk back from end to find a clean boundary
        split_at = end
        for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
            pos = text.rfind(sep, start + overlap, end)
            if pos != -1 and pos > start:
                split_at = pos + len(sep)
                break

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, split_at - overlap)

    return [c for c in chunks if c.strip()]


def expand_with_splits(items: list, id_field: str) -> list:
    """
    Apply smart_split to every item whose text exceeds MAX_CHUNK_CHARS.
    Adds parent linkage metadata so Step 4 can reconstruct full context.

    Sub-chunk IDs: "{original_id}_sub0", "_sub1", ...
    Metadata added:
      is_sub_chunk    - True if this item was produced by splitting
      parent_chunk_id - original chunk id (empty for unsplit items)
      sub_index       - position within the split sequence (0-based)
    """
    expanded = []
    n_items_split = 0

    for item in items:
        text  = item.get("text", "")
        ctype = item.get("content_type", "")

        # Clean caption text before length check
        if ctype == "image_caption":
            text = clean_caption_text(text)

        parts = smart_split(text)

        if len(parts) == 1:
            new_item                  = dict(item)
            new_item["text"]          = parts[0]
            new_item["is_sub_chunk"]  = False
            new_item["parent_chunk_id"] = ""
            new_item["sub_index"]     = 0
            expanded.append(new_item)
        else:
            original_id = str(item.get(id_field, ""))
            n_items_split += 1
            for i, part in enumerate(parts):
                new_item                    = dict(item)
                new_item["text"]            = part
                new_item[id_field]          = f"{original_id}_sub{i}"
                new_item["is_sub_chunk"]    = True
                new_item["parent_chunk_id"] = original_id
                new_item["sub_index"]       = i
                expanded.append(new_item)

    if n_items_split:
        before = len(items)
        after  = len(expanded)
        print(f"  Split {n_items_split} oversized items: {before} -> {after} chunks")

    return expanded


# Diagnostics

def print_diagnostics(items: list, label: str) -> None:
    """
    Print chunk size distribution and source breakdown.
    Helps catch problems like a 300,000-char MOR chunk before it becomes
    a bad embedding.
    """
    if not items:
        return

    sizes = [len(item.get("text", "")) for item in items]
    avg_s = sum(sizes) / len(sizes)
    max_s = max(sizes)
    over  = sum(1 for s in sizes if s > MAX_CHUNK_CHARS)

    by_source: dict = {}
    by_type:   dict = {}
    for item in items:
        src   = item.get("source_doc", "?")
        ctype = item.get("content_type", "?")
        by_source[src]  = by_source.get(src, 0)  + 1
        by_type[ctype]  = by_type.get(ctype, 0) + 1

    print(f"  ── {label} ──")
    print(f"  Total         : {len(items)} chunks")
    print(f"  Avg size      : {avg_s:,.0f} chars")
    print(f"  Max size      : {max_s:,} chars" +
          (f"  ⚠  {over} chunks exceed {MAX_CHUNK_CHARS} limit" if over else ""))
    print(f"  By source     : {dict(sorted(by_source.items()))}")
    print(f"  By type       : {dict(sorted(by_type.items()))}")
    print()


# Text preparation for embedding

def build_embed_text(item: dict) -> str:
    """
    Build the string that gets embedded.

    Prefix with structured metadata so:
      "OBC switch on procedure"      -> retrieves FCP-5201
      "MRS recovery flow diagram"    -> retrieves the right image caption
      "OBCS2370 command"             -> retrieves the exact TCH record
    """
    parts = []
    ctype = item.get("content_type", "")

    # Image captions: add explicit type + location prefix.
    # Without this, captions embed like plain text and lose the
    # "this is a diagram of X" signal that image queries need.
    if ctype == "image_caption":
        src  = item.get("source_doc", "")
        page = item.get("page")
        parts.append("Image caption")
        if src:
            parts.append(f"Source: {src}")
        if page:
            parts.append(f"Page: {page}")

    # Standard metadata prefix for all types
    if item.get("fcp_number"):
        parts.append(f"FCP {item['fcp_number']}")
    sub = item.get("subsystem", "")
    if sub and sub not in ("", "UNKNOWN"):
        parts.append(f"Subsystem: {sub}")
    if item.get("mnemonic"):
        parts.append(f"Command: {item['mnemonic']}")
    if item.get("cid"):
        parts.append(f"CID: {item['cid']}")
    if item.get("section_heading"):
        parts.append(item["section_heading"])

    # Body - caption text is already cleaned by expand_with_splits
    body = item.get("text", "").strip()
    if body:
        parts.append(body)

    return "\n".join(parts)


def build_metadata(item: dict) -> dict:
    """
    Flatten item to ChromaDB-safe types (str / int / float / bool only).
    Lists and nested dicts are not allowed in ChromaDB metadata.
    """
    images = item.get("images", [])
    return {
        # Core
        "source_doc":       str(item.get("source_doc", "")),
        "doc_filename":     str(item.get("doc_filename", "")),
        "content_type":     str(item.get("content_type", "")),
        "page":             int(item.get("page") or 0),
        "section_heading":  str(item.get("section_heading", ""))[:400],
        "fcp_number":       str(item.get("fcp_number", "")),
        "subsystem":        str(item.get("subsystem", "")),
        # Parent linkage (for reconstructing context from sub-chunks)
        "is_sub_chunk":     bool(item.get("is_sub_chunk", False)),
        "parent_chunk_id":  str(item.get("parent_chunk_id", "")),
        "sub_index":        int(item.get("sub_index", 0)),
        # TCH-specific
        "date":             str(item.get("date", "")),
        "time":             str(item.get("time", "")),
        "cid":              str(item.get("cid", "")),
        "mnemonic":         str(item.get("mnemonic", "")),
        "orbit_no":         str(item.get("orbit_no", "")),
        "status":           str(item.get("status", "")),
        "source_station":   str(item.get("source_station", "")),
        "dest_station":     str(item.get("dest_station", "")),
        # Image flags
        "has_images":       bool(images),
        "image_count":      int(len(images)),
        "image_file":       str(item.get("image_file", "")),
    }


# Deduplication

def deduplicate(items: list, id_field: str) -> list:
    """
    Remove items with duplicate IDs before sending to ChromaDB.

    Step1 can produce colliding chunk_ids when the same document section is
    processed more than once, or when page-based ID generation isn't globally
    unique (e.g. two chunks both end up as 'MOR_1_0').
    ChromaDB raises DuplicateIDError if ANY item in a batch shares an ID,
    so we must strip duplicates before upsert. First occurrence is kept.
    """
    seen:   set  = set()
    result: list = []
    dupes:  int  = 0
    for item in items:
        item_id = str(item.get(id_field, ""))
        if item_id in seen:
            dupes += 1
        else:
            seen.add(item_id)
            result.append(item)
    if dupes:
        print(f"  ⚠  Removed {dupes} duplicate IDs  "
              f"({len(items)} -> {len(result)} unique items)")
        print(f"     Root cause: step1.py ID collision - consider fixing ID generation")
    return result


# BM25

def tokenize(text: str) -> list:
    """
    Tokenizer for aerospace documents.

    Handles compound tokens AND their parts:
      L1/L2    -> ['l1/l2', 'l1', 'l2']   exact slash-match + individual parts
      DC-DC-11 -> ['dc-dc-11']             full hyphenated token preserved
      OBCS2370 -> ['obcs2370']             CID preserved as single token
      TT&C     -> ['tt', 'c']              & dropped; fine for BM25
    """
    text_lower = text.lower()
    compound   = re.findall(r'[a-z0-9]+(?:[/\.][a-z0-9]+)+', text_lower)
    standard   = re.findall(r'[a-z0-9_\-]+', text_lower)
    seen, result = set(), []
    for token in compound + standard:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def build_bm25_index(items: list, out_path: Path) -> None:
    if not HAS_BM25:
        return
    print(f"  BM25 index -> {out_path.name} ...", end=" ", flush=True)
    corpus = [tokenize(build_embed_text(item)) for item in items]
    ids    = [item.get("chunk_id") or item.get("record_id") for item in items]
    bm25   = BM25Okapi(corpus)
    out_path.write_bytes(pickle.dumps({"bm25": bm25, "ids": ids, "items": items}))
    print(f"({len(ids)} docs)")


# ChromaDB upsert

def upsert_collection(collection, items: list, embedder: Embedder, id_field: str) -> int:
    """
    Embed + upsert in batches.
    Skips items already in the collection - safe to re-run after a crash.
    """
    existing = set(collection.get(include=[])["ids"])
    pending  = [it for it in items if str(it.get(id_field, "")) not in existing]

    if not pending:
        print(f"  All {len(existing)} already stored - nothing to add.")
        return 0

    print(f"  Existing: {len(existing)} | New: {len(pending)}")
    n_done = 0

    for i in range(0, len(pending), embedder.batch_size):
        batch      = pending[i : i + embedder.batch_size]
        ids        = [str(it[id_field])          for it in batch]
        texts      = [build_embed_text(it)       for it in batch]
        documents  = [it.get("text", "")[:8000]  for it in batch]
        metadatas  = [build_metadata(it)         for it in batch]
        embeddings = embedder.embed_batch(texts)

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas,
        )
        n_done += len(batch)
        print(f"  {n_done}/{len(pending)} ...", end="\r", flush=True)

    print(f"  {n_done} items embedded & stored.          ")
    return n_done


# Main

def run(reset: bool = False):
    VECTORSTORE_DIR.mkdir(exist_ok=True)

    print("=" * 62)
    print("Aditya-L1 RAG - Step 3: Build Vector Store")
    print(f"Embedding model : {EMBED_MODEL}")
    print("=" * 62)

    embedder = Embedder()

    # ChromaDB
    client = chromadb.PersistentClient(
        path     = str(VECTORSTORE_DIR),
        settings = Settings(anonymized_telemetry=False),
    )

    if reset:
        print("  --reset: deleting existing collections ...")
        for name in [COLLECTION_KNOWLEDGE, COLLECTION_COMMANDS]:
            try:
                client.delete_collection(name)
            except Exception:
                pass

    col_knowledge = client.get_or_create_collection(
        COLLECTION_KNOWLEDGE, metadata={"hnsw:space": "cosine"}
    )
    col_commands = client.get_or_create_collection(
        COLLECTION_COMMANDS, metadata={"hnsw:space": "cosine"}
    )

    # Load inputs
    knowledge_items: list = []
    tch_items:       list = []

    for path, label in [
        (CHUNKS_FILE,   "chunks.json"),
        (CAPTIONS_FILE, "image_captions.json"),
    ]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"  {label:<28} {len(data):>5} items")
            knowledge_items.extend(data)
        else:
            print(f"  {label:<28}  not found - skipping")

    if TCH_FILE.exists():
        data = json.loads(TCH_FILE.read_text(encoding="utf-8"))
        print(f"  {'tch_records.json':<28} {len(data):>5} records")
        tch_items.extend(data)
    else:
        print(f"  {'tch_records.json':<28}  not found - skipping")

    print()

    # Deduplicate (fix step1 ID collisions)
    knowledge_items = deduplicate(knowledge_items, id_field="chunk_id")
    tch_items       = deduplicate(tch_items,       id_field="record_id")

    # Diagnostics + split oversized chunks
    print_diagnostics(knowledge_items, "knowledge (before split)")
    knowledge_items = expand_with_splits(knowledge_items, id_field="chunk_id")
    print_diagnostics(knowledge_items, "knowledge (after split)")

    print_diagnostics(tch_items, "commands")
    tch_items = expand_with_splits(tch_items, id_field="record_id")

    # Embed + store knowledge
    print(f"[1/2] Knowledge collection  ({len(knowledge_items)} chunks)")
    upsert_collection(col_knowledge, knowledge_items, embedder, id_field="chunk_id")
    build_bm25_index(knowledge_items, VECTORSTORE_DIR / "bm25_knowledge.pkl")

    # Embed + store TCH
    print(f"\n[2/2] Commands collection  ({len(tch_items)} records)")
    upsert_collection(col_commands, tch_items, embedder, id_field="record_id")
    build_bm25_index(tch_items, VECTORSTORE_DIR / "bm25_commands.pkl")

    # Final summary
    print()
    print("=" * 62)
    print("Step 3 complete")
    print("=" * 62)
    print(f"  al1_knowledge  {col_knowledge.count():>5} vectors")
    print(f"  al1_commands   {col_commands.count():>5} vectors")
    print(f"  Location       {VECTORSTORE_DIR.resolve()}")
    if HAS_BM25:
        print(f"  BM25           bm25_knowledge.pkl  bm25_commands.pkl")
    print()
    print("Next: python step4_query.py")
    print("=" * 62)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 3: Build Aditya-L1 vector store")
    ap.add_argument("--reset", action="store_true",
                    help="Drop existing collections and rebuild from scratch")
    args = ap.parse_args()
    run(reset=args.reset)

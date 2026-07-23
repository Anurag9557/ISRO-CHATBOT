#!/usr/bin/env python3
"""
Step 4: Query Engine - Aditya-L1 RAG Pipeline
==============================================
Semantic + keyword hybrid search over the vector store built in Step 3.
Generates answers using a local Ollama model. No data leaves your machine.

USAGE:
  Interactive mode (chat loop):
    python step4_query.py

  Single question:
    python step4_query.py "What is the procedure for switching on OBC?"

  Single question, show retrieved chunks too:
    python step4_query.py "What are safety logics?" --show-sources

  Date-based command log query:
    python step4_query.py "What commands were uplinked on 2026-06-09?"

  Filter by subsystem:
    python step4_query.py "Switch on procedure" --subsystem OBC

GENERATION (fully local - no cloud, no external API):
    ollama pull qwen2.5:7b-instruct   (text model used for answers)

DEPENDENCIES:
  pip install chromadb rank_bm25
  + same embedding backend as Step 3 (BGE-M3 / sentence-transformers)
"""

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("Missing: pip install chromadb")

# Embedding (must match the model used in Step 3)
try:
    from sentence_transformers import SentenceTransformer as _ST
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

if not HAS_ST and not HAS_OLLAMA:
    sys.exit("Missing embedding backend. Install sentence-transformers or ollama.")

# BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# Config
VECTORSTORE_DIR      = Path("vectorstore")
EMBED_MODEL_ST       = "BAAI/bge-m3"          # must match step3
EMBED_MODEL_OLLAMA   = "nomic-embed-text"      # fallback if ST not installed

OLLAMA_GEN_MODEL     = "qwen2.5:7b-instruct"   # text model - far better at procedures than the vision model
# NOTE: step2 captioning still uses qwen2.5vl (vision). Generation here uses
# the text model qwen2.5:7b-instruct: it lists FCP command steps properly and is
# lighter on the GPU, which also avoids the intermittent CUDA crash from qwen2.5vl.

COLLECTION_KNOWLEDGE = "al1_knowledge"
COLLECTION_COMMANDS  = "al1_commands"

TOP_K_SEMANTIC  = 10    # semantic hits before merge
TOP_K_BM25      = 10    # BM25 keyword hits before merge
TOP_K_FINAL     = 5     # chunks sent to LLM after RRF merge
RRF_K           = 60    # RRF smoothing constant

SYSTEM_PROMPT = """You are an Aditya-L1 spacecraft operations assistant for ISRO mission control.
Answer questions using ONLY the provided context from official mission documents.

Rules:
- For FCP procedures: cite the FCP number and list steps explicitly.
- For image captions: mention the diagram and its source page.
- If the answer is not in the context, say "Not found in available documents." Do not guess.
- Never speculate about spacecraft operations beyond what the documents state.
- State which document and section each claim comes from.
- Keep answers concise and operationally precise."""


# Result type

@dataclass
class Result:
    chunk_id: str
    text: str
    score: float
    metadata: dict


# Embedding

_embed_model = None

def embed(text: str) -> list:
    global _embed_model
    if HAS_ST:
        if _embed_model is None:
            print("  Loading embedding model ...", end=" ", flush=True)
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embed_model = _ST(EMBED_MODEL_ST, device=device)
            print("ready")
        return _embed_model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
    # Fallback: Ollama
    return _ollama.embeddings(model=EMBED_MODEL_OLLAMA, prompt=text)["embedding"]


# Query routing

def route(query: str) -> str:
    """
    Send a question to the command-log store only on strong log signals,
    otherwise to the knowledge store. The words 'command'/'telecommand' alone
    aren't enough (they show up in procedure titles), so I look for an explicit
    date, a CID code, 'uplinked', 'command/telecommand history/log', 'TCH', or
    an 'orbit <n>' reference.
    """
    if re.search(r'\b(20\d{2})[-/\s](0[1-9]|1[0-2])[-/\s](0[1-9]|[12]\d|3[01])\b', query):
        return "commands"                                     # explicit date
    if re.search(r'\b[A-Z]{3,5}\d{3,4}\b', query):
        return "commands"                                     # a CID, e.g. OBCS2370
    if re.search(r'\buplink(ed|s|ing)?\b', query, re.IGNORECASE):
        return "commands"                                     # 'uplinked' is log-specific
    if re.search(r'\b(?:tele)?command\s+(?:history|log)\b', query, re.IGNORECASE):
        return "commands"                                     # 'command/telecommand history/log'
    if re.search(r'\bTCH\b', query):
        return "commands"
    if re.search(r'\borbit\s+\d+\b', query, re.IGNORECASE):
        return "commands"                                     # 'orbit 66'
    return "knowledge"


def extract_date(query: str) -> Optional[str]:
    m = re.search(r'\b(20\d{2})[-/\s](0[1-9]|1[0-2])[-/\s](0[1-9]|[12]\d|3[01])\b', query)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_subsystem(query: str) -> Optional[str]:
    mapping = [
        ("OBC",        r'\bOBC\b'),
        ("AOCS",       r'\bAOCS\b'),
        ("PAYLOAD",    r'\bpayload\b|\bPAPA\b|\bASPEX\b|\bVELC\b|\bHELIOS\b|\bSOLEXS\b'),
        ("POWER",      r'\bpower\b|\bbattery\b|\bFPGA\b|\bDC.DC\b'),
        ("DTG",        r'\bDTG\b|\bgyro\b|\bCSA\b'),
        ("MECHANISM",  r'\bHGA\b|\bmechanism\b|\bmotor\b|\bSPDM\b|\bMCE\b'),
        ("PROPULSION", r'\bthruster\b|\bpropulsion\b|\blatch valve\b|\bfuel\b'),
        ("TTC_XBAND",  r'\bTTC\b|\bX.band\b|\btransponder\b|\breceiver\b|\bTWTA\b'),
        ("WHEEL",      r'\bwheel\b|\bWDE\b|\bsuspension\b'),
        ("SENSOR",     r'\bstar sensor\b'),
        ("ODHS",       r'\bODHS\b|\bSSR\b|\bDFU\b'),
    ]
    for sub, pat in mapping:
        if re.search(pat, query, re.IGNORECASE):
            return sub
    return None


# BM25 search

def tokenize(text: str) -> list:
    text_lower = text.lower()
    compound = re.findall(r'[a-z0-9]+(?:[/\.][a-z0-9]+)+', text_lower)
    standard = re.findall(r'[a-z0-9_\-]+', text_lower)
    seen, result = set(), []
    for tok in compound + standard:
        if tok and tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def bm25_search(query: str, pkl_path: Path, k: int = TOP_K_BM25,
                subsystem: Optional[str] = None) -> list:
    if not HAS_BM25 or not pkl_path.exists():
        return []
    data  = pickle.loads(pkl_path.read_bytes())
    bm25  = data["bm25"]
    ids   = data["ids"]
    items = data["items"]
    scores = bm25.get_scores(tokenize(query))
    # Get more candidates when filtering so we have enough after subsystem filter
    fetch_k = k * 3 if subsystem else k
    top_n   = sorted(range(len(scores)), key=lambda i: -scores[i])[:fetch_k]
    results = []
    for idx in top_n:
        if scores[idx] <= 0:
            break
        item = items[idx]
        # Apply subsystem filter: skip non-matching items but keep UNKNOWN
        # (UNKNOWN means subsystem wasn't detected, not that it's wrong)
        if subsystem and item.get("subsystem") not in (subsystem, "UNKNOWN", ""):
            continue
        results.append(Result(
            chunk_id = ids[idx],
            text     = item.get("text", ""),
            score    = float(scores[idx]),
            metadata = item,
        ))
        if len(results) >= k:
            break
    return results


# Semantic search

def semantic_search(
    query: str, collection, k: int = TOP_K_SEMANTIC,
    where: Optional[dict] = None,
) -> list:
    qe  = embed(query)
    kw  = dict(query_embeddings=[qe], n_results=k,
               include=["documents", "metadatas", "distances"])
    if where:
        kw["where"] = where
    res = collection.query(**kw)
    results = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0]
    ):
        results.append(Result(
            chunk_id = cid,
            text     = doc,
            score    = 1 - dist,       # cosine distance -> similarity
            metadata = meta,
        ))
    return results


# RRF merge

def rrf_merge(ranked_lists: list, k: int = RRF_K) -> list:
    scores: dict = {}
    store:  dict = {}
    for ranked in ranked_lists:
        for rank, r in enumerate(ranked):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank + 1)
            store[r.chunk_id]  = r
    return [store[cid] for cid in sorted(scores, key=lambda c: -scores[c])]


# Context builder

def build_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        m   = r.metadata
        src = m.get("source_doc", "?")
        fcp = m.get("fcp_number", "")
        sub = m.get("subsystem", "")
        pg  = m.get("page", 0)
        ctype = m.get("content_type", "")

        header = f"[Source {i} | {src}"
        if fcp:
            header += f" FCP-{fcp}"
        if sub and sub not in ("", "UNKNOWN"):
            header += f" · {sub}"
        if pg:
            header += f" · page {pg}"
        header += f" · {ctype}]"
        parts.append(f"{header}\n{r.text.strip()[:2000]}")

    return "\n\n" + ("\n\n" + "─" * 50 + "\n\n").join(parts)


def format_sources(results: list) -> str:
    lines = []
    seen  = set()   # dedup key: (fcp_number, heading_prefix) or (source, heading_prefix)
    idx   = 1
    for r in results:
        m       = r.metadata
        fcp     = m.get("fcp_number", "")
        heading = m.get("section_heading", "")
        src     = m.get("source_doc", "?")
        # Build a dedup key - sub-chunks of the same FCP section look identical.
        # Image captions key on their file so they are never collapsed into a
        # text chunk from the same page (we want the diagram as its own source).
        if m.get("content_type") == "image_caption":
            dedup_key = ("IMG", m.get("image_file", ""))
        else:
            dedup_key = (src, fcp, heading[:40])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        parts = [f"[{idx}]", src]
        if fcp:
            parts.append(f"FCP-{fcp}")
        if m.get("subsystem") and m["subsystem"] not in ("", "UNKNOWN"):
            parts.append(m["subsystem"])
        if m.get("page"):
            parts.append(f"p.{m['page']}")
        if heading:
            parts.append(f'"{heading[:55]}{"..." if len(heading) > 55 else ""}"')
        if m.get("content_type") == "image_caption":
            parts.append("🖼 diagram")
        lines.append("  " + " · ".join(parts))
        # Show the image file path so diagram/figure sources can be opened
        if m.get("content_type") == "image_caption":
            img = m.get("image_file", "")
            if img:
                lines.append(f"      📎 {img}")
        idx += 1
    return "\n".join(lines)


def wants_image(query: str) -> bool:
    """True if the query is asking for a diagram / figure / image."""
    return bool(re.search(
        r'\b(diagram|figure|flow[\s-]?chart|flowchart|block\s+diagram|schematic|'
        r'image|picture|illustration|drawing|screenshot|chart|plot|graph)\b',
        query, re.IGNORECASE))


def collect_images(results: list) -> list:
    """Ordered, de-duplicated image-file paths from image_caption results."""
    files = []
    for r in results:
        m = r.metadata
        if m.get("content_type") == "image_caption":
            f = m.get("image_file", "")
            if f and f not in files:
                files.append(f)
    return files


def open_images(files: list) -> None:
    """Open image files in the OS default viewer (Windows: os.startfile)."""
    import os
    for f in files:
        try:
            if not os.path.exists(f):
                print(f"  (missing on disk) {f}")
            elif hasattr(os, "startfile"):
                print(f"  Opening: {f}")
                os.startfile(f)                      # Windows default viewer
            else:                                    # macOS / Linux fallback
                import subprocess, sys
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", f])
                print(f"  Opening: {f}")
        except Exception as exc:
            print(f"  Could not open {f}: {exc}")


def wants_procedure(query: str) -> bool:
    """
    True if the query looks like a procedure / 'which FCP for X' lookup.
    Used to boost the (short) FCP index + procedure chunks, which otherwise
    get out-ranked by longer MOR text chunks for near-exact title queries.
    """
    return bool(re.search(
        r'\b(fcp|procedure|how\s+to|steps?|switch\w*|power\w*|turn\w*|'
        r'enabl\w*|disabl\w*|reset|select\w*|activat\w*|deactivat\w*)\b',
        query, re.IGNORECASE))


# Generation

def generate_ollama(query: str, context: str) -> str:
    resp = _ollama.chat(
        model    = OLLAMA_GEN_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        options  = {"num_ctx": 8192},
    )
    return resp["message"]["content"]


def generate(query: str, context: str) -> str:
    """Generate an answer with the local Ollama text model (fully offline)."""
    if not HAS_OLLAMA:
        return f"[ERROR] Ollama not installed. Run: pip install ollama && ollama pull {OLLAMA_GEN_MODEL}"
    return generate_ollama(query, context)


# Main query function

# Lazy-loaded singletons
_client = None
_col_know = None
_col_cmd  = None

def _load_stores():
    global _client, _col_know, _col_cmd
    if _client is None:
        _client   = chromadb.PersistentClient(
            path     = str(VECTORSTORE_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
        _col_know = _client.get_collection(COLLECTION_KNOWLEDGE)
        _col_cmd  = _client.get_collection(COLLECTION_COMMANDS)


def ask(
    query:          str,
    top_k:          int           = TOP_K_FINAL,
    subsystem_hint: Optional[str] = None,
) -> tuple:
    """
    Route -> search -> rank -> generate.
    Returns (answer: str, sources: list[Result]).
    """
    _load_stores()
    target = route(query)

    # Command log query
    if target == "commands":
        date_filter = extract_date(query)

        # A "what commands on <date>" question wants the whole list, not the
        # top few by relevance, so when there's a date I pull every matching
        # record straight from the store and summarise it.
        if date_filter:
            got   = _col_cmd.get(where={"date": {"$eq": date_filter}},
                                 include=["documents", "metadatas"])
            metas = got.get("metadatas") or []
            ids   = got.get("ids") or []
            docs  = got.get("documents") or []
            if not metas:
                return f"No commands found for {date_filter}.", []

            # Chronological order (by time-of-day)
            order = sorted(range(len(metas)), key=lambda i: metas[i].get("time", ""))
            metas = [metas[i] for i in order]
            ids   = [ids[i]   if i < len(ids)  else "" for i in order]
            docs  = [docs[i]  if i < len(docs) else "" for i in order]

            from collections import Counter
            by_sub = Counter(m.get("subsystem", "?") for m in metas)
            by_mn  = Counter(m.get("mnemonic",  "?") for m in metas)
            sub_str = ", ".join(f"{v} {k}" for k, v in by_sub.most_common())

            lines = [f"**{len(metas)} commands uplinked on {date_filter}**  ({sub_str})\n"]
            lines.append("Unique commands (count × mnemonic):")
            for mn, c in by_mn.most_common():
                lines.append(f"  {c:>3} ×  {mn}")
            lines.append("")
            lines.append(f"{'#':<4}  {'Time':<10}  {'CID':<14}  {'Mnemonic':<40}  {'Orbit':<7}  Status")
            lines.append("─" * 96)
            results = []
            for i, m in enumerate(metas, 1):
                lines.append(
                    f"{i:<4}  {m.get('time','?'):<10}  {m.get('cid','?'):<14}  "
                    f"{m.get('mnemonic','?'):<40}  {str(m.get('orbit_no','?')):<7}  {m.get('status','?')}"
                )
                results.append(Result(chunk_id=ids[i-1], text=docs[i-1],
                                      score=1.0, metadata=m))
            return "\n".join(lines), results

        # Non-date command query: relevance-ranked top-k (unchanged)
        sem  = semantic_search(query, _col_cmd, k=top_k * 2)
        bm25 = bm25_search(query, VECTORSTORE_DIR / "bm25_commands.pkl", k=top_k * 2)
        merged = rrf_merge([sem, bm25])[:top_k]

        if not merged:
            return "No matching command records found.", []

        lines = [f"Commands matching: **{query}**\n"]
        lines.append(f"{'#':<4}  {'CID':<14}  {'Mnemonic':<40}  {'Date':<12}  {'Time':<10}  Status")
        lines.append("─" * 90)
        for i, r in enumerate(merged, 1):
            m = r.metadata
            lines.append(
                f"{i:<4}  {m.get('cid','?'):<14}  {m.get('mnemonic','?'):<40}  "
                f"{m.get('date','?'):<12}  {m.get('time','?'):<10}  {m.get('status','?')}"
            )
        return "\n".join(lines), merged

    # Knowledge query
    sub   = subsystem_hint or extract_subsystem(query)
    # keep UNKNOWN-tagged chunks in the filter too, so a chunk isn't dropped
    # just because its subsystem couldn't be inferred (BM25 keeps them anyway)
    where = {"subsystem": {"$in": [sub, "UNKNOWN", ""]}} if sub else None

    sem  = semantic_search(query, _col_know, k=top_k * 2, where=where)
    bm25 = bm25_search(query, VECTORSTORE_DIR / "bm25_knowledge.pkl",
                       k=top_k * 2, subsystem=sub)

    # If filtered search returned nothing, retry without filter
    if not sem and where:
        sem = semantic_search(query, _col_know, k=top_k * 2)

    merged = rrf_merge([sem, bm25])[:top_k]

    # if the user asks for a diagram, pull the best image-caption chunks to the
    # front so the figure is both cited and described
    if wants_image(query):
        imgs = semantic_search(query, _col_know, k=3,
                               where={"content_type": {"$eq": "image_caption"}})
        if imgs:
            seen_ids, combined = set(), []
            for r in imgs[:2] + merged:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    combined.append(r)
            merged = combined

    # for procedure / "which FCP for X" questions, pull the best-matching FCP
    # chunks to the front. The index records are short and otherwise lose the
    # fused ranking to longer MOR text even on near-exact title matches.
    if wants_procedure(query):
        fcp_hits = semantic_search(query, _col_know, k=3,
                                   where={"source_doc": {"$eq": "FCP"}})
        if fcp_hits:
            seen_ids, combined = set(), []
            for r in fcp_hits[:2] + merged:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    combined.append(r)
            merged = combined

    if not merged:
        return "No relevant documents found.", []

    # Diagram/figure requests: answer from the caption, skip the text LLM
    # A text model asked to "provide a diagram" tends to reply "Not found in
    # available documents" (it cannot render an image) even though the figure
    # WAS retrieved and is shown to the user. For image-intent queries we answer
    # directly from the best-matching caption instead - clearer, and it also
    # avoids an unnecessary (crash-prone) generation call.
    if wants_image(query):
        caps = [r for r in merged
                if (r.metadata or {}).get("content_type") == "image_caption"]
        if caps:
            m   = caps[0].metadata or {}
            loc = m.get("source_doc", "document")
            if m.get("page"):
                loc += f", page {m['page']}"
            desc = (caps[0].text or "").strip()
            answer = (f"Here is the figure matching your request (from {loc}):\n\n"
                      f"{desc}\n\n_The diagram is shown below._")
            return answer, merged

    context = build_context(merged)
    # a transient Ollama/GPU error shouldn't kill the whole query; on failure
    # still return the retrieved sources so the user has something to go on
    try:
        answer = generate(query, context)
    except Exception as exc:
        answer = (
            f"[Generation unavailable - {type(exc).__name__}] "
            "Retrieval succeeded but the local model failed to produce an answer "
            "(usually a transient Ollama/GPU error; re-running often works). "
            "The most relevant sources are listed below."
        )
    return answer, merged


# Interactive CLI

HELP_TEXT = """
Commands:
  :exit              quit
  :k <number>        change number of chunks retrieved (default: 5)
  :sources on/off    toggle source display
  :sub <name>        force subsystem filter (OBC, AOCS, PAYLOAD, POWER, etc.)
  :sub off           clear subsystem filter
  :help              show this help
"""

def interactive(show_sources: bool = True) -> None:
    print("=" * 62)
    print("Aditya-L1 RAG - Step 4: Query Engine")
    print("=" * 62)
    print(f"  Generation  : Ollama ({OLLAMA_GEN_MODEL})")
    print(f"  Vector store: {VECTORSTORE_DIR}")
    print(f"  Sources     : {'on' if show_sources else 'off'}")
    print(f"  Type :help for commands, :exit to quit")
    print("=" * 62)
    print()

    top_k     = TOP_K_FINAL
    sub_force = None

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue

        # Meta-commands
        if query == ":exit":
            break
        if query == ":help":
            print(HELP_TEXT)
            continue
        if query.startswith(":k "):
            try:
                top_k = int(query.split()[1])
                print(f"  top_k -> {top_k}")
            except ValueError:
                print("  Usage: :k <integer>")
            continue
        if query.startswith(":sources "):
            show_sources = query.split()[1].lower() == "on"
            print(f"  Sources -> {'on' if show_sources else 'off'}")
            continue
        if query.startswith(":sub "):
            val = query.split()[1]
            sub_force = None if val.lower() == "off" else val.upper()
            print(f"  Subsystem filter -> {sub_force or 'none'}")
            continue

        print()
        try:
            answer, sources = ask(query, top_k=top_k,
                                  subsystem_hint=sub_force)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            print()
            continue

        print(f"Answer:\n{answer}")
        if show_sources and sources:
            print(f"\nSources:\n{format_sources(sources)}")

        # Offer to open any referenced diagram/figure images
        imgs = collect_images(sources)
        if imgs:
            print(f"\n  {len(imgs)} diagram/image(s) referenced:")
            for f in imgs:
                print(f"    {f}")
            try:
                if input("  Open them now? [y/N] ").strip().lower() == "y":
                    open_images(imgs)
            except (EOFError, KeyboardInterrupt):
                pass
        print()


# CLI entry point

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 4: Query Aditya-L1 RAG")
    ap.add_argument("query", nargs="?",
                    help="Question to ask (omit for interactive mode)")
    ap.add_argument("--k", type=int, default=TOP_K_FINAL,
                    help=f"Chunks to retrieve (default: {TOP_K_FINAL})")
    ap.add_argument("--subsystem", default=None,
                    help="Force subsystem filter: OBC, AOCS, PAYLOAD, etc.")
    ap.add_argument("--show-sources", action="store_true", default=True,
                    help="Print source citations after answer (default: on)")
    ap.add_argument("--no-sources", dest="show_sources", action="store_false")
    ap.add_argument("--open", dest="open_images", action="store_true",
                    help="Open any referenced diagram/figure images in the default viewer")
    args = ap.parse_args()

    if args.query:
        answer, sources = ask(
            args.query, top_k=args.k, subsystem_hint=args.subsystem
        )
        print(f"\n{answer}\n")
        if args.show_sources and sources:
            print(f"Sources:\n{format_sources(sources)}")

        # List (and optionally open) any referenced diagram/figure images
        imgs = collect_images(sources)
        if imgs:
            print(f"\n{len(imgs)} diagram/image(s) referenced:")
            for f in imgs:
                print(f"  {f}")
            if args.open_images:
                open_images(imgs)
            else:
                print("  (add --open to open them in your image viewer)")
    else:
        interactive(show_sources=args.show_sources)

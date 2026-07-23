#!/usr/bin/env python3
"""
Step 2: Image Captioning - Aditya-L1 RAG Pipeline
===================================================
Sends each substantive image extracted by Step 1 to a LOCAL vision model
(Ollama running Qwen2.5-VL) and generates a technical caption.  Captions are
written back into chunks.json and also stored as standalone caption chunks
ready for embedding in Step 3.  Runs fully offline - no API key, no cloud.

Inputs  (produced by Step 1):
  extracted/chunks.json      - text chunks, some with images[] lists
  extracted/images/          - image files (JPEG, PNG, WMF, EMF)

Outputs:
  extracted/chunks.json          - same file, caption fields populated
  extracted/image_captions.json  - one caption chunk per image (for embedding)

Resume:
  A hidden state file (extracted/.captioning_state.json) tracks which images
  have already been captioned.  Re-running the script is safe and skips done work.

Prerequisite:
  ollama pull qwen2.5vl        # local vision model - no API key needed

Usage:
  python step2_image_captioning.py                  # caption all valid images
  python step2_image_captioning.py --limit 20       # process at most 20 images
  python step2_image_captioning.py --dry-run        # show plan, no model calls
  python step2_image_captioning.py --input-dir /path/to/docs/

What gets skipped (logged but not sent to the model):
  - WMF / EMF vector files  (no reliable converter in this environment)
  - Images smaller than 5 KB  (spacers, icons, colour swatches)

Dependencies:
  pip install ollama pymupdf pillow
"""

import argparse
import base64
import bisect
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

# Dependency checks
try:
    import ollama
except ImportError:
    raise ImportError("Missing: pip install ollama")

try:
    import fitz  # PyMuPDF - used to pull surrounding page text for context
except ImportError:
    raise ImportError("Missing: pip install pymupdf")

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Missing: pip install pillow")


# Paths
EXTRACTED_DIR = Path("extracted")
IMAGES_DIR    = EXTRACTED_DIR / "images"
CHUNKS_FILE   = EXTRACTED_DIR / "chunks.json"
CAPTIONS_FILE = EXTRACTED_DIR / "image_captions.json"
STATE_FILE    = EXTRACTED_DIR / ".captioning_state.json"


# Model / API settings
MODEL         = "qwen2.5vl"
MAX_TOKENS    = 400        # 2-5 sentence captions; keep output tight
CALL_DELAY    = 0.6        # seconds between successful API calls
RETRY_DELAYS  = [15, 30, 60]   # back-off schedule on rate-limit errors


# Image filtering
MIN_BYTES    = 5_000       # images smaller than this are spacers / icons
MIN_PIXELS   = 60          # skip if either dimension is below this (px)
SKIP_EXTS    = {".wmf", ".emf"}   # Windows metafile: no converter available
VALID_EXTS   = {".jpg", ".jpeg", ".png"}

MEDIA_TYPES  = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
}

# Map doc_key -> source PDF filename (for page-text context extraction)
PDF_SOURCES = {
    "FCP": "Fcp_document_AL1_v1_0.pdf",
    "MOR": "AL1_MOR_DOC_V2_16Aug2023.pdf",
}


# Context helpers

def parse_image_filename(name: str) -> tuple[str, Optional[int], int]:
    """
    Decode the naming convention used by Step 1.

    PDF images  : "{DOC_KEY}_p{PAGE}_img{IDX}.{ext}"
                  e.g. "MOR_p75_img2.jpeg"
    DOCX images : "{DOC_KEY}_img{IDX}.{ext}"
                  e.g. "OPS_MANUAL_img003.png"

    Returns (doc_key, page_num_or_None, img_idx).
    """
    stem = Path(name).stem
    m = re.match(r"^(.+?)_p(\d+)_img(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(.+?)_img(\d+)$", stem)
    if m:
        return m.group(1), None, int(m.group(2))
    return stem, None, 0


def build_page_index(chunks: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    """
    Return {doc_key: [(page_num, chunk_dict), ...]} sorted by page.
    Used for nearest-chunk lookup by page number.
    """
    index: dict[str, list] = {}
    for chunk in chunks:
        key  = chunk.get("source_doc", "")
        page = chunk.get("page")
        if key and page:
            index.setdefault(key, []).append((page, chunk))
    for key in index:
        index[key].sort(key=lambda t: t[0])
    return index


def nearest_chunk(
    page_index: dict,
    doc_key: str,
    page_num: Optional[int],
) -> Optional[dict]:
    """
    Return the chunk from doc_key whose page number is closest to page_num.
    Falls back to the first chunk for that doc if page_num is None.
    """
    entries = page_index.get(doc_key)
    if not entries:
        return None
    if page_num is None:
        return entries[0][1]
    pages = [e[0] for e in entries]
    idx   = bisect.bisect_left(pages, page_num)
    idx   = max(0, min(idx, len(entries) - 1))
    return entries[idx][1]


def get_page_text(doc_key: str, page_num: int, input_dir: Path) -> str:
    """
    Extract plain text from a specific PDF page (for captioning context).
    Returns an empty string if the PDF can't be opened or page doesn't exist.
    """
    pdf_name = PDF_SOURCES.get(doc_key)
    if not pdf_name:
        return ""
    pdf_path = input_dir / pdf_name
    if not pdf_path.exists():
        return ""
    try:
        doc  = fitz.open(str(pdf_path))
        page = doc[page_num - 1]       # fitz uses 0-based indexing
        text = page.get_text("text").strip()
        doc.close()
        # Trim to ~600 chars so the prompt stays concise
        return text[:600] + ("..." if len(text) > 600 else "")
    except Exception:
        return ""


# Prompt builder

SYSTEM_PROMPT = (
    "You are a technical documentation analyst for the Aditya-L1 solar "
    "observation spacecraft mission operated by ISRO. You will be shown images "
    "extracted from mission planning and operations documents. Generate precise, "
    "keyword-rich technical captions that would allow an engineer to locate and "
    "understand the image through semantic search."
)


def build_user_prompt(chunk: Optional[dict], page_text: str) -> str:
    """
    Build a context-injected caption request from the nearest chunk's metadata
    and the raw text from the same PDF page.
    """
    ctx_lines = []

    if chunk:
        if chunk.get("doc_filename"):
            ctx_lines.append(f"Document  : {chunk['doc_filename']}")
        if chunk.get("source_doc"):
            ctx_lines.append(f"Doc key   : {chunk['source_doc']}")
        if chunk.get("page"):
            ctx_lines.append(f"Page      : {chunk['page']}")
        if chunk.get("section_heading"):
            ctx_lines.append(f"Section   : {chunk['section_heading']}")
        if chunk.get("fcp_number"):
            ctx_lines.append(f"FCP       : {chunk['fcp_number']}")
        if chunk.get("subsystem"):
            ctx_lines.append(f"Subsystem : {chunk['subsystem']}")

    if page_text:
        ctx_lines.append(f"\nSurrounding page text (excerpt):\n{page_text}")

    context_block = "\n".join(ctx_lines) if ctx_lines else "(no context available)"

    return f"""Context from source document:
        {context_block}

        You are an ISRO spacecraft documentation analyst.

        Write a technical caption optimized for semantic search and RAG retrieval.

        Include:
        1. Figure type (block diagram, orbit plot, telemetry chart, command table, photograph, flow chart, etc.)
        2. Mission phase or operation being shown
        3. Subsystem involved (if identifiable)
        4. Important labels, identifiers, CID codes, parameter names, and numerical values visible
        5. Operational or engineering significance

        Requirements:
        - Use aerospace and spacecraft operations terminology.
        - Preserve technical terms exactly as shown.
        - Mention important acronyms.
        - Be factual and concise.
        - Maximum 5 sentences.
        - Do not speculate beyond the provided context.

        Caption:
    """


# Image preparation

class SkipImage(Exception):
    """Raised when an image should be skipped (not an API error)."""


def prepare_image(image_path: Path) -> tuple[str, str]:
    """
    Load an image file, validate it, normalise to PNG, and return
    (base64_data, media_type). Used to gate which images are worth captioning.

    Raises SkipImage with a reason string if the image should be skipped.
    """
    ext  = image_path.suffix.lower()
    size = image_path.stat().st_size

    if ext in SKIP_EXTS:
        raise SkipImage(f"unsupported vector format ({ext})")

    if ext not in VALID_EXTS:
        raise SkipImage(f"unsupported extension ({ext})")

    if size < MIN_BYTES:
        raise SkipImage(f"too small ({size:,} bytes < {MIN_BYTES:,})")

    # Validate dimensions with Pillow
    try:
        img = PILImage.open(image_path)
        w, h = img.size
        if w < MIN_PIXELS or h < MIN_PIXELS:
            raise SkipImage(f"too small ({w}×{h} px)")
    except SkipImage:
        raise
    except Exception as exc:
        raise SkipImage(f"Pillow could not open image: {exc}")

    # Normalise to PNG so we always send a consistent media type.
    # This avoids edge-cases with progressive JPEG or unusual colour spaces.
    buf = io.BytesIO()
    try:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buf, format="PNG")
    except Exception as exc:
        raise SkipImage(f"normalisation to PNG failed: {exc}")

    return base64.b64encode(buf.getvalue()).decode(), "image/png"


# API call

def call_qwen(image_path, user_prompt):
    response = ollama.chat(
        model="qwen2.5vl",
        options={"num_ctx": 8192},
        messages=[
            {
                "role": "user",
                "content": user_prompt,
                "images": [str(image_path)]
            }
        ]
    )

    return response["message"]["content"]


# State management (resume)

def load_state() -> dict:
    """Load the captioning-state file (maps image filename -> caption text)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# Output helpers

def make_caption_chunk(
    image_path: Path,
    caption: str,
    chunk: Optional[dict],
) -> dict:
    """
    Build a standalone caption chunk in the same schema as Step 1 text chunks,
    so the embedding step can treat text and caption chunks identically.
    """
    doc_key, page_num, img_idx = parse_image_filename(image_path.name)
    return {
        "chunk_id":        f"IMG_{image_path.stem}",
        "text":            caption,
        "source_doc":      doc_key,
        "doc_filename":    (chunk or {}).get("doc_filename", ""),
        "content_type":    "image_caption",
        "page":            page_num,
        "section_heading": (chunk or {}).get("section_heading", ""),
        "fcp_number":      (chunk or {}).get("fcp_number", ""),
        "subsystem":       (chunk or {}).get("subsystem", "UNKNOWN"),
        "image_file":      str(image_path),
        "images":          [],
    }


def update_chunks_with_captions(
    chunks: list[dict],
    state: dict,
) -> None:
    """
    Write captions from the state dict back into the images[] lists of every
    chunk that references a now-captioned image.
    """
    for chunk in chunks:
        for img_rec in chunk.get("images", []):
            fname = Path(img_rec["file"]).name
            if fname in state:
                img_rec["caption"] = state[fname]


# Main orchestrator

def run_captioning(
    input_dir: str = ".",
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """
    Caption all valid images in extracted/images/ using the local Ollama
    vision model (Qwen2.5-VL).

    Args:
        input_dir : folder containing the original source documents
                    (needed to open PDFs for page-text context)
        limit     : if set, stop after this many model calls (useful for testing)
        dry_run   : print the plan but do not call the model
    """
    src = Path(input_dir)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"{IMAGES_DIR} not found.  Run step1_extraction.py first."
        )
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found.  Run step1_extraction.py first."
        )

    # Load state and chunks
    state  = load_state()
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    pg_idx = build_page_index(chunks)

    # Build image work-list
    all_images = sorted(IMAGES_DIR.iterdir())

    print("=" * 62)
    print("Aditya-L1 RAG - Step 2: Image Captioning")
    print("=" * 62)
    print(f"  Images found  : {len(all_images)}")
    print(f"  Already done  : {len(state)} (from previous runs)")
    print(f"  Model         : {MODEL}")
    print(f"  Dry run       : {dry_run}")
    if limit:
        print(f"  Limit         : {limit}")
    print()

    # Caption loop
    caption_chunks: list[dict] = []
    n_done = n_skipped = n_errors = n_api_calls = 0

    for image_path in all_images:
        fname = image_path.name

        # Already captioned in a previous run?
        if fname in state:
            caption_chunks.append(
                make_caption_chunk(
                    image_path,
                    state[fname],
                    nearest_chunk(pg_idx, *parse_image_filename(fname)[:2]),
                )
            )
            n_done += 1
            continue

        # Prepare image bytes (validates size, format, dimensions)
        try:
            prepare_image(image_path)
        except SkipImage as reason:
            print(f"  SKIP  {fname}  ({reason})")
            n_skipped += 1
            continue

        # Find nearest chunk for context
        doc_key, page_num, _ = parse_image_filename(fname)
        chunk    = nearest_chunk(pg_idx, doc_key, page_num)
        pg_text  = get_page_text(doc_key, page_num, src) if page_num else ""
        prompt   = build_user_prompt(chunk, pg_text)

        if dry_run:
            print(f"  WOULD caption  {fname}  [{doc_key}, p{page_num}]")
            n_done += 1
            if limit and n_done >= limit:
                break
            continue

        # API call
        try:
            caption = call_qwen(
                image_path,
                prompt
            )
        except Exception as exc:
            print(f"  ERROR {fname}: {exc}")
            n_errors += 1
            continue

        n_api_calls += 1
        print(f"  [{n_api_calls:3d}] {fname}")
        print(f"        -> {caption[:90]}{'...' if len(caption) > 90 else ''}")

        # Persist immediately so a crash doesn't lose progress
        state[fname] = caption
        save_state(state)

        caption_chunks.append(make_caption_chunk(image_path, caption, chunk))

        if limit and n_api_calls >= limit:
            print(f"\n  Reached --limit {limit}. Stopping.")
            break

        time.sleep(CALL_DELAY)

    # Write outputs
    if not dry_run:
        # Update captions inside chunks.json
        update_chunks_with_captions(chunks, state)
        CHUNKS_FILE.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Write standalone caption chunks
        CAPTIONS_FILE.write_text(
            json.dumps(caption_chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Summary
    total_captioned = len([v for v in state.values() if v])
    print()
    print("=" * 62)
    print("Step 2 complete")
    print("=" * 62)
    print(f"  API calls this run  : {n_api_calls}")
    print(f"  Total captioned     : {total_captioned}  (cumulative across runs)")
    print(f"  Skipped             : {n_skipped}  (too small / unsupported format)")
    print(f"  Errors              : {n_errors}")
    print()
    if not dry_run:
        print(f"  {CHUNKS_FILE}  ← captions written back into images[] lists")
        print(f"  {CAPTIONS_FILE}  ← {len(caption_chunks)} caption chunks for embedding")
        print()
    print("Next step: run step3_embedding.py to build the vector store")
    print("=" * 62)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Caption Aditya-L1 document images with a local Ollama vision model (Qwen2.5-VL)"
    )
    parser.add_argument(
        "--input-dir", default=".",
        help="Directory containing the original source documents (default: .)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after this many API calls (useful for testing / cost control)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show which images would be captioned without calling the API"
    )
    args = parser.parse_args()
    run_captioning(
        input_dir=args.input_dir,
        limit=args.limit,
        dry_run=args.dry_run,
    )

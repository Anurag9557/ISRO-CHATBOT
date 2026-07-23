#!/usr/bin/env python3
"""
Step 1: read the five source documents and turn them into JSON.

Writes:
  extracted/chunks.json       text chunks from FCP, MOR, OPS_MANUAL, GUIDELINES
  extracted/tch_records.json  structured command records from the TCH log
  extracted/images/           images pulled out of the PDFs and DOCX files

Usage:
  python step1_extraction.py                 # docs in the current folder
  python step1_extraction.py /path/to/docs/  # or point it at a folder

Needs pymupdf and python-docx (pip install), plus LibreOffice for the .doc file.
"""
import subprocess
import shutil
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Dependency check
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Missing: pip install pymupdf")

try:
    from docx import Document as DocxDocument
except ImportError:
    raise ImportError("Missing: pip install python-docx")


# Document registry
DOCS = {
    "FCP":        "Fcp_document_AL1_v1_0.pdf",
    "MOR":        "AL1_MOR_DOC_V2_16Aug2023.pdf",
    "OPS_MANUAL": "AL1OperationsManual_latest.docx",
    "GUIDELINES": "AL1_NormalPhaseGuideLines_16Dec2024__3_.doc",
    "TCH":        "Tch_1_.txt",
}

OUTPUT_DIR  = Path("extracted")
IMAGES_DIR  = OUTPUT_DIR / "images"
CHUNKS_FILE = OUTPUT_DIR / "chunks.json"
TCH_FILE    = OUTPUT_DIR / "tch_records.json"


# Data model
@dataclass
class ImageRecord:
    file: str
    page: int
    index_on_page: int
    caption: str = ""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    doc_filename: str
    content_type: str
    page: Optional[int] = None
    section_heading: str = ""
    fcp_number: str = ""
    subsystem: str = ""
    images: list = field(default_factory=list)


# Helpers

def convert_vector_to_png(image_path: Path):
    if image_path.suffix.lower() not in [".emf", ".wmf"]:
        return image_path
    magick = (
        shutil.which("magick")
        or r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    )
    png_path = image_path.with_suffix(".png")
    try:
        subprocess.run(
            [magick, str(image_path), str(png_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if png_path.exists():
            image_path.unlink(missing_ok=True)
            return png_path
    except Exception as e:
        print(f"  Conversion failed: {e}")
    return image_path


def extract_fcp_number(text: str) -> str:
    m = re.search(r'\b([56]\d{3})\b', text)
    return m.group(1) if m else ""


# FCP table-of-contents parsing

# Canonical subsystem names, keyed by the raw label used in the FCP TOC column.
# Includes the "PAYLAOD" misspelling that appears several times in the source.
TOC_SUBSYS_MAP = {
    "OBC": "OBC",
    "SENSOR": "SENSOR",
    "POWER": "POWER",
    "TTC&XBAND": "TTC_XBAND",
    "TTCXBAND": "TTC_XBAND",
    "DTG": "DTG",
    "WHEEL": "WHEEL",
    "MECHANISM": "MECHANISM",
    "PROPULSION": "PROPULSION",
    "AOCS": "AOCS",
    "PAYLOAD": "PAYLOAD",
    "PAYLAOD": "PAYLOAD",     # common misspelling in the source document
    "ODHS": "ODHS",
}

# A TOC row anchor:  "<serial>.  <FCP no>  <SUBSYSTEM>"
# e.g. "1.  5201  OBC ...", "63.  6006  TTC&XBAND ...", "162.  6160  \n PAYLAOD ..."
# \s+ spans the newline that occasionally separates the number from the subsystem.
TOC_ROW_RE = re.compile(r'(\d{1,3})\s*\.\s+([56]\d{3})\s+([A-Z][A-Z&]{1,})')


def parse_fcp_toc(toc_text: str, pdf_name: str) -> list:
    """
    Parse the FCP Table of Contents into one small index record per procedure.

    Each record maps FCP number -> subsystem -> title, so the RAG can answer
    "which FCP is used for X?" for ALL listed procedures, even the ones whose
    full body is not present in this excerpt file.

    Returns a list of Chunk dicts with content_type="toc_index".
    """
    matches = list(TOC_ROW_RE.finditer(toc_text))
    records: list = []
    seen_ids: set = set()

    for i, m in enumerate(matches):
        serial   = m.group(1)
        fcp      = m.group(2)
        raw_sub  = m.group(3).upper()
        subsystem = TOC_SUBSYS_MAP.get(raw_sub, "UNKNOWN")

        # Title = text between this anchor and the next row anchor (or end).
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(toc_text)
        title = re.sub(r'\s+', ' ', toc_text[start:end]).strip(" .\n\t")

        # Unique id even when the source reuses an FCP number (6201, 6232 appear
        # twice); serial number disambiguates.
        chunk_id = f"FCP_IDX_{int(serial):03d}_{fcp}"
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        records.append(asdict(Chunk(
            chunk_id        = chunk_id,
            text            = f"FCP {fcp} - {subsystem} - {title}",
            source_doc      = "FCP",
            doc_filename    = pdf_name,
            content_type    = "toc_index",
            page            = 1,
            section_heading = "FCP Index",
            fcp_number      = fcp,
            subsystem       = subsystem,
        )))

    return records


def infer_subsystem(text: str, fcp_number: str = "") -> str:
    keyword_map = [
        ("OBC",        r'\bOBC\b'),
        ("AOCS",       r'\bAOCS\b'),
        ("DTG",        r'\bDTG\b|\bGYRO\b|\bCSA\b'),
        ("SENSOR",     r'\bSTAR SENSOR\b'),
        ("WHEEL",      r'\bWHEEL\b|\bWDE\b|\bSUSPENSION\b'),
        ("MECHANISM",  r'\bMECHANISM\b|\bHGA\b|\bSPDM\b|\bMCE\b|\bVELC COVER\b|\bSOLEX COVER\b'),
        ("PROPULSION", r'\bPROPULSION\b|\bLATCH VALVE\b|\bTHRUSTER\b|\bFUEL\b|\bOXIDISER\b'),
        ("POWER",      r'\bFPGA\b|\bBATTERY\b|\bTCR\b|\bDC-DC\b'),
        ("TTC_XBAND",  r'\bTTC\b|\bX-BAND\b|\bTRANSPONDER\b|\bTWTA\b|\bXBS\b'),
        ("PAYLOAD",    r'\bPAPA\b|\bASPEX\b|\bSOLEXS\b|\bHELIOS\b|\bSUIT\b|\bVELC\b|\bMAGNETOMETER\b'),
        ("ODHS",       r'\bODHS\b|\bSSR\b|\bDFU\b|\bBDH\b'),
    ]
    for subsystem, pattern in keyword_map:
        if re.search(pattern, text, re.IGNORECASE):
            return subsystem
    if fcp_number:
        try:
            n = int(fcp_number)
            ranges = [
                (5000, 5099, "AOCS"),   (5200, 5299, "OBC"),
                (5400, 5499, "SENSOR"), (5551, 5560, "WHEEL"),
                (5601, 5620, "DTG"),    (5650, 5665, "PROPULSION"),
                (5700, 5730, "POWER"),  (5950, 5980, "MECHANISM"),
                (6000, 6050, "TTC_XBAND"), (6100, 6250, "PAYLOAD"),
                (6220, 6240, "ODHS"),
            ]
            for lo, hi, sub in ranges:
                if lo <= n <= hi:
                    return sub
        except ValueError:
            pass
    return "UNKNOWN"


def detect_content_type(text: str) -> str:
    # A short procedure and its trailing NOTES section can land in one chunk,
    # so check for procedure signals first and only fall back to "note".
    has_procedure = (
        re.search(r'\bPROCEDURE\s*:', text, re.IGNORECASE)
        or re.search(r'\bEND OF PROCEDURE\b', text, re.IGNORECASE)
        or re.search(r'\bAPPLICABILITY\s*:', text, re.IGNORECASE)
        or re.search(r'\b[56]\d{3}\s*:', text)          # body header e.g. "5201:"
        or re.search(r'\b[56]\d{3}0\d{2}\b', text)       # step ref e.g. "5201001"
    )
    if has_procedure:
        return "procedure"
    if re.search(r'\bNOTES?\s*:', text, re.IGNORECASE):
        return "note"
    return "section"


def cid_prefix_to_subsystem(cid: str) -> str:
    prefix_map = {
        "OBCS": "OBC", "OBCD": "OBC", "OBCA": "OBC",
        "DTGS": "DTG", "WDEA": "WHEEL", "PWRS": "POWER",
        "MECS": "MECHANISM", "TTCS": "TTC_XBAND", "PROS": "PROPULSION",
        "AOCS": "AOCS", "PLDA": "PAYLOAD", "ODHS": "ODHS", "SENS": "SENSOR",
    }
    for prefix, subsystem in prefix_map.items():
        if cid.upper().startswith(prefix):
            return subsystem
    return "UNKNOWN"


# PDF extractor

def extract_pdf(doc_key: str, pdf_path: Path, chunks: list):
    print(f"\n[PDF] {pdf_path.name}")
    doc = fitz.open(str(pdf_path))

    def split_text(text, size=1000):
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        result, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < size:
                current += "\n" + para
            else:
                if current:
                    result.append(current.strip())
                current = para
        if current:
            result.append(current.strip())
        return result

    # MOR: section-based extraction
    if doc_key == "MOR":
        heading_re = re.compile(
            r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Z0-9\s\-/(),]{5,})$',
            re.MULTILINE
        )
        full_text = ""
        page_map  = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            page_map.append((len(full_text), page_num))
            full_text += "\n" + text

            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_img  = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    img_ext   = base_img["ext"]
                    fname     = f"{doc_key}_p{page_num}_img{img_idx}.{img_ext}"
                    fpath     = IMAGES_DIR / fname
                    fpath.write_bytes(img_bytes)
                    fpath     = convert_vector_to_png(fpath)
                    print(f"  + image: {fpath.name}")
                except Exception as exc:
                    print(f"  ! image skipped: {exc}")

        matches = list(heading_re.finditer(full_text))

        for i, match in enumerate(matches):           # i is unique per section
            start = match.start()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            section = full_text[start:end].strip()
            heading = match.group(0).strip()
            title   = heading.split(maxsplit=1)[1] if " " in heading else ""

            if ("|" in heading or "Page" in heading or "P a g e" in heading
                    or len(title) < 5 or title.replace(" ", "").isdigit()):
                continue

            page = 1
            for pos, p in page_map:
                if pos <= start:
                    page = p

            for idx, chunk in enumerate(split_text(section)):
                # key the id on the section index so two sections on the same
                # page don't collide
                chunks.append(asdict(Chunk(
                    chunk_id        = f"MOR_s{i}_c{idx}",
                    text            = chunk,
                    source_doc      = "MOR",
                    doc_filename    = pdf_path.name,
                    content_type    = "section",
                    page            = page,
                    section_heading = heading,
                    subsystem       = infer_subsystem(chunk),
                )))

        doc.close()
        print(f"  -> {sum(1 for c in chunks if c['source_doc']=='MOR')} chunks")
        return

    # FCP (and any other PDF): procedure-block extraction

    # Diagnostic: show how many pages have readable text vs image-only
    # This immediately tells us if OCR is needed
    text_counts = [(len(doc[i].get_text().strip()), len(doc[i].get_images()))
                   for i in range(len(doc))]
    text_pages = sum(1 for chars, _ in text_counts if chars > 50)
    img_only   = len(doc) - text_pages
    print(f"  Pages: {len(doc)} total | {text_pages} with text | {img_only} image-only")
    if img_only > 0:
        print(f"  First 20 pages breakdown:")
        for i, (chars, imgs) in enumerate(text_counts[:20], start=1):
            label = f"{chars:>6} chars" if chars > 50 else "IMAGE ONLY"
            print(f"    Page {i:>3}: {label}  ({imgs} embedded images)")
    if img_only > len(doc) * 0.3:
        print(f"  ⚠  {img_only} image-only pages - OCR needed for full FCP coverage")

    # The FCP file opens with a Table of Contents listing every procedure
    # (FCP number -> subsystem -> title), then gives the full body for only a
    # subset. We parse the TOC region into one index record per procedure so
    # the RAG can answer "which FCP is for X?" for ALL listed procedures, and
    # we skip the TOC pages in the body loop below so the old giant TOC blob
    # (which used to be mis-stamped with fcp_number 5201) is never emitted.
    body_start_page = 1
    if doc_key == "FCP":
        body_probe = re.compile(r'\b[A-Z]+\s*-?\s*[56]\d{3}\s*:\s*.+')
        toc_parts: list[str] = []
        for i in range(len(doc)):
            ptext = doc[i].get_text("text")
            if body_probe.search(ptext):
                body_start_page = i + 1        # 1-based page where bodies start
                break
            toc_parts.append(ptext)
        else:
            body_start_page = len(doc) + 1     # no bodies found - whole file is TOC

        toc_records = parse_fcp_toc("\n".join(toc_parts), pdf_path.name)
        chunks.extend(toc_records)
        print(f"  TOC index    : {len(toc_records)} procedures indexed "
              f"(bodies begin on page {body_start_page})")

    # Chunk state
    current_pages:    list[str]       = []
    current_images:   list[ImageRecord] = []
    current_fcp:      str             = ""
    current_heading:  str             = ""
    chunk_start_page: int             = 1
    fcp_chunk_counter: list[int]      = [0]  # stable counter, not len(chunks)

    fcp_header_re = re.compile(
        r'\b([A-Z]+)\s*-?\s*([56]\d{3})\s*:\s*(.+)',
        re.IGNORECASE
    )

    def flush(end_page: int):
        nonlocal current_pages, current_images, current_fcp, current_heading
        text = "\n".join(current_pages).strip()
        if not text:
            return
        fcp   = current_fcp or extract_fcp_number(text)
        sub   = infer_subsystem(current_heading + " " + text, fcp)
        ctype = detect_content_type(text)
        chunks.append(asdict(Chunk(
            chunk_id        = f"{doc_key}_p{chunk_start_page}_{fcp_chunk_counter[0]}",
            text            = text,
            source_doc      = doc_key,
            doc_filename    = pdf_path.name,
            content_type    = ctype,
            page            = chunk_start_page,
            section_heading = current_heading,
            fcp_number      = fcp,
            subsystem       = sub,
            images          = [asdict(i) for i in current_images],
        )))
        fcp_chunk_counter[0] += 1
        current_pages  = []
        current_images = []

    for page_num, page in enumerate(doc, start=1):
        if page_num < body_start_page:
            continue                    # skip TOC pages (already indexed above)
        text = page.get_text("text")

        header_match = fcp_header_re.search(text)
        if header_match and current_pages:
            flush(page_num - 1)
            chunk_start_page = page_num
            current_fcp      = header_match.group(2)
            current_heading  = header_match.group(0).strip()
        elif not current_pages:
            chunk_start_page = page_num
            if header_match:
                current_fcp     = header_match.group(2)
                current_heading = header_match.group(0).strip()

        current_pages.append(text)

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]
                img_ext   = base_img["ext"]
                fname     = f"{doc_key}_p{page_num}_img{img_idx}.{img_ext}"
                fpath     = IMAGES_DIR / fname
                fpath.write_bytes(img_bytes)
                current_images.append(ImageRecord(
                    file=str(fpath), page=page_num, index_on_page=img_idx,
                ))
                print(f"  + image: {fname}")
            except Exception as exc:
                print(f"  ! image xref={xref} skipped: {exc}")

        if "END OF PROCEDURE" in text and current_pages:
            flush(page_num)
            chunk_start_page = page_num + 1
            # don't reset current_fcp / current_heading here: notes pages that
            # follow END OF PROCEDURE belong to the same FCP. The reset happens
            # on its own when the next header is detected.

    flush(len(doc))
    doc.close()
    n = sum(1 for c in chunks if c["source_doc"] == doc_key)
    print(f"  -> {n} chunks")


# DOCX extractor

def extract_docx(doc_key: str, docx_path: Path, chunks: list):
    print(f"\n[DOCX] {docx_path.name}")
    doc = DocxDocument(str(docx_path))

    current_heading = "Preamble"
    current_paras:  list[str] = []
    chunk_num = 0

    def flush_docx():
        nonlocal chunk_num, current_paras, current_heading
        text = "\n".join(current_paras).strip()
        if not text:
            current_paras = []
            return
        fcp = extract_fcp_number(current_heading + " " + text)
        sub = infer_subsystem(current_heading + " " + text, fcp)
        chunks.append(asdict(Chunk(
            chunk_id        = f"{doc_key}_{chunk_num:04d}",
            text            = text,
            source_doc      = doc_key,
            doc_filename    = docx_path.name,
            content_type    = "section",
            section_heading = current_heading,
            fcp_number      = fcp,
            subsystem       = sub,
        )))
        chunk_num    += 1
        current_paras = []

    for para in doc.paragraphs:
        style_name = ""
        try:
            if para.style is not None:
                style_name = para.style.name or ""
        except Exception:
            style_name = ""

        if style_name.startswith("Heading"):
            flush_docx()
            current_heading = para.text.strip()
        elif para.text.strip():
            current_paras.append(para.text.strip())

    flush_docx()

    img_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            try:
                blob     = rel.target_part.blob
                ext      = rel.target_ref.rsplit(".", 1)[-1].lower()
                fname    = f"{doc_key}_img{img_count:03d}.{ext}"
                img_path = IMAGES_DIR / fname
                img_path.write_bytes(blob)
                img_path = convert_vector_to_png(img_path)
                img_count += 1
                print(f"  + image: {img_path.name}")
            except Exception as exc:
                print(f"  ! image skipped: {exc}")

    print(f"  -> {chunk_num} chunks, {img_count} images")


# DOC extractor (via LibreOffice)

def extract_doc(doc_key: str, doc_path: Path, chunks: list):
    print(f"\n[DOC] {doc_path.name}")
    converted = OUTPUT_DIR / (doc_path.stem + "_converted.docx")

    if converted.exists():
        print(f"  Using cached: {converted.name}")
    else:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            print("  ERROR: LibreOffice not found. Install it and re-run.")
            return
        print(f"  Converting with LibreOffice...")
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "docx",
             "--outdir", str(OUTPUT_DIR), str(doc_path)],
            capture_output=True, text=True, timeout=120,
        )
        lo_output = OUTPUT_DIR / (doc_path.stem + ".docx")
        if lo_output.exists():
            lo_output.rename(converted)
            print(f"  Converted -> {converted.name}")
        else:
            print(f"  ERROR: Conversion failed.\n  {result.stderr}")
            return

    extract_docx(doc_key, converted, chunks)


# TCH extractor

def extract_tch(doc_key: str, tch_path: Path, tch_records: list):
    print(f"\n[TCH] {tch_path.name}")
    raw   = tch_path.read_bytes().decode("latin-1")

    # A few command rows carry raw binary payloads whose bytes can include a
    # stray newline, which would split the record across two physical lines.
    # Grouping the text at each "NNNN |" row start keeps every record whole.
    _starts = [m.start() for m in re.finditer(r'(?m)^\s{0,2}\d{4}\s*\|', raw)]
    records = [raw[_starts[i]:(_starts[i + 1] if i + 1 < len(_starts) else len(raw))]
               for i in range(len(_starts))]

    row_re = re.compile(
        r'^\s{0,2}(\d{4})\s*\|\s*'
        r'([\w\-]+|-{4,})\s*\|\s*'
        r'([A-Z][A-Z0-9]{2,6}\d+|-{4,})\s*\|\s*'
        r'([^\|]{10,50}?)\s*\|\s*'
        r'(\d{4} \d{2} \d{2} \d{2}:\d{2}:\d{2}:\d{3})\s*\|'
        r'(.*?)\s*\|\s*'
        r'(.*?)\s*\|\s*'
        r'(\d{3,6})\s*\|\s*'
        r'(\w+)\s*\|\s*'
        r'(\w+)\s*\|\s*'
        r'(\w+)\s*\|',
        re.DOTALL,
    )

    count = 0
    for record in records:
        m = row_re.match(record)
        if not m:
            continue
        (sl_no, code, cid, mnemonic, cmd_time,
         cmd_data, data_val, orbit_no, status, src_stn, dst_stn) = m.groups()

        dt = cmd_time.strip()
        try:
            date_str = f"{dt[0:4]}-{dt[5:7]}-{dt[8:10]}"
            time_str = dt[11:19]
            ms_str   = dt[20:23]
        except IndexError:
            date_str = time_str = ms_str = ""

        safe_data = re.sub(
            r'[^\x20-\x7e]', lambda b: f'\\x{ord(b.group()):02x}', cmd_data
        )
        subsystem = cid_prefix_to_subsystem(cid.strip())

        tch_records.append({
            "record_id":     f"TCH_{sl_no.strip()}",
            "source_doc":    doc_key,
            "doc_filename":  tch_path.name,
            "content_type":  "command_log",
            "sl_no":         sl_no.strip(),
            "code":          code.strip(),
            "cid":           cid.strip(),
            "mnemonic":      mnemonic.strip(),
            "command_time":  cmd_time.strip(),
            "date":          date_str,
            "time":          time_str,
            "milliseconds":  ms_str,
            "command_data":  safe_data.strip(),
            "data_cmd_value": data_val.strip(),
            "orbit_no":      orbit_no.strip(),
            "status":        status.strip(),
            "source_station": src_stn.strip(),
            "dest_station":  dst_stn.strip(),
            "subsystem":     subsystem,
            "text": (
                f"Command {mnemonic.strip()} (CID: {cid.strip()}) was uplinked on "
                f"{date_str} at {time_str} UTC from {src_stn.strip()} to {dst_stn.strip()}. "
                f"Orbit number: {orbit_no.strip()}. Status: {status.strip()}. "
                f"Subsystem: {subsystem}."
            ),
        })
        count += 1

    print(f"  -> {count} command records")


# Main

def run_extraction(input_dir: str = "."):
    src = Path(input_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    all_chunks:  list[dict] = []
    tch_records: list[dict] = []

    print("=" * 60)
    print("Aditya-L1 RAG - Step 1: Extraction")
    print("=" * 60)

    for doc_key, filename in DOCS.items():
        fpath = src / filename
        if not fpath.exists():
            print(f"\n[SKIP] {filename} - not found at {fpath}")
            continue

        if doc_key == "TCH":
            extract_tch(doc_key, fpath, tch_records)
        elif filename.endswith(".pdf"):
            extract_pdf(doc_key, fpath, all_chunks)
        elif filename.endswith(".docx"):
            extract_docx(doc_key, fpath, all_chunks)
        elif filename.endswith(".doc"):
            extract_doc(doc_key, fpath, all_chunks)

    CHUNKS_FILE.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    TCH_FILE.write_text(
        json.dumps(tch_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    img_files = list(IMAGES_DIR.iterdir())
    print()
    print("=" * 60)
    print("Extraction complete")
    print("=" * 60)
    by_doc = {}
    for c in all_chunks:
        by_doc[c["source_doc"]] = by_doc.get(c["source_doc"], 0) + 1
    for k, v in by_doc.items():
        print(f"  {k:<14} {v:>4} text chunks")
    print(f"  {'TCH':<14} {len(tch_records):>4} command records")
    print(f"  {'Images':<14} {len(img_files):>4} files -> {IMAGES_DIR}")
    print()
    print("Next: run step2_image_captioning.py")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    run_extraction(sys.argv[1] if len(sys.argv) > 1 else ".")

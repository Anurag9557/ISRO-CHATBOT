#!/usr/bin/env python3
"""
Step 5: check how well retrieval works, against a labelled set of questions.

Reports Hit@1/3/5/10 (was a relevant chunk in the top-k) and MRR (how high the
first relevant chunk ranked). It imports the retrieval code from step4_query so
it measures the real system, and it only needs the vector store + BGE-M3 (no
Ollama), so it runs in a few seconds.

Usage:
    python evaluate.py            # print the table
    python evaluate.py --html     # also write eval_report.html
    python evaluate.py --json     # also write eval_results.json
    python evaluate.py --answers  # also generate answers per question (needs Ollama)

Each question below lists one or more `accept` conditions; a retrieved chunk
counts as relevant if it satisfies any of them. A condition matches when all its
keys match - metadata keys (fcp_number, subsystem, source_doc, date,
content_type, page) match exactly, and text_contains / text_contains_any check
for substrings in the chunk text.
"""

import argparse
import html
import json
import sys
from pathlib import Path

try:
    import step4_query as q
except Exception as exc:
    sys.exit(f"Could not import step4_query.py (run this in the same folder): {exc}")


# Labelled question set  (extend freely - this is your thesis test bench)
# Each entry:  id, question, accept=[condition, ...], note
# A retrieved chunk is relevant if it satisfies ANY condition in `accept`.

TEST_SET = [
    # FCP procedures whose full body IS in the file
    {"id": "obc_on",      "q": "What is the procedure for switching on OBC?",
     "accept": [{"fcp_number": "5201"}], "note": "FCP-5201 body present"},
    {"id": "obc_reset",   "q": "How do I perform an OBC system reset?",
     "accept": [{"fcp_number": "5203"}], "note": "FCP-5203 body present"},
    {"id": "obc_cpu",     "q": "OBC CPU reset procedure",
     "accept": [{"fcp_number": "5204"}], "note": "FCP-5204 body present"},
    {"id": "tc_link",     "q": "telecommand link test procedure",
     "accept": [{"fcp_number": "5215"}], "note": "FCP-5215 body present"},

    # "Which FCP for X?" - answered by the parsed TOC index (225 records)
    {"id": "hga_motors",  "q": "Which FCP is used to switch on HGA motors?",
     "accept": [{"fcp_number": "5957"}], "note": "TOC index record"},
    {"id": "tone_rng",    "q": "tone ranging of transponder-1 procedure",
     "accept": [{"fcp_number": "6009"}], "note": "TOC index record"},
    {"id": "star_ss1",    "q": "powering on star sensor-1",
     "accept": [{"fcp_number": "5412"}], "note": "TOC index record"},
    {"id": "gyro_on",     "q": "gyro power on, off and selection",
     "accept": [{"fcp_number": "5601"}], "note": "TOC index record"},
    {"id": "latch_valve", "q": "opening and closing of latch valves",
     "accept": [{"fcp_number": "5652"}], "note": "TOC index record"},
    {"id": "suit_on",     "q": "SUIT payload on procedure",
     "accept": [{"fcp_number": "6219"}], "note": "TOC index record"},

    # Concept questions spanning MOR / Guidelines / Ops Manual
    {"id": "safety",      "q": "What are the safety logics?",
     "accept": [{"fcp_number": "5245"},
                {"text_contains_any": ["safe mode", "safety logic", "fdi", "fdir"]}],
     "note": "power safety logics FCP or safe-mode text"},
    {"id": "payload_gl",  "q": "What are the guidelines for Payload operations?",
     "accept": [{"subsystem": "PAYLOAD",
                 "text_contains_any": ["payload sequencer", "pop", "mep", "proposal"]},
                {"text_contains_any": ["payload sequencer", "master mep"]}],
     "note": "Ops Manual / Guidelines payload section"},
    {"id": "orbit",       "q": "What are the Aditya-L1 orbit constraints?",
     "accept": [{"source_doc": "MOR",
                 "text_contains_any": ["halo", "sev", "sun earth", "orbit"]}],
     "note": "MOR halo-orbit design/selection"},

    # Image / diagram retrieval (caption chunks)
    {"id": "mrs_diagram", "q": "Provide me the MRS recovery flow diagram",
     "accept": [{"text_contains_any": ["master recovery", "safe mode recovery", "mrs"]},
                {"content_type": "image_caption", "source_doc": "MOR"}],
     "note": "MRS block diagram / SafeModeRecovery flow"},

    # Command-log query (commands collection, exact date filter)
    {"id": "cmds_date",   "q": "What commands were uplinked on 2026-06-09?",
     "accept": [{"date": "2026-06-09"}], "note": "TCH date filter -> commands store"},
]


# Relevance matching

def _matches_condition(result, cond: dict) -> bool:
    """True if `result` satisfies every key in a single accept condition."""
    meta = result.metadata or {}
    text = (result.text or "").lower()
    for key, val in cond.items():
        if key == "text_contains":
            if str(val).lower() not in text:
                return False
        elif key == "text_contains_any":
            if not any(str(v).lower() in text for v in val):
                return False
        else:  # exact metadata field match
            if str(meta.get(key, "")).strip().lower() != str(val).strip().lower():
                return False
    return True


def _is_relevant(result, accept: list) -> bool:
    return any(_matches_condition(result, c) for c in accept)


def first_relevant_rank(results: list, accept: list) -> int:
    """1-based rank of the first relevant chunk, or 0 if none in the list."""
    for i, r in enumerate(results, start=1):
        if _is_relevant(r, accept):
            return i
    return 0


# Retrieval  (mirrors step4_query.ask() up to - but not including - generation)

def retrieve(question: str, pool: int = 10):
    """Return (route, [Result, ...]) - the ranked candidates the system would use."""
    q._load_stores()
    target = q.route(question)

    # Commands collection
    if target == "commands":
        date = q.extract_date(question)
        if date:  # exact date filter -> every matching record
            got = q._col_cmd.get(where={"date": {"$eq": date}},
                                 include=["documents", "metadatas"])
            metas = got.get("metadatas") or []
            ids   = got.get("ids") or []
            docs  = got.get("documents") or []
            res = [q.Result(chunk_id=(ids[i] if i < len(ids) else ""),
                            text=(docs[i] if i < len(docs) else ""),
                            score=1.0, metadata=metas[i]) for i in range(len(metas))]
            return target, res[:pool]
        sem  = q.semantic_search(question, q._col_cmd, k=pool)
        bm25 = q.bm25_search(question, q.VECTORSTORE_DIR / "bm25_commands.pkl", k=pool)
        return target, q.rrf_merge([sem, bm25])[:pool]

    # Knowledge collection (hybrid + image-intent boost)
    sub   = q.extract_subsystem(question)
    where = {"subsystem": {"$eq": sub}} if sub else None
    sem   = q.semantic_search(question, q._col_know, k=pool, where=where)
    if not sem and where:
        sem = q.semantic_search(question, q._col_know, k=pool)
    bm25  = q.bm25_search(question, q.VECTORSTORE_DIR / "bm25_knowledge.pkl",
                          k=pool, subsystem=sub)
    merged = q.rrf_merge([sem, bm25])[:pool]

    if q.wants_image(question):
        imgs = q.semantic_search(question, q._col_know, k=3,
                                 where={"content_type": {"$eq": "image_caption"}})
        if imgs:
            seen, comb = set(), []
            for r in imgs[:2] + merged:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    comb.append(r)
            merged = comb

    if getattr(q, "wants_procedure", None) and q.wants_procedure(question):
        fcp_hits = q.semantic_search(question, q._col_know, k=3,
                                     where={"source_doc": {"$eq": "FCP"}})
        if fcp_hits:
            seen, comb = set(), []
            for r in fcp_hits[:2] + merged:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    comb.append(r)
            merged = comb
    return target, merged[:pool]


# Evaluation

def evaluate(pool: int = 10, with_answers: bool = False) -> list:
    rows = []
    for case in TEST_SET:
        target, results = retrieve(case["q"], pool=pool)
        rank = first_relevant_rank(results, case["accept"])
        top = results[0] if results else None
        top_label = ""
        if top is not None:
            m = top.metadata or {}
            top_label = " · ".join(x for x in [
                m.get("source_doc", ""),
                (f"FCP-{m.get('fcp_number')}" if m.get("fcp_number") else ""),
                (f"p.{m.get('page')}" if m.get("page") else ""),
                m.get("content_type", ""),
            ] if x)

        row = {
            "id": case["id"],
            "question": case["q"],
            "route": target,
            "rank": rank,                       # 0 = miss
            "hit@1": int(rank == 1),
            "hit@3": int(1 <= rank <= 3),
            "hit@5": int(1 <= rank <= 5),
            "hit@10": int(rank >= 1),
            "rr": (1.0 / rank) if rank else 0.0,
            "top_source": top_label,
            "note": case.get("note", ""),
        }
        if with_answers:
            try:
                ans, _ = q.ask(case["q"])
                row["answer"] = ans
            except Exception as exc:
                row["answer"] = f"[generation failed: {exc}]"
        rows.append(row)
    return rows


def aggregate(rows: list) -> dict:
    n = len(rows) or 1
    agg = {
        "n": len(rows),
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
        "mrr":  sum(r["rr"]   for r in rows) / n,
        "misses": [r["id"] for r in rows if r["rank"] == 0],
    }
    return agg


# Reporting

def print_report(rows: list, agg: dict) -> None:
    print("=" * 92)
    print("Aditya-L1 RAG - Retrieval Evaluation")
    print("=" * 92)
    print(f"{'id':<13} {'route':<10} {'rank':<5} {'H@1':<4} {'H@3':<4} {'H@5':<4} top source")
    print("-" * 92)
    for r in rows:
        rank = r["rank"] if r["rank"] else "-"
        flag = "" if r["rank"] and r["rank"] <= 5 else "  ⚠ miss@5"
        print(f"{r['id']:<13} {r['route']:<10} {str(rank):<5} "
              f"{r['hit@1']:<4} {r['hit@3']:<4} {r['hit@5']:<4} {r['top_source'][:40]}{flag}")
    print("-" * 92)
    print(f"  Questions      : {agg['n']}")
    print(f"  Hit@1          : {agg['hit@1']*100:5.1f}%")
    print(f"  Hit@3          : {agg['hit@3']*100:5.1f}%")
    print(f"  Hit@5          : {agg['hit@5']*100:5.1f}%")
    print(f"  Hit@10         : {agg['hit@10']*100:5.1f}%")
    print(f"  MRR            : {agg['mrr']:.3f}")
    if agg["misses"]:
        print(f"  Missed (not in top-10): {', '.join(agg['misses'])}")
    print("=" * 92)


def write_json(rows: list, agg: dict, path: Path) -> None:
    path.write_text(json.dumps({"aggregate": agg, "per_question": rows},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {path}")


def write_html(rows: list, agg: dict, path: Path) -> None:
    def esc(s): return html.escape(str(s))
    trs = []
    for r in rows:
        ok = r["rank"] and r["rank"] <= 5
        color = "#e9f7ef" if ok else "#fdecea"
        rank = r["rank"] if r["rank"] else "miss"
        trs.append(
            f"<tr style='background:{color}'>"
            f"<td>{esc(r['id'])}</td><td>{esc(r['question'])}</td>"
            f"<td style='text-align:center'>{esc(r['route'])}</td>"
            f"<td style='text-align:center'>{esc(rank)}</td>"
            f"<td style='text-align:center'>{'✓' if r['hit@1'] else ''}</td>"
            f"<td style='text-align:center'>{'✓' if r['hit@3'] else ''}</td>"
            f"<td style='text-align:center'>{'✓' if r['hit@5'] else ''}</td>"
            f"<td>{esc(r['top_source'])}</td><td>{esc(r['note'])}</td></tr>")
    cards = "".join(
        f"<div class='c'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
        for k, v in [("Hit@1", f"{agg['hit@1']*100:.0f}%"),
                     ("Hit@3", f"{agg['hit@3']*100:.0f}%"),
                     ("Hit@5", f"{agg['hit@5']*100:.0f}%"),
                     ("Hit@10", f"{agg['hit@10']*100:.0f}%"),
                     ("MRR", f"{agg['mrr']:.3f}"),
                     ("Questions", agg["n"])])
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Aditya-L1 RAG - Retrieval Evaluation</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1b2231;background:#eef1f6}}
.wrap{{max-width:1000px;margin:0 auto;background:#fff;border:1px solid #dde3ee;border-radius:12px;padding:24px 28px}}
h1{{font-size:20px;margin:0 0 2px}}.sub{{color:#5a6b82;font-size:12.5px;margin:0 0 16px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
.c{{background:#f5f8fc;border:1px solid #dde3ee;border-radius:9px;padding:12px 18px;text-align:center;min-width:96px}}
.c .v{{font-size:24px;font-weight:700;color:#2f6fd6}}.c .k{{font-size:11px;color:#5a6b82;text-transform:uppercase;letter-spacing:.6px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{border:1px solid #e6ebf3;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#12365f;color:#fff;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
.foot{{color:#5a6b82;font-size:11px;margin-top:14px;text-align:center}}</style></head><body>
<div class="wrap"><h1>Aditya-L1 RAG - Retrieval Evaluation</h1>
<p class="sub">Hit@k = a relevant chunk was in the top-k retrieved candidates · MRR = mean reciprocal rank of the first relevant chunk · retrieval-only (no generation).</p>
<div class="cards">{cards}</div>
<table><tr><th>ID</th><th>Question</th><th>Route</th><th>Rank</th><th>H@1</th><th>H@3</th><th>H@5</th><th>Top source</th><th>Expected</th></tr>
{''.join(trs)}</table>
<div class="foot">Green = relevant chunk in top-5 · Red = miss. Generated by evaluate.py over the labelled question set.</div>
</div></body></html>"""
    path.write_text(doc, encoding="utf-8")
    print(f"  -> wrote {path}")


# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 5: Evaluate Aditya-L1 RAG retrieval")
    ap.add_argument("--pool", type=int, default=10,
                    help="Candidates retrieved per question (default 10)")
    ap.add_argument("--html", action="store_true", help="Write eval_report.html")
    ap.add_argument("--json", action="store_true", help="Write eval_results.json")
    ap.add_argument("--answers", action="store_true",
                    help="Also generate answers per question (needs Ollama; for manual review)")
    args = ap.parse_args()

    rows = evaluate(pool=args.pool, with_answers=args.answers)
    agg  = aggregate(rows)
    print_report(rows, agg)

    if args.json or args.answers:
        write_json(rows, agg, Path("eval_results.json"))
    if args.html:
        write_html(rows, agg, Path("eval_report.html"))

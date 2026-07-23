#!/usr/bin/env python3
"""
Run:
    pip install streamlit pandas
    streamlit run app.py

Notes for a real deployment: there's no authentication here, so put it behind a
login before exposing it; queries are written to assistant_audit.log; and for
many users the retrieval could be split into a separate service. It is advisory
only and never commands the spacecraft.
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import step4_query as q

# Config
APP_TITLE   = "ISRO Assistant"
SUBSYSTEMS  = ["Auto", "OBC", "AOCS", "PAYLOAD", "POWER", "DTG", "MECHANISM",
               "PROPULSION", "TTC_XBAND", "WHEEL", "SENSOR", "ODHS"]
EXAMPLES    = [
    "What is the procedure for switching on OBC?",
    "Which FCP is used to switch on HGA motors?",
    "What commands were uplinked on 2026-06-09?",
    "Provide me the MRS recovery flow diagram",
    "What are the safety logics?",
    "What are the Aditya-L1 orbit constraints?",
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("assistant_audit.log"), logging.StreamHandler()],
)
log = logging.getLogger("al1-assistant")


# One-time model / store initialisation (cached across reruns)
@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def init_backend():
    """Connect ChromaDB and warm the embedding model exactly once per server."""
    q._load_stores()
    q.embed("warm up the embedding model")   # forces BGE-M3 to load now, not per query
    know = q._col_know.count()
    cmds = q._col_cmd.count()
    log.info("Backend ready - knowledge=%d, commands=%d", know, cmds)
    return {"knowledge": know, "commands": cmds}


# Query handling (routes exactly like the CLI, but returns UI-friendly data)
def answer_query(query: str, top_k: int, subsystem: str | None) -> dict:
    """Return a dict the UI knows how to render."""
    log.info("QUERY | k=%d sub=%s | %s", top_k, subsystem, query)
    target = q.route(query)

    # Date/command questions -> structured, interactive table
    if target == "commands" and q.extract_date(query):
        date = q.extract_date(query)
        got  = q._col_cmd.get(where={"date": {"$eq": date}}, include=["metadatas"])
        metas = got.get("metadatas") or []
        rows = sorted(
            [{
                "Time":     m.get("time", ""),
                "CID":      m.get("cid", ""),
                "Mnemonic": m.get("mnemonic", ""),
                "Orbit":    m.get("orbit_no", ""),
                "Status":   m.get("status", ""),
                "From":     m.get("source_station", ""),
                "To":       m.get("dest_station", ""),
                "Subsystem": m.get("subsystem", ""),
            } for m in metas],
            key=lambda r: r["Time"],
        )
        return {"type": "commands", "date": date, "rows": rows}

    # Everything else -> generated answer + sources + any diagrams
    try:
        answer, sources = q.ask(query, top_k=top_k, subsystem_hint=subsystem)
    except Exception as exc:                              # never crash the UI
        log.exception("ask() failed")
        return {"type": "error", "message": str(exc)}

    images = q.collect_images(sources)
    return {"type": "knowledge", "answer": answer, "sources": sources, "images": images}


# Renderers
def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        seen = set()
        for r in sources:
            m = r.metadata or {}
            key = (m.get("source_doc"), m.get("fcp_number"),
                   (m.get("section_heading") or "")[:40], m.get("image_file"))
            if key in seen:
                continue
            seen.add(key)
            bits = [f"**{m.get('source_doc','?')}**"]
            if m.get("fcp_number"):
                bits.append(f"FCP-{m['fcp_number']}")
            if m.get("subsystem") and m["subsystem"] not in ("", "UNKNOWN"):
                bits.append(m["subsystem"])
            if m.get("page"):
                bits.append(f"p.{m['page']}")
            if m.get("content_type") == "image_caption":
                bits.append("🖼 diagram")
            heading = (m.get("section_heading") or "").strip().replace("\n", " ")
            line = " · ".join(bits)
            if heading:
                line += f" - *{heading[:70]}*"
            st.markdown(line)


def render_result(res: dict):
    if res["type"] == "error":
        st.error(f"Something went wrong: {res['message']}")
        return

    if res["type"] == "commands":
        rows = res["rows"]
        if not rows:
            st.info(f"No commands found for {res['date']}.")
            return
        df = pd.DataFrame(rows)
        by_sub = df["Subsystem"].value_counts().to_dict()
        summary = ", ".join(f"{v} {k}" for k, v in by_sub.items())
        st.markdown(f"**{len(df)} commands uplinked on {res['date']}**  ({summary})")
        top_mn = df["Mnemonic"].value_counts().head(5)
        st.caption("Most frequent: " + ", ".join(f"{c}× {mn}" for mn, c in top_mn.items()))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                           file_name=f"commands_{res['date']}.csv", mime="text/csv")
        return

    # knowledge
    st.markdown(res["answer"])
    imgs = [i for i in res.get("images", []) if Path(i).exists()]
    for img in imgs[:3]:                       # show the top few most-relevant figures
        st.image(img, caption=Path(img).name, use_container_width=True)
    if len(imgs) > 3:
        st.caption(f"+ {len(imgs) - 3} more related figure(s) retrieved.")
    render_sources(res.get("sources", []))


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(msg["content"])
    else:
        render_result(msg["result"])


# UI
st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🛰️ Controls")

    try:
        stats = init_backend()
        st.success(f"Store online · {stats['knowledge']} knowledge · "
                   f"{stats['commands']} commands")
    except Exception as exc:
        st.error(f"Backend not available: {exc}")
        st.stop()

    st.subheader("Settings")
    st.caption("Generation: local Ollama (Qwen2.5-7B) - fully offline.")
    top_k   = st.slider("Chunks retrieved (k)", 3, 12, q.TOP_K_FINAL)
    sub_sel = st.selectbox("Subsystem filter", SUBSYSTEMS, index=0,
                           help="'Auto' lets the system infer the subsystem from the question.")
    subsystem = None if sub_sel == "Auto" else sub_sel

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("About"):
        st.markdown(
            "Local Retrieval-Augmented Generation over five Aditya-L1 mission "
            "documents. Hybrid search (semantic + BM25 -> RRF), BGE-M3 embeddings, "
            "ChromaDB, Ollama / Qwen generation. Fully offline.")

# Main pane
st.title(APP_TITLE)


if "messages" not in st.session_state:
    st.session_state.messages = []

# Example prompts (only before the first question)
if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, use_container_width=True):
            st.session_state.pending = ex

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🛰️" if msg["role"] == "assistant" else None):
        render_message(msg)

# Input (typed box or a clicked example)
typed  = st.chat_input("Ask about Aditya-L1 operations...")
prompt = typed or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🛰️"):
        with st.spinner("Retrieving and generating..."):
            result = answer_query(prompt, top_k, subsystem)
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "result": result})

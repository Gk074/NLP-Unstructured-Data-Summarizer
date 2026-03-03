from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from run_mom import build_mom
from src.render import mom_to_markdown
from src.loaders.loader import load_transcript


st.set_page_config(page_title="MoM Summarizer", layout="wide")

st.title("Meeting Minutes (MoM) Summarizer")
st.caption("Upload a transcript → generate structured minutes + download outputs.")

meeting_title = st.text_input("Meeting title", value="MoM")
uploaded = st.file_uploader(
    "Upload transcript",
    type=["txt", "md", "docx", "vtt", "srt", "json", "csv"],
)

# Keep outputs across reruns
if "md" not in st.session_state:
    st.session_state["md"] = None
if "js" not in st.session_state:
    st.session_state["js"] = None
if "err" not in st.session_state:
    st.session_state["err"] = None

if uploaded is None:
    st.info("Upload a file to generate minutes.")
    st.stop()

# Save the upload to a persistent temp file that survives Streamlit reruns
tmp_dir = Path(tempfile.gettempdir()) / "mom_streamlit"
tmp_dir.mkdir(parents=True, exist_ok=True)

uploaded_path = tmp_dir / uploaded.name
uploaded_path.write_bytes(uploaded.getvalue())

text, is_valid, err = load_transcript(uploaded_path)

if err:
    st.error(err)
    st.stop()

# Always show a preview so you know it loaded
with st.expander("Preview normalized transcript (first 40 lines)", expanded=False):
    st.text("\n".join(text.splitlines()[:40]) if text else "(empty)")

if not is_valid:
    st.warning(
        "This file does not appear to be a transcript.\n\n"
        "Expected speaker blocks or timestamps.\n"
        "Try exporting transcript again (Teams/Zoom) or ensure speakers are present."
    )
    st.stop()

# Write normalized version (this is what build_mom reads)
normalized_path = tmp_dir / "normalized_transcript.txt"
normalized_path.write_text(text, encoding="utf-8")

run_btn = st.button("Generate MoM", type="primary")

if run_btn:
    st.session_state["err"] = None
    st.session_state["md"] = None
    st.session_state["js"] = None

    try:
        with st.spinner("Generating MoM (may take ~10-30s on first run)..."):
            mom = build_mom(str(normalized_path), meeting_title=meeting_title)
            md = mom_to_markdown(mom)
            js = mom.model_dump()

        st.session_state["md"] = md
        st.session_state["js"] = js

    except Exception as e:
        st.session_state["err"] = repr(e)

# Show error/output AFTER the button click (persists across reruns)
if st.session_state["err"]:
    st.error("Generation failed:")
    st.code(st.session_state["err"])
    st.stop()

if st.session_state["md"] and st.session_state["js"]:
    col1, col2 = st.columns([1.3, 0.7], gap="large")

    with col1:
        st.subheader("MoM (Markdown)")
        st.markdown(st.session_state["md"])

    with col2:
        st.subheader("Downloads")
        st.download_button(
            "Download mom.md",
            data=st.session_state["md"].encode("utf-8"),
            file_name="mom.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download mom.json",
            data=json.dumps(st.session_state["js"], indent=2).encode("utf-8"),
            file_name="mom.json",
            mime="application/json",
        )
        st.success("Done.")
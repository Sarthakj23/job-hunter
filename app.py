"""
app.py — Streamlit front end for the Gemini-powered LangGraph job hunter.

The one thing to understand about Streamlit + LangGraph: Streamlit re-runs this
entire script top to bottom on every interaction. If the graph and its
checkpointer were built at module level they would be rebuilt on every click and
the paused thread would vanish. `@st.cache_resource` pins one graph instance for
the whole server process, so MemorySaver keeps the checkpoint alive between the
"Find jobs" click and the "Apply" click.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent import (
    GEMINI_FAST_MODEL,
    GEMINI_MODEL,
    build_graph,
    get_pending_interrupt,
    resume_command,
)

st.set_page_config(page_title="Job Hunter", page_icon="•", layout="wide")

NODE_LABELS = {
    "parse_resume_node": "Reading the resume",
    "search_jobs_node": "Searching job boards",
    "evaluate_jobs_node": "Scoring each posting against the profile",
    "human_approval_node": "Recording your selection",
    "auto_apply_node": "Filling application forms",
}

STATUS_STYLE = {
    "submitted": ("Submitted", "✅"),
    "submitted_unconfirmed": ("Sent, no confirmation on the page", "🟡"),
    "filled_not_submitted": ("Filled — review and send it yourself", "📝"),
    "blocked": ("Stopped before submitting", "⛔"),
    "manual_required": ("Needs a human", "🙋"),
    "unsupported": ("Board not supported", "🚫"),
    "failed": ("Failed", "❌"),
    "error": ("Error", "❌"),
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_graph():
    """One compiled graph + MemorySaver for the life of the server process."""
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_upload_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="job_hunter_"))


graph = get_graph()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "thread_id": str(uuid.uuid4()),
    "phase": "setup",          # setup -> awaiting_approval -> done
    "resume_path": None,
    "log": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

config = {"configurable": {"thread_id": st.session_state.thread_id}}


def run_graph(payload, status_box):
    """Stream the graph and push node names into the status box as they finish."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue
            status_box.write(f"{NODE_LABELS.get(node, node)} — done")
            if isinstance(update, dict) and update.get("errors"):
                for err in update["errors"]:
                    st.session_state.log.append(err)


# ---------------------------------------------------------------------------
# Sidebar — search settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Search settings")

    keywords = st.text_input("Role keywords", placeholder="backend engineer, python")
    location = st.text_input("Location", placeholder="Mumbai / Remote")
    remote_only = st.checkbox("Remote roles only", value=False)

    st.divider()
    threshold = st.slider("Minimum fit score", 0, 100, 65, step=5)
    max_shortlist = st.slider("Maximum jobs to shortlist", 1, 20, 8)

    st.divider()
    st.subheader("Applying")
    submit_mode = st.radio(
        "What should the browser do?",
        ["Fill the form and stop", "Fill the form and submit"],
        index=0,
        help="Start with 'fill and stop' on any new job board. Check the screenshot, then switch.",
    )
    show_browser = st.checkbox("Show the browser window", value=False)

    st.divider()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.error(
            "Set GEMINI_API_KEY in your environment or .env file. "
            "Get one from aistudio.google.com/apikey"
        )
    if not os.getenv("TAVILY_API_KEY"):
        st.info("No TAVILY_API_KEY found. Falling back to DuckDuckGo, which returns fewer usable postings.")

    st.caption(f"Scoring: `{GEMINI_MODEL}`")
    st.caption(f"Extraction: `{GEMINI_FAST_MODEL}`")
    st.caption(f"Thread `{st.session_state.thread_id[:8]}`")

    if st.button("Start over", use_container_width=True):
        for key in defaults:
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Job hunter")
st.caption("Finds openings that match your resume, scores them, and waits for your go-ahead before applying.")

# --- Phase 1: upload and run --------------------------------------------------

if st.session_state.phase == "setup":
    uploaded = st.file_uploader("Your resume (PDF)", type=["pdf"])

    if uploaded is not None:
        path = get_upload_dir() / uploaded.name
        path.write_bytes(uploaded.getvalue())
        st.session_state.resume_path = str(path)
        st.success(f"Loaded {uploaded.name}")

    if st.button("Find matching jobs", type="primary", disabled=st.session_state.resume_path is None):
        initial = {
            "resume_path": st.session_state.resume_path,
            "search_preferences": {
                "keywords": keywords,
                "location": location,
                "remote_only": remote_only,
            },
            "score_threshold": threshold,
            "max_shortlist": max_shortlist,
            "submit_applications": submit_mode.endswith("submit"),
            "headless": not show_browser,
        }

        with st.status("Working…", expanded=True) as status_box:
            try:
                run_graph(initial, status_box)
                status_box.update(label="Search complete", state="complete")
            except Exception as exc:  # noqa: BLE001
                status_box.update(label="The run stopped", state="error")
                st.error(str(exc))
                st.stop()

        snapshot = graph.get_state(config)
        st.session_state.phase = "awaiting_approval" if get_pending_interrupt(snapshot) else "done"
        st.rerun()

# --- Phase 2: the breakpoint --------------------------------------------------

elif st.session_state.phase == "awaiting_approval":
    snapshot = graph.get_state(config)
    payload = get_pending_interrupt(snapshot)

    if payload is None:
        st.session_state.phase = "done"
        st.rerun()

    st.subheader("Review before anything is sent")
    st.write(payload["message"])
    st.caption(f"Graph paused at: {', '.join(snapshot.next)}")

    selected: list[str] = []
    for job in payload["jobs"]:
        with st.container(border=True):
            head, meta = st.columns([5, 1])
            with head:
                checked = st.checkbox(
                    f"**{job['title']}** — {job['company']}",
                    key=f"job_{job['id']}",
                )
                st.caption(f"{job.get('location') or 'Location not listed'} · [Open posting]({job['url']})")
                st.write(job.get("reasoning", ""))
                if job.get("missing_skills"):
                    st.caption("Gaps: " + ", ".join(job["missing_skills"][:5]))
            with meta:
                st.metric("Fit", job["score"])
            if checked:
                selected.append(job["id"])

    st.divider()
    left, right = st.columns([1, 3])
    with left:
        approve = st.button(
            f"Apply to {len(selected)} job{'s' if len(selected) != 1 else ''}",
            type="primary",
            disabled=not selected,
        )
    with right:
        skip = st.button("Skip all and finish")

    if approve or skip:
        ids = selected if approve else []
        with st.status("Applying…", expanded=True) as status_box:
            run_graph(resume_command(ids), status_box)
            status_box.update(label="Done", state="complete")
        st.session_state.phase = "done"
        st.rerun()

# --- Phase 3: results ---------------------------------------------------------

elif st.session_state.phase == "done":
    final = graph.get_state(config).values
    results = final.get("application_results") or []
    errors = final.get("errors") or []

    for err in errors:
        st.error(err)

    if not results:
        scored = final.get("scored_jobs") or []
        if scored:
            st.warning(
                f"Scored {len(scored)} postings but none reached your threshold of {threshold}. "
                "Lower the minimum fit score or broaden the keywords in the sidebar."
            )
        elif not errors:
            st.warning("No applications were sent.")
    else:
        st.subheader(f"{len(results)} application{'s' if len(results) != 1 else ''}")

        for r in results:
            label, icon = STATUS_STYLE.get(r.get("status", "error"), (r.get("status", ""), "•"))
            with st.expander(f"{icon} {r.get('title')} — {r.get('company')} · {label}"):
                st.write(f"[Open the posting]({r.get('url')})")
                if r.get("filled_fields"):
                    st.write("Filled: " + ", ".join(r["filled_fields"]))
                if r.get("resume_uploaded"):
                    st.write("Resume uploaded.")
                if r.get("unfilled_required"):
                    st.warning("Still empty and required: " + ", ".join(r["unfilled_required"]))
                if r.get("error"):
                    st.error(r["error"])
                shot = r.get("screenshot")
                if shot and os.path.exists(shot):
                    st.image(shot, caption="Page at the end of the run")

    if st.button("Run another search"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.phase = "setup"
        st.rerun()

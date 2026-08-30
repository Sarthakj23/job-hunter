"""
agent.py — LangGraph StateGraph for the AI Job Hunter & Auto-Applier.
Powered by Google Gemini via langchain-google-genai.

Graph shape:

    START
      |
      v
    parse_resume_node            (Gemini structured-output chain over PDF text)
      |
      v
    search_jobs_node             (Tavily / DuckDuckGo -> Gemini normalisation)
      |
      v
    evaluate_jobs_node           (Gemini scores 0-100, filters below threshold)
      |
      v
    human_approval_node   <----  *** BREAKPOINT: interrupt() pauses the graph here ***
      |
      v
    auto_apply_node              (Playwright tool, one approved job at a time)
      |
      v
     END

The pause uses LangGraph's dynamic `interrupt()`. When the graph hits it,
`.invoke()`/`.stream()` returns with the checkpoint persisted. The caller resumes
with `Command(resume={"approved_job_ids": [...]})`, which makes `interrupt()`
return that payload *inside the same node*, and execution carries on into
`auto_apply_node`.

IMPORTANT: a node containing `interrupt()` re-executes from its first line on
resume. Keep everything above the `interrupt()` call side-effect free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from apply_tool import apply_to_job

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0. Gemini configuration
# ---------------------------------------------------------------------------
#
# Two tiers on purpose. Resume parsing and search-result cleanup are extraction
# work — a Flash-Lite model does them at a fraction of the cost. Job scoring is
# the judgement call the whole system rests on, so it gets the good model.

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")

# Gemini's free tier has a low requests-per-minute ceiling. Raise this once you
# are on a paid key.
SCORING_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "3"))


def _resolve_api_key() -> str:
    """
    langchain-google-genai reads GOOGLE_API_KEY. Google's own docs and AI Studio
    hand you a GEMINI_API_KEY. Accept either and normalise.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key
    return key


def _safety_settings():
    """
    Off by default. Turn on with GEMINI_RELAX_SAFETY=1 if resume or job text is
    tripping Gemini's content filters — it happens with security, defence and
    medical roles, where legitimate job descriptions read as dangerous content.
    BLOCK_ONLY_HIGH, not BLOCK_NONE: enough to stop false positives without
    switching the filters off entirely.
    """
    if os.getenv("GEMINI_RELAX_SAFETY", "0") != "1":
        return None
    try:
        from langchain_google_genai import HarmBlockThreshold, HarmCategory

        return {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    except ImportError:
        logger.warning("Could not import HarmCategory — running with default safety settings.")
        return None


def get_llm(model: str | None = None, reasoning_effort: str | None = None):
    """
    Build a Gemini chat model.

    Note what is *not* here: `temperature`. Google deprecated temperature, top_p
    and top_k on the Gemini 3.x line. Output consistency is controlled through
    reasoning_effort ("low" | "medium" | "high") and through prompts that leave
    less room to improvise.

    Unknown kwargs are retried away rather than crashing, because
    langchain-google-genai renamed several of them in the 4.0 rewrite onto the
    consolidated google-genai SDK.
    """
    if not _resolve_api_key():
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
            "environment or .env file. Get one from https://aistudio.google.com/apikey"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {"model": model or GEMINI_MODEL, "max_retries": 3}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    safety = _safety_settings()
    if safety:
        kwargs["safety_settings"] = safety

    try:
        return ChatGoogleGenerativeAI(**kwargs)
    except (TypeError, ValueError) as exc:
        logger.warning("Gemini rejected an option (%s) — retrying with the model name only.", exc)
        return ChatGoogleGenerativeAI(model=kwargs["model"])


# ---------------------------------------------------------------------------
# 1. Structured output schemas
# ---------------------------------------------------------------------------
#
# Gemini's structured output accepts a subset of OpenAPI schema. Two rules shape
# everything below:
#
#   - No Optional[X] on schema fields. It compiles to `anyOf`, which Gemini
#     rejects. Use a concrete type with a default instead.
#   - No ge/le/min_length constraints. They are dropped, so validate in Python
#     rather than trusting the schema to enforce them.
#
# Literal[...] compiles to a string enum and works fine. Nested models work.


class WorkExperience(BaseModel):
    title: str = Field(default="", description="Job title held")
    company: str = Field(default="", description="Employer name")
    duration: str = Field(default="", description="e.g. 'Jan 2023 - Present'")
    highlights: list[str] = Field(default_factory=list, description="Up to 3 achievements")


class ResumeData(BaseModel):
    """Everything the downstream nodes need from the candidate's PDF."""

    first_name: str = Field(default="", description="Given name only")
    last_name: str = Field(default="", description="Family name only")
    email: str = Field(default="", description="Primary email address")
    phone: str = Field(default="", description="Phone number including country code if present")
    location: str = Field(default="", description="City, State/Country")
    linkedin: str = Field(default="", description="Full LinkedIn URL, empty string if absent")
    github: str = Field(default="", description="Full GitHub URL, empty string if absent")
    portfolio: str = Field(default="", description="Personal site URL, empty string if absent")

    current_title: str = Field(default="", description="Most recent job title")
    years_experience: float = Field(default=0.0, description="Total professional years, 0 if fresher")
    skills: list[str] = Field(default_factory=list, description="Technical and tool skills")
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list, description="Degree, institution, year")
    summary: str = Field(default="", description="Two-sentence profile summary")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class JobPosting(BaseModel):
    title: str = Field(default="", description="Role title")
    company: str = Field(default="Unknown", description="Hiring company")
    location: str = Field(default="", description="City or 'Remote'")
    url: str = Field(default="", description="Exact URL from the search result")
    description: str = Field(default="", description="Snippet or summary of the listing")
    # bool with a default, not Optional[bool] — see the note above.
    is_remote: bool = Field(default=False, description="True if the listing says remote")


class ExtractedJobs(BaseModel):
    """Wrapper so Gemini can return a list under structured output."""

    jobs: list[JobPosting] = Field(default_factory=list)


class JobEvaluation(BaseModel):
    # No ge/le here — Gemini drops numeric constraints. Clamped after the call.
    score: int = Field(default=0, description="Fit score from 0 to 100")
    verdict: Literal["strong", "possible", "weak"] = Field(
        default="weak", description="Bucketed recommendation"
    )
    reasoning: str = Field(default="", description="Two sentences on why this score")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Agent state
# ---------------------------------------------------------------------------
#
# Everything in state is a plain JSON-serialisable dict or primitive. Pydantic
# models are converted with .model_dump() before they go in, so the state stays
# portable across MemorySaver, SqliteSaver and PostgresSaver.


class SearchPreferences(TypedDict, total=False):
    keywords: str            # free-text override, e.g. "backend python fintech"
    location: str            # "Mumbai", "Remote", "India"
    remote_only: bool
    boards: list[str]        # domains to restrict search to


class AgentState(TypedDict, total=False):
    # --- inputs, set once by the caller -------------------------------------
    resume_path: str
    search_preferences: SearchPreferences
    score_threshold: int          # jobs below this are dropped
    max_shortlist: int            # cap on how many jobs reach the human
    submit_applications: bool     # False = fill the form and stop (dry run)
    headless: bool

    # --- produced by parse_resume_node --------------------------------------
    resume_data: dict

    # --- produced by search_jobs_node ---------------------------------------
    search_queries: list[str]
    found_jobs: list[dict]

    # --- produced by evaluate_jobs_node -------------------------------------
    scored_jobs: list[dict]       # every job, with score attached
    shortlisted_jobs: list[dict]  # >= threshold, sorted, capped

    # --- produced by human_approval_node ------------------------------------
    approved_jobs: list[dict]
    rejected_job_ids: list[str]

    # --- produced by auto_apply_node ----------------------------------------
    # Reducer: each write appends instead of replacing, so a partially completed
    # apply run survives in the checkpoint if something dies halfway.
    application_results: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]

    # --- cross-cutting -------------------------------------------------------
    errors: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    status: str


# ---------------------------------------------------------------------------
# 3. Search tooling
# ---------------------------------------------------------------------------


def get_search_tool():
    """
    Tavily where a key exists (it supports domain filtering, which matters here),
    DuckDuckGo otherwise so the graph still runs on a Gemini key alone.

    Gemini's own Google Search grounding is a third option — see ARCHITECTURE.md
    for why it is not the default.
    """
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            return "tavily", TavilySearch(max_results=10, search_depth="advanced")
        except ImportError:
            from langchain_community.tools.tavily_search import TavilySearchResults

            return "tavily", TavilySearchResults(max_results=10)

    from langchain_community.tools import DuckDuckGoSearchResults

    return "duckduckgo", DuckDuckGoSearchResults(output_format="list", num_results=10)


def _normalise_search_output(provider: str, raw: Any) -> list[dict]:
    """Flatten the possible shapes into [{title, url, content}]."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if isinstance(raw, dict):
        raw = raw.get("results", [])

    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        if not url:
            continue
        out.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "content": (item.get("content") or item.get("snippet") or "")[:1200],
            }
        )
    return out


def job_id(url: str) -> str:
    """Stable, short identifier so the UI can round-trip selections through state."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------

PARSE_SYSTEM = """You extract structured data from resumes.

Return only what is present in the text. Use an empty string for anything absent —
never invent an email, phone number or URL. Normalise skills to their common names
(e.g. "ReactJS" -> "React"). Estimate years_experience from the dated roles; use 0
for a candidate with only internships or education."""


def parse_resume_node(state: AgentState) -> dict:
    """Read the uploaded PDF and run a Gemini structured-output chain over its text."""
    path = state.get("resume_path", "")
    if not path or not os.path.exists(path):
        return {"errors": [f"Resume not found at {path!r}"], "status": "resume_missing"}

    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"Could not read the PDF: {exc}"], "status": "resume_unreadable"}

    if len(text.strip()) < 100:
        return {
            "errors": [
                "The PDF contains almost no selectable text. It is probably a scan — "
                "run it through OCR or export a text-based PDF and try again."
            ],
            "status": "resume_unreadable",
        }

    # Extraction, not judgement — the cheap model with minimal thinking.
    chain = get_llm(model=GEMINI_FAST_MODEL, reasoning_effort="low").with_structured_output(ResumeData)
    try:
        parsed: ResumeData = chain.invoke(
            [
                SystemMessage(content=PARSE_SYSTEM),
                HumanMessage(content=f"Resume text:\n\n{text[:20000]}"),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"Resume parsing failed: {exc}"], "status": "resume_unreadable"}

    data = parsed.model_dump()
    data["full_name"] = parsed.full_name
    logger.info("Parsed resume for %s (%d skills)", data["full_name"] or "unknown", len(data["skills"]))
    return {"resume_data": data, "status": "resume_parsed"}


# Restricting search to these ATS domains is deliberate: they are the same
# platforms the Playwright tool in apply_tool.py knows how to fill. Searching
# LinkedIn or Indeed surfaces jobs the applier cannot actually complete.
DEFAULT_BOARDS = [
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "apply.workable.com",
]

EXTRACT_SYSTEM = """You turn raw web search results into structured job postings.

Rules:
- Keep only results that are a single job posting. Drop board landing pages,
  "all jobs" listings, blog posts and salary guides.
- The company is usually in the URL path (boards.greenhouse.io/<company>/...)
  or the page title.
- Copy the URL exactly as given. Never construct or shorten a URL.
- If you cannot tell what the role is, drop the result rather than guessing."""


def search_jobs_node(state: AgentState) -> dict:
    """Build queries from the parsed profile, search, then normalise with Gemini."""
    resume = state.get("resume_data") or {}
    prefs: SearchPreferences = state.get("search_preferences") or {}

    title = prefs.get("keywords") or resume.get("current_title") or "software engineer"
    location = prefs.get("location", "")
    top_skills = (resume.get("skills") or [])[:6]

    queries = [
        f"{title} jobs {location}".strip(),
        f"{title} {' '.join(top_skills[:3])} hiring".strip(),
    ]
    if prefs.get("remote_only"):
        queries.append(f"remote {title} jobs")
    if top_skills[3:]:
        queries.append(f"{' '.join(top_skills[3:6])} engineer jobs {location}".strip())

    provider, tool = get_search_tool()
    boards = prefs.get("boards") or DEFAULT_BOARDS

    raw_results: list[dict] = []
    for q in queries:
        query = q if provider == "tavily" else f"{q} site:{boards[0]}"
        try:
            if provider == "tavily":
                payload = tool.invoke({"query": query, "include_domains": boards})
            else:
                payload = tool.invoke(query)
            raw_results.extend(_normalise_search_output(provider, payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search failed for %r: %s", query, exc)

    # De-duplicate on URL before spending tokens on extraction.
    seen, deduped = set(), []
    for r in raw_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    if not deduped:
        return {
            "search_queries": queries,
            "found_jobs": [],
            "errors": ["Search returned nothing. Check your search API key or widen the keywords."],
            "status": "no_jobs_found",
        }

    chain = get_llm(model=GEMINI_FAST_MODEL, reasoning_effort="low").with_structured_output(ExtractedJobs)
    try:
        extracted: ExtractedJobs = chain.invoke(
            [
                SystemMessage(content=EXTRACT_SYSTEM),
                HumanMessage(content=f"Search results:\n\n{json.dumps(deduped[:40], indent=2)}"),
            ]
        )
        postings = [p for p in extracted.jobs if p.url]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Extraction failed, falling back to raw results: %s", exc)
        postings = [JobPosting(title=r["title"], url=r["url"], description=r["content"]) for r in deduped]

    jobs = []
    for p in postings:
        d = p.model_dump()
        d["id"] = job_id(p.url)
        d["source"] = provider
        jobs.append(d)

    logger.info("Found %d postings from %d search results", len(jobs), len(deduped))
    return {"search_queries": queries, "found_jobs": jobs, "status": "jobs_found"}


EVAL_SYSTEM = """You are a blunt technical recruiter scoring one job against one candidate.

Scoring guide:
  85-100  Candidate meets or exceeds the core requirements. Worth applying today.
  65-84   Real overlap, one or two gaps that a cover letter can address.
  40-64   Adjacent role. Applying is a long shot.
  0-39    Wrong field, wrong seniority, or a hard requirement the candidate lacks
          (visa, clearance, licence, 10+ years when they have 2).

Penalise seniority mismatch heavily in both directions. Do not inflate scores to
be encouraging — a shortlist full of 90s is useless to the candidate."""


def evaluate_jobs_node(state: AgentState) -> dict:
    """Score every found job against the resume, then filter and rank."""
    jobs = state.get("found_jobs") or []
    resume = state.get("resume_data") or {}
    threshold = state.get("score_threshold", 65)
    max_shortlist = state.get("max_shortlist", 10)

    if not jobs:
        return {"scored_jobs": [], "shortlisted_jobs": [], "status": "no_jobs_found"}

    profile = json.dumps(
        {
            "current_title": resume.get("current_title"),
            "years_experience": resume.get("years_experience"),
            "skills": resume.get("skills"),
            "experience": resume.get("experience"),
            "education": resume.get("education"),
            "location": resume.get("location"),
        },
        indent=2,
    )

    # This is the judgement call the whole system rests on — good model, real
    # thinking budget.
    chain = get_llm(model=GEMINI_MODEL, reasoning_effort="medium").with_structured_output(JobEvaluation)
    batch_inputs = [
        [
            SystemMessage(content=EVAL_SYSTEM),
            HumanMessage(
                content=(
                    f"CANDIDATE:\n{profile}\n\n"
                    f"JOB:\ntitle: {j.get('title')}\ncompany: {j.get('company')}\n"
                    f"location: {j.get('location')}\ndescription: {j.get('description', '')[:2000]}"
                )
            ),
        ]
        for j in jobs
    ]

    # Concurrency is capped low because Gemini's free tier limits requests per
    # minute. return_exceptions keeps one bad posting from killing the node.
    try:
        evaluations = chain.batch(
            batch_inputs,
            config={"max_concurrency": SCORING_CONCURRENCY},
            return_exceptions=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"Scoring failed: {exc}"], "status": "evaluation_failed"}

    scored = []
    for job, ev in zip(jobs, evaluations):
        enriched = dict(job)
        if isinstance(ev, JobEvaluation):
            enriched.update(
                # Clamp in Python — Gemini drops ge/le from the schema.
                score=max(0, min(100, int(ev.score))),
                verdict=ev.verdict,
                reasoning=ev.reasoning,
                matched_skills=ev.matched_skills,
                missing_skills=ev.missing_skills,
            )
        else:
            logger.warning("Scoring failed for %s: %s", job.get("url"), ev)
            enriched.update(score=0, verdict="weak", reasoning="Scoring failed for this posting.")
        scored.append(enriched)

    scored.sort(key=lambda j: j["score"], reverse=True)
    shortlist = [j for j in scored if j["score"] >= threshold][:max_shortlist]

    logger.info("Scored %d jobs, %d cleared the threshold of %d", len(scored), len(shortlist), threshold)
    return {
        "scored_jobs": scored,
        "shortlisted_jobs": shortlist,
        "status": "awaiting_approval" if shortlist else "no_matches",
    }


def human_approval_node(state: AgentState) -> dict:
    """
    *** THE BREAKPOINT ***

    `interrupt()` throws a GraphInterrupt. LangGraph persists the checkpoint and
    returns control to the caller. Nothing below this line runs until the caller
    resumes with Command(resume=...), at which point the whole node re-executes
    and interrupt() returns the resume payload.

    Expected resume payload:
        {"approved_job_ids": ["a1b2c3d4e5", ...]}
    or  {"approved_job_ids": []}
    """
    shortlist = state.get("shortlisted_jobs") or []

    decision = interrupt(
        {
            "type": "approval_required",
            "message": (
                f"{len(shortlist)} job(s) cleared the fit threshold. "
                "Select the ones to apply to. Nothing is submitted until you approve."
            ),
            "jobs": [
                {
                    "id": j["id"],
                    "title": j.get("title"),
                    "company": j.get("company"),
                    "location": j.get("location"),
                    "url": j.get("url"),
                    "score": j.get("score"),
                    "reasoning": j.get("reasoning"),
                    "missing_skills": j.get("missing_skills", []),
                }
                for j in shortlist
            ],
        }
    )

    # --- everything below runs only after resume ---------------------------
    approved_ids = set((decision or {}).get("approved_job_ids", []))
    approved = [j for j in shortlist if j["id"] in approved_ids]
    rejected = [j["id"] for j in shortlist if j["id"] not in approved_ids]

    logger.info("Human approved %d of %d jobs", len(approved), len(shortlist))
    return {
        "approved_jobs": approved,
        "rejected_job_ids": rejected,
        "status": "approved" if approved else "declined",
    }


def auto_apply_node(state: AgentState) -> dict:
    """Drive the Playwright tool over the approved jobs, one at a time."""
    approved = state.get("approved_jobs") or []
    resume = state.get("resume_data") or {}
    resume_path = state.get("resume_path", "")
    submit = state.get("submit_applications", False)
    headless = state.get("headless", True)

    candidate = {
        "first_name": resume.get("first_name", ""),
        "last_name": resume.get("last_name", ""),
        "full_name": resume.get("full_name", ""),
        "email": resume.get("email", ""),
        "phone": resume.get("phone", ""),
        "location": resume.get("location", ""),
        "linkedin": resume.get("linkedin", ""),
        "github": resume.get("github", ""),
        "portfolio": resume.get("portfolio", ""),
        "summary": resume.get("summary", ""),
    }

    results = []
    for i, job in enumerate(approved):
        try:
            result = apply_to_job.invoke(
                {
                    "job_url": job["url"],
                    "candidate": candidate,
                    "resume_path": resume_path,
                    "submit": submit,
                    "headless": headless,
                }
            )
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "error": str(exc)}

        result.update(job_id=job["id"], title=job.get("title"), company=job.get("company"), url=job["url"])
        results.append(result)

        # Serial, with a gap. Parallel browsers on the same ATS is the fastest
        # way to get rate-limited or flagged.
        if i < len(approved) - 1:
            time.sleep(4)

    return {"application_results": results, "status": "applications_complete"}


# ---------------------------------------------------------------------------
# 5. Conditional edges
# ---------------------------------------------------------------------------


def route_after_parse(state: AgentState) -> Literal["search_jobs_node", "__end__"]:
    return "search_jobs_node" if state.get("resume_data") else END


def route_after_evaluate(state: AgentState) -> Literal["human_approval_node", "__end__"]:
    return "human_approval_node" if state.get("shortlisted_jobs") else END


def route_after_approval(state: AgentState) -> Literal["auto_apply_node", "__end__"]:
    return "auto_apply_node" if state.get("approved_jobs") else END


# ---------------------------------------------------------------------------
# 6. Build + compile
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """
    Compile the graph. A checkpointer is mandatory for interrupt() to work —
    without persisted state there is nothing to resume into.

    MemorySaver keeps checkpoints in the process only. Swap it for
    SqliteSaver.from_conn_string("checkpoints.sqlite") to survive restarts.
    """
    builder = StateGraph(AgentState)

    builder.add_node("parse_resume_node", parse_resume_node)
    builder.add_node("search_jobs_node", search_jobs_node)
    builder.add_node("evaluate_jobs_node", evaluate_jobs_node)
    builder.add_node("human_approval_node", human_approval_node)
    builder.add_node("auto_apply_node", auto_apply_node)

    builder.add_edge(START, "parse_resume_node")
    builder.add_conditional_edges("parse_resume_node", route_after_parse)
    builder.add_edge("search_jobs_node", "evaluate_jobs_node")
    builder.add_conditional_edges("evaluate_jobs_node", route_after_evaluate)
    builder.add_conditional_edges("human_approval_node", route_after_approval)
    builder.add_edge("auto_apply_node", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())

    # --- Alternative: static breakpoint -------------------------------------
    # If you would rather set the approved jobs by writing directly to state
    # instead of passing a resume payload, drop interrupt() from
    # human_approval_node and compile like this:
    #
    #     graph = builder.compile(checkpointer=cp, interrupt_before=["auto_apply_node"])
    #
    # then resume with:
    #
    #     graph.update_state(config, {"approved_jobs": [...]})
    #     graph.invoke(None, config)     # None = "continue from the checkpoint"
    #
    # interrupt() is the better default: the pause is declared by the node that
    # owns it, and the payload the UI needs travels with the interrupt instead of
    # having to be dug out of the state snapshot.


# ---------------------------------------------------------------------------
# 7. Helpers the UI layer needs
# ---------------------------------------------------------------------------


def get_pending_interrupt(snapshot) -> Optional[dict]:
    """
    Pull the interrupt payload out of a StateSnapshot, across LangGraph versions.
    Returns None if the graph is not currently paused.
    """
    for attr in ("interrupts", "__interrupts__"):
        vals = getattr(snapshot, attr, None)
        if vals:
            return vals[0].value if hasattr(vals[0], "value") else vals[0]

    for task in getattr(snapshot, "tasks", ()) or ():
        if getattr(task, "interrupts", None):
            return task.interrupts[0].value
    return None


def is_paused(snapshot) -> bool:
    return bool(getattr(snapshot, "next", ())) and get_pending_interrupt(snapshot) is not None


def resume_command(approved_job_ids: list[str]) -> Command:
    """Build the payload human_approval_node expects."""
    return Command(resume={"approved_job_ids": approved_job_ids})


# ---------------------------------------------------------------------------
# 8. CLI harness — run the whole flow from a terminal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Run the Gemini job hunter graph end to end.")
    ap.add_argument("resume", help="Path to a text-based resume PDF")
    ap.add_argument("--location", default="")
    ap.add_argument("--keywords", default="")
    ap.add_argument("--threshold", type=int, default=65)
    ap.add_argument("--submit", action="store_true", help="Actually click submit (default: fill only)")
    ap.add_argument("--show-browser", action="store_true")
    args = ap.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": "cli-session"}}

    initial: AgentState = {
        "resume_path": args.resume,
        "search_preferences": {"keywords": args.keywords, "location": args.location},
        "score_threshold": args.threshold,
        "max_shortlist": 10,
        "submit_applications": args.submit,
        "headless": not args.show_browser,
    }

    for chunk in graph.stream(initial, config, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"[{node}] {list(update.keys()) if isinstance(update, dict) else update}")

    snapshot = graph.get_state(config)
    payload = get_pending_interrupt(snapshot)

    if not payload:
        print("\nGraph finished without pausing.")
        for err in snapshot.values.get("errors", []):
            print(f"  - {err}")
        raise SystemExit(0)

    print(f"\n{payload['message']}\n")
    for idx, job in enumerate(payload["jobs"], 1):
        print(f"{idx}. [{job['score']}] {job['title']} — {job['company']}")
        print(f"   {job['url']}")
        print(f"   {job['reasoning']}\n")

    raw = input("Numbers to apply to (comma separated, blank to skip): ").strip()
    chosen = [payload["jobs"][int(n) - 1]["id"] for n in raw.split(",") if n.strip().isdigit()]

    for chunk in graph.stream(resume_command(chosen), config, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"[{node}] {update}")

    final = graph.get_state(config).values
    print("\nResults:")
    print(json.dumps(final.get("application_results", []), indent=2))

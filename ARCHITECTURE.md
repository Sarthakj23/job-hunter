# AI Job Hunter — architecture plan (Gemini edition)

## 1. Why a StateGraph and not an agent loop

A ReAct-style agent would decide for itself when to search, when to score and when
to apply. That is the wrong shape here, because one of the steps is irreversible:
a submitted application cannot be recalled, and a bad one costs the candidate a
real relationship with a real company.

A `StateGraph` makes the order of operations a property of the system rather than
a property of the model's reasoning that turn. `auto_apply_node` is unreachable
except through `human_approval_node`. That is enforced by the edge topology, not
by a prompt asking the model to please check first.

```
START → parse_resume → search_jobs → evaluate_jobs → [PAUSE] human_approval → auto_apply → END
                            │              │                     │
                            └── no results ┴── nothing over ─────┴── nothing approved
                                              threshold                     │
                                                  ↓                         ↓
                                                 END                       END
```

## 2. Gemini setup

### Two model tiers

| Node | Model | Reasoning effort | Why |
|---|---|---|---|
| `parse_resume_node` | `gemini-3.5-flash-lite` | low | Copying fields out of text. No judgement involved. |
| `search_jobs_node` | `gemini-3.5-flash-lite` | low | Same — cleaning search results into a schema. |
| `evaluate_jobs_node` | `gemini-3.7-flash` | medium | The judgement call the whole shortlist rests on. |

Both are overridable through `GEMINI_MODEL` and `GEMINI_FAST_MODEL`. Scoring is
also the only node that runs once per job, so it dominates cost — putting the
cheap model on the two single-call nodes saves less than putting it on scoring
would, which is exactly why scoring keeps the good model.

Gemini 2.5 models still work but shut down on 16 October 2026. Don't start a new
build on them.

### No temperature

Google deprecated `temperature`, `top_p` and `top_k` on the Gemini 3.x line. Any
tutorial you find that passes `temperature=0` to `ChatGoogleGenerativeAI` was
written for 2.x. Output consistency now comes from two places: `reasoning_effort`
(`"low" | "medium" | "high"`) and prompts that leave less room to improvise.

`get_llm()` also catches `TypeError`/`ValueError` on construction and retries with
just the model name. `langchain-google-genai` 4.0 was a rewrite onto the
consolidated `google-genai` SDK and renamed several kwargs; the fallback means a
version skew degrades instead of crashing.

### Key naming

AI Studio gives you a `GEMINI_API_KEY`. `langchain-google-genai` reads
`GOOGLE_API_KEY`. `_resolve_api_key()` accepts either and copies across, because
this mismatch is the single most common reason a first run fails with an
authentication error that names a variable you never set.

### Schema constraints that shaped the Pydantic models

Gemini's structured output accepts a subset of OpenAPI schema, and two of the
gaps change how you write models:

- **No `Optional[X]`.** It compiles to `anyOf`, which Gemini rejects. `JobPosting.is_remote`
  is `bool = False`, not `Optional[bool] = None`. Every string field defaults to
  `""` rather than being nullable.
- **No numeric constraints.** `Field(ge=0, le=100)` is dropped rather than
  enforced, so `JobEvaluation.score` carries no bounds in the schema and is
  clamped in Python after the call: `max(0, min(100, int(ev.score)))`.

`Literal[...]` compiles to a string enum and works. Nested models
(`ResumeData.experience: list[WorkExperience]`) work.

### Rate limits

`evaluate_jobs_node` uses `.batch()` with `max_concurrency` from
`GEMINI_MAX_CONCURRENCY`, defaulting to 3. The free tier's requests-per-minute
ceiling is low enough that scoring ten jobs at concurrency 10 will return 429s for
most of them. `return_exceptions=True` means those come back as exceptions in the
result list rather than killing the node — failed postings get score 0 and a note.

### Safety filters

Off by default, behind `GEMINI_RELAX_SAFETY=1`. Worth knowing about because
legitimate job descriptions do trip Gemini's content filters — security, defence
and clinical roles most often. When it fires you get an empty or blocked response
that looks like a parsing bug. The setting uses `BLOCK_ONLY_HIGH`, not
`BLOCK_NONE`: enough to clear false positives without switching filters off.

## 3. State design

`AgentState` is a `TypedDict`, and everything stored in it is a plain dict or
primitive. Pydantic models live at the boundaries — `with_structured_output`
returns them, and they are converted with `.model_dump()` before entering state.

That is deliberate. The moment you move off `MemorySaver` onto `SqliteSaver` or
`PostgresSaver`, the state has to survive a serialisation round trip. Plain JSON
does; nested custom classes need a custom serialiser.

Two fields carry reducers:

```python
application_results: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]
errors:              Annotated[list[str],  lambda a, b: (a or []) + (b or [])]
```

Appending rather than replacing means a partially completed apply run is still in
the checkpoint if something dies halfway. Everything else uses the default
last-write-wins, because a re-run of `search_jobs_node` should replace the old
results rather than accumulate duplicates.

The state separates three lists that are tempting to collapse into one:

| Field | Written by | Meaning |
|---|---|---|
| `found_jobs` | search | Everything the search turned up |
| `scored_jobs` | evaluate | All of the above, with a score |
| `shortlisted_jobs` | evaluate | Above threshold, ranked, capped |
| `approved_jobs` | human | What the person actually picked |

`auto_apply_node` reads only `approved_jobs`. If a future change accidentally
writes to `shortlisted_jobs` instead, nothing gets applied to — which is the
correct direction to fail in.

## 4. The breakpoint

Two mechanisms exist in LangGraph. Both are in the code; `interrupt()` is wired up.

**Dynamic — `interrupt()` (used here).** The node calls `interrupt(payload)`.
LangGraph raises `GraphInterrupt`, persists the checkpoint, and returns control to
the caller. The caller resumes with `Command(resume={...})`, and `interrupt()`
returns that value from inside the node.

```python
decision = interrupt({"type": "approval_required", "jobs": [...]})
approved_ids = set(decision["approved_job_ids"])   # only runs after resume
```

The payload the UI needs travels *with* the interrupt, so the front end never has
to reach into the state snapshot and guess which field to render.

**Static — `interrupt_before=["auto_apply_node"]`.** Compile-time declaration,
resumed with `graph.update_state(config, {...})` then `graph.invoke(None, config)`.
Useful when the pause is an operator concern rather than a step in the workflow.

**The thing that catches people out:** a node containing `interrupt()` re-executes
from its first line on resume. Anything above the call runs twice. Keep that
region side-effect free — no writes, no API calls, no counters. In
`human_approval_node` there is nothing above the call at all.

## 5. Checkpointer choice

`MemorySaver` keeps checkpoints in the Python process. Fine for a demo, wrong the
moment you want a person to approve jobs from their phone an hour later, or you
restart the server, or you run more than one Streamlit worker.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    graph = build_graph(checkpointer=cp)
```

The `thread_id` in `config` is the resume token. Store it wherever the user's
session lives and the paused graph can be picked up from any process that shares
the checkpoint store.

## 6. Search strategy

`search_jobs_node` restricts search to Greenhouse, Lever, Ashby and Workable.

This looks arbitrary and it is the most important decision in the system. Those
four are the boards the Playwright tool can actually complete: predictable DOM, no
login, no bot wall. Searching LinkedIn or Indeed would produce a longer shortlist
of jobs the applier then fails on, which is worse than a short shortlist that
works.

Web search remains the weakest link. Results are aggregator pages as often as
postings, which is why a Gemini extraction pass sits between the search tool and
`found_jobs`, told to drop anything that is not a single posting.

**Why not Gemini's Google Search grounding?** It is the obvious Google-native
option and it is genuinely good at finding things. Two problems for this
particular node: grounding gives you a synthesised answer plus citation metadata
rather than a clean result list, and the metadata shape has moved between SDK
versions — so URL extraction becomes version-sensitive parsing. More decisively,
grounding has no equivalent of Tavily's `include_domains`, and domain restriction
is the whole reason the search stage works here. If you want to try it anyway,
bind the search tool to a Gemini model and read `response_metadata` for grounding
chunks, but treat the URLs as untrusted until you have checked the host.

For reliability rather than demonstrability, skip web search entirely and use the
boards' own public JSON endpoints:

```
https://boards-api.greenhouse.io/v1/boards/{company}/jobs
https://api.lever.co/v0/postings/{company}?mode=json
```

## 7. Scoring

The prompt tells Gemini to penalise seniority mismatch in both directions and
explicitly not to inflate. A grading model left unprompted hands out 85s to
everything, and a shortlist where every job scores 85 gives the human nothing to
decide with — which quietly defeats the purpose of the breakpoint.

## 8. The apply tool

`apply_tool.py` contains no model calls at all. It is deterministic browser
automation, which is why switching the whole project to Gemini did not touch it.

Three details that matter:

- **`submit=False` by default.** The tool navigates, fills, screenshots and stops.
  You get visual proof of what it would have sent. Flip to `submit=True` per board
  once you have seen the screenshot.
- **Runs in a fresh thread.** Playwright's sync API raises if it finds an asyncio
  loop in the current thread, which Streamlit and some LangGraph runtimes have.
  A `ThreadPoolExecutor(max_workers=1)` gives it a clean thread.
- **Named unsupported boards.** Workday, LinkedIn, Indeed, Taleo and iCIMS return
  `status: "unsupported"` with a reason instead of half-filling a form and
  reporting success. Silent partial failure is the worst outcome for this tool.

Field resolution goes CSS selectors → accessible label regex → give up and report
the field as skipped. Applications run serially with a four-second gap. Parallel
browser sessions against one ATS is the fastest route to a rate limit.

## 9. Streamlit integration

The trap: Streamlit re-runs the whole script on every widget interaction. A graph
built at module scope is rebuilt on every click, taking its `MemorySaver` and the
paused thread with it.

```python
@st.cache_resource
def get_graph():
    return build_graph()
```

`cache_resource` pins one instance per server process. The `thread_id` lives in
`st.session_state`, and `st.session_state.phase` drives which of the three views
renders — setup, approval, results.

The approval view reads the interrupt payload fresh from `graph.get_state(config)`
on every rerun rather than caching it in session state, so the checkpoint stays
the single source of truth about where the graph is.

## 10. What this does not solve

- **Workday and LinkedIn.** Roughly half the market. Workday needs an account per
  employer and a multi-step wizard; LinkedIn Easy Apply is against its terms of
  use and actively defended against. There is no clean automated path.
- **Custom questions.** "Why do you want to work here?", visa status, notice
  period, salary expectation. The tool reports them under `unfilled_required` and
  stops rather than guessing. Answering them with Gemini is the obvious next
  feature and also the one most likely to submit something embarrassing.
- **Job board terms of service.** Most prohibit automated submission. The
  human-in-the-loop breakpoint is what keeps this a tool that helps you apply
  rather than a bot that sprays applications — the value of the design is lost if
  you approve 200 jobs at once.
- **Volume is not the goal.** Recruiters spot bulk applications easily, and a
  scattergun approach damages the candidate more than it helps. The score
  threshold exists to keep the shortlist short.

## 11. Extensions worth building next

| Feature | Where it goes |
|---|---|
| Cover letter generation | New node between approval and apply, writing to `approved_jobs[i]["cover_letter"]` |
| Answering custom questions | New tool called from `auto_apply_node` when `unfilled_required` is non-empty, gated by a second interrupt |
| Multimodal resume parsing | Gemini accepts PDFs natively — send the file instead of `pypdf` text and you keep layout, tables and two-column resumes that text extraction mangles |
| Deduplication across runs | Persist applied URLs; filter in `search_jobs_node` |
| Retry on transient failure | Conditional edge from `auto_apply_node` back to itself with a retry counter in state |
| Application tracking | Postgres checkpointer plus a separate results table keyed on `thread_id` |

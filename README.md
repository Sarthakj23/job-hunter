# AI Job Hunter & Auto-Applier — Gemini edition

LangGraph `StateGraph` that parses a resume, finds matching roles, scores them,
**pauses for your approval**, then fills application forms with Playwright.
All model calls run on Google Gemini.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env      # add your GEMINI_API_KEY
```

Get a key at https://aistudio.google.com/apikey

## Run

```bash
streamlit run app.py                             # UI
python agent.py resume.pdf --location Mumbai     # same graph, terminal
```

## Models

| Job | Default | Why |
|---|---|---|
| Scoring | `gemini-3.7-flash` | The shortlist is only as good as this call |
| Parsing, extraction | `gemini-3.5-flash-lite` | Pure extraction — cheap model is enough |

Override both in `.env`. Stable alternatives: `gemini-3.6-flash`,
`gemini-3.5-flash`, `gemini-3.1-flash-lite`.

Gemini 2.5 models still work but shut down on 16 October 2026 — don't start there.

## Files

| File | What's in it |
|---|---|
| `agent.py` | State schema, the five nodes, Gemini setup, graph compilation, CLI |
| `apply_tool.py` | The Playwright `@tool` and per-ATS field maps |
| `app.py` | Streamlit UI — runs, renders the pause, resumes |
| `ARCHITECTURE.md` | Design decisions and the reasoning behind them |

## First run

Leave the sidebar on **"Fill the form and stop"**. The tool completes each form
and screenshots it without submitting. Check the screenshots. Only switch to
"Fill the form and submit" once the selectors work for that board.

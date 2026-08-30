"""
apply_tool.py — the LangChain tool that actually touches a browser.

Design notes worth knowing before you extend this:

1. Playwright's *sync* API refuses to run inside a thread that already has an
   asyncio event loop. Streamlit and some LangGraph runtimes do have one. Every
   browser session is therefore pushed onto a fresh worker thread, which has no
   loop of its own.

2. `submit=False` is the default. The tool navigates, fills, screenshots and
   stops. You get a browser window (or a screenshot) showing a completed form
   that a human can eyeball and send. Turn submit on only once you trust the
   selectors for that specific board.

3. Selectors are per-ATS, with a generic label-based fallback. Greenhouse, Lever,
   Ashby and Workable are the realistic targets: predictable DOM, no login,
   no anti-bot wall. Workday and LinkedIn are not supported and the tool says so
   rather than half-filling something and claiming success.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("JOB_HUNTER_ARTIFACTS", "./artifacts"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


class ApplyToJobInput(BaseModel):
    job_url: str = Field(description="Direct URL of the job application page")
    candidate: dict = Field(
        description=(
            "Candidate fields: first_name, last_name, full_name, email, phone, "
            "location, linkedin, github, portfolio, summary"
        )
    )
    resume_path: str = Field(description="Absolute path to the resume PDF to upload")
    submit: bool = Field(default=False, description="Click the submit button. Default False = fill only.")
    headless: bool = Field(default=True, description="Run the browser without a visible window")
    timeout_ms: int = Field(default=30000, description="Per-action timeout in milliseconds")


# ---------------------------------------------------------------------------
# ATS detection + field maps
# ---------------------------------------------------------------------------

UNSUPPORTED = {
    "myworkdayjobs.com": "Workday requires an account and a multi-step wizard.",
    "workday.com": "Workday requires an account and a multi-step wizard.",
    "linkedin.com": "LinkedIn blocks automation and Easy Apply is against its terms of use.",
    "indeed.com": "Indeed serves a bot challenge to automated browsers.",
    "taleo.net": "Taleo requires account creation before the form appears.",
    "icims.com": "iCIMS gates the form behind a login on most tenants.",
}


def detect_ats(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    if "workable.com" in host:
        return "workable"
    return "generic"


# Each entry: canonical field -> (candidate key, [css selectors tried in order],
# [label regexes for the get_by_label fallback])
FIELD_MAPS: dict[str, dict[str, tuple[str, list[str], list[str]]]] = {
    "greenhouse": {
        "first_name": ("first_name", ["#first_name", "input[name='first_name']"], [r"first name"]),
        "last_name": ("last_name", ["#last_name", "input[name='last_name']"], [r"last name"]),
        "email": ("email", ["#email", "input[name='email']"], [r"email"]),
        "phone": ("phone", ["#phone", "input[name='phone']"], [r"phone"]),
        "linkedin": ("linkedin", ["input[name*='linkedin' i]"], [r"linkedin"]),
    },
    "lever": {
        "full_name": ("full_name", ["input[name='name']"], [r"full name", r"^name$"]),
        "email": ("email", ["input[name='email']"], [r"email"]),
        "phone": ("phone", ["input[name='phone']"], [r"phone"]),
        "location": ("location", ["input[name='location']"], [r"location"]),
        "linkedin": ("linkedin", ["input[name='urls[LinkedIn]']"], [r"linkedin"]),
        "github": ("github", ["input[name='urls[GitHub]']"], [r"github"]),
        "portfolio": ("portfolio", ["input[name='urls[Portfolio]']"], [r"portfolio"]),
    },
    "ashby": {
        "full_name": ("full_name", ["input[name='_systemfield_name']"], [r"name"]),
        "email": ("email", ["input[name='_systemfield_email']"], [r"email"]),
        "phone": ("phone", ["input[name='_systemfield_phone']"], [r"phone"]),
        "linkedin": ("linkedin", ["input[name*='linkedin' i]"], [r"linkedin"]),
    },
    "workable": {
        "first_name": ("first_name", ["input[name='firstname']", "#firstname"], [r"first name"]),
        "last_name": ("last_name", ["input[name='lastname']", "#lastname"], [r"last name"]),
        "email": ("email", ["input[name='email']", "#email"], [r"email"]),
        "phone": ("phone", ["input[name='phone']", "#phone"], [r"phone"]),
    },
    "generic": {
        "first_name": (
            "first_name",
            ["input[name*='first' i]", "input[id*='first' i]", "input[autocomplete='given-name']"],
            [r"first name", r"given name"],
        ),
        "last_name": (
            "last_name",
            ["input[name*='last' i]", "input[id*='last' i]", "input[autocomplete='family-name']"],
            [r"last name", r"surname", r"family name"],
        ),
        "full_name": ("full_name", ["input[name='name']", "input[autocomplete='name']"], [r"full name"]),
        "email": (
            "email",
            ["input[type='email']", "input[name*='email' i]", "input[id*='email' i]"],
            [r"e-?mail"],
        ),
        "phone": (
            "phone",
            ["input[type='tel']", "input[name*='phone' i]", "input[id*='phone' i]"],
            [r"phone", r"mobile"],
        ),
        "linkedin": ("linkedin", ["input[name*='linkedin' i]", "input[id*='linkedin' i]"], [r"linkedin"]),
        "github": ("github", ["input[name*='github' i]"], [r"github"]),
        "location": ("location", ["input[name*='location' i]", "input[name*='city' i]"], [r"location", r"city"]),
    },
}

SUBMIT_SELECTORS = [
    "button[type='submit']",
    "input[type='submit']",
    "#submit_app",
    "button:has-text('Submit application')",
    "button:has-text('Submit Application')",
    "button:has-text('Submit')",
    "button:has-text('Send application')",
]

BOT_WALL_MARKERS = [
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[title*='challenge' i]",
    "#cf-challenge-running",
    "div[class*='cloudflare' i]",
]

LOGIN_MARKERS = [
    "input[type='password']",
    "button:has-text('Sign in')",
    "button:has-text('Create account')",
]


# ---------------------------------------------------------------------------
# Low-level fill helpers
# ---------------------------------------------------------------------------


def _try_fill(page, selectors: list[str], label_patterns: list[str], value: str) -> bool:
    """Try CSS selectors first, then accessible-label matching. Return True on success."""
    if not value:
        return False

    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=1500) and loc.is_editable(timeout=1500):
                loc.fill(value, timeout=5000)
                return True
        except Exception:  # noqa: BLE001 - a miss on one selector is expected
            continue

    for pattern in label_patterns:
        try:
            loc = page.get_by_label(re.compile(pattern, re.I)).first
            if loc.count() and loc.is_visible(timeout=1500):
                loc.fill(value, timeout=5000)
                return True
        except Exception:  # noqa: BLE001
            continue

    return False


def _upload_resume(page, resume_path: str) -> bool:
    if not resume_path or not os.path.exists(resume_path):
        return False

    # Prefer a file input whose surrounding text mentions the resume; fall back
    # to the first file input on the page.
    candidates = [
        "input[type='file'][name*='resume' i]",
        "input[type='file'][id*='resume' i]",
        "input[type='file'][name*='cv' i]",
        "input[type='file']",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count():
                # Many ATS hide the real input behind a styled button.
                loc.set_input_files(resume_path, timeout=10000)
                page.wait_for_timeout(2500)  # let the upload widget settle
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _find_blocking_wall(page) -> Optional[str]:
    for sel in BOT_WALL_MARKERS:
        try:
            if page.locator(sel).first.count():
                return "captcha"
        except Exception:  # noqa: BLE001
            pass
    for sel in LOGIN_MARKERS:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=1000):
                return "login_required"
        except Exception:  # noqa: BLE001
            pass
    return None


def _unfilled_required_fields(page) -> list[str]:
    """Report required inputs still empty, so the human knows what to finish."""
    out = []
    try:
        for loc in page.locator("input[required], select[required], textarea[required]").all()[:40]:
            try:
                if not loc.is_visible(timeout=500):
                    continue
                if (loc.input_value(timeout=500) or "").strip():
                    continue
                name = loc.get_attribute("name") or loc.get_attribute("id") or loc.get_attribute("aria-label")
                if name:
                    out.append(name)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# The actual browser session
# ---------------------------------------------------------------------------


def _run_application(
    job_url: str,
    candidate: dict,
    resume_path: str,
    submit: bool,
    headless: bool,
    timeout_ms: int,
) -> dict:
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    ats = detect_ats(job_url)
    stamp = int(time.time())
    shot_path = ARTIFACT_DIR / f"{ats}-{stamp}.png"

    result: dict[str, Any] = {
        "status": "error",
        "ats": ats,
        "filled_fields": [],
        "skipped_fields": [],
        "resume_uploaded": False,
        "unfilled_required": [],
        "screenshot": None,
        "error": None,
    }

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            page.goto(job_url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(2000)

            # Some boards put the form behind an "Apply" button on the listing.
            for label in ("Apply for this job", "Apply now", "Apply"):
                try:
                    btn = page.get_by_role("button", name=re.compile(label, re.I)).first
                    if btn.count() and btn.is_visible(timeout=1000):
                        btn.click(timeout=5000)
                        page.wait_for_timeout(2000)
                        break
                except Exception:  # noqa: BLE001
                    continue

            wall = _find_blocking_wall(page)
            if wall:
                page.screenshot(path=str(shot_path), full_page=False)
                result.update(
                    status="manual_required",
                    error=("A CAPTCHA is on the page." if wall == "captcha" else "The form is behind a login."),
                    screenshot=str(shot_path),
                )
                return result

            # --- fill -------------------------------------------------------
            field_map = FIELD_MAPS.get(ats, FIELD_MAPS["generic"])
            for canonical, (cand_key, selectors, labels) in field_map.items():
                value = str(candidate.get(cand_key) or "").strip()
                if not value:
                    result["skipped_fields"].append(canonical)
                    continue
                if _try_fill(page, selectors, labels, value):
                    result["filled_fields"].append(canonical)
                else:
                    result["skipped_fields"].append(canonical)

            # Greenhouse and Ashby sometimes only expose a full-name field; if we
            # filled nothing name-shaped, try the generic map as a second pass.
            if not {"first_name", "last_name", "full_name"} & set(result["filled_fields"]):
                for canonical in ("first_name", "last_name", "full_name"):
                    cand_key, selectors, labels = FIELD_MAPS["generic"][canonical]
                    if _try_fill(page, selectors, labels, str(candidate.get(cand_key) or "")):
                        result["filled_fields"].append(canonical)

            result["resume_uploaded"] = _upload_resume(page, resume_path)
            result["unfilled_required"] = _unfilled_required_fields(page)

            page.screenshot(path=str(shot_path), full_page=True)
            result["screenshot"] = str(shot_path)

            if not result["filled_fields"]:
                result.update(status="failed", error="No known fields matched. This page needs a custom adapter.")
                return result

            # --- submit -----------------------------------------------------
            if not submit:
                result["status"] = "filled_not_submitted"
                return result

            if result["unfilled_required"]:
                result.update(
                    status="blocked",
                    error=f"Required fields are still empty: {', '.join(result['unfilled_required'][:5])}",
                )
                return result

            clicked = False
            for sel in SUBMIT_SELECTORS:
                try:
                    btn = page.locator(sel).first
                    if btn.count() and btn.is_visible(timeout=1500) and btn.is_enabled(timeout=1500):
                        btn.click(timeout=8000)
                        clicked = True
                        break
                except Exception:  # noqa: BLE001
                    continue

            if not clicked:
                result.update(status="blocked", error="Could not find an enabled submit button.")
                return result

            page.wait_for_timeout(5000)
            body = (page.inner_text("body") or "").lower()
            confirmed = any(
                phrase in body
                for phrase in ("thank you", "application received", "we received", "successfully submitted")
            )
            page.screenshot(path=str(shot_path), full_page=True)
            result.update(
                status="submitted" if confirmed else "submitted_unconfirmed",
                screenshot=str(shot_path),
                final_url=page.url,
            )
            return result

        except PWTimeout as exc:
            result.update(status="failed", error=f"Timed out: {exc}")
            return result
        except Exception as exc:  # noqa: BLE001
            result.update(status="error", error=str(exc))
            return result
        finally:
            try:
                context.close()
                browser.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# The LangChain tool
# ---------------------------------------------------------------------------


@tool("apply_to_job", args_schema=ApplyToJobInput)
def apply_to_job(
    job_url: str,
    candidate: dict,
    resume_path: str,
    submit: bool = False,
    headless: bool = True,
    timeout_ms: int = 30000,
) -> dict:
    """Open a job application page in a real browser, fill the standard fields
    (first name, last name, email, phone, links) and upload the resume PDF.

    Returns a dict with:
      status  — submitted | submitted_unconfirmed | filled_not_submitted |
                blocked | manual_required | failed | unsupported | error
      filled_fields, skipped_fields, unfilled_required, resume_uploaded,
      screenshot (path to a full-page PNG), error

    With submit=False the form is filled and left alone — use this to verify
    selectors on a new job board before trusting it to send anything.
    """
    host = (urlparse(job_url).netloc or "").lower()
    for domain, reason in UNSUPPORTED.items():
        if domain in host:
            return {
                "status": "unsupported",
                "ats": domain,
                "error": reason,
                "filled_fields": [],
                "screenshot": None,
            }

    # Fresh thread: keeps Playwright's sync API away from any running event loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_application, job_url, candidate, resume_path, submit, headless, timeout_ms
        )
        try:
            return future.result(timeout=(timeout_ms / 1000) * 4 + 60)
        except concurrent.futures.TimeoutError:
            return {"status": "failed", "error": "Browser session exceeded its overall time budget.", "filled_fields": []}


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    demo_candidate = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "full_name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+91 90000 00000",
        "location": "Mumbai, India",
        "linkedin": "https://linkedin.com/in/example",
    }
    url = sys.argv[1] if len(sys.argv) > 1 else "https://jobs.lever.co/example/some-role"
    print(
        json.dumps(
            apply_to_job.invoke(
                {
                    "job_url": url,
                    "candidate": demo_candidate,
                    "resume_path": sys.argv[2] if len(sys.argv) > 2 else "",
                    "submit": False,
                    "headless": False,
                }
            ),
            indent=2,
        )
    )

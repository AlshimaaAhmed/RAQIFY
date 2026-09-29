import json
import os
import re

from groq import Groq

client = Groq(api_key="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")
MODEL = "qwen/qwen3-32b"


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — EXTRACTION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """\
You are a requirements analyst. Read raw client input and extract intent into JSON.

Sources: USER MESSAGE (casual/colloquial), AUDIO TRANSCRIPTION (speech noise), IMAGE EXTRACTION (OCR).
Ignore noise. Extract only what the client wants to build. Respond with ONLY valid JSON.

{
  "userIntent": "One clear sentence — what the client wants to build.",
  "goals": ["explicit goals only — never invent"],
  "constraints": ["tech, budget, deadlines, payment gateways mentioned"],
  "uiHints": ["colours, layouts, screens from any source"],
  "platform": "web | mobile | both | unspecified",
  "missingInfo": ["critical unknowns a developer must know before coding"]
}

Rules: English only. infer platform from context (app→mobile, website→web). Empty = [] or "unspecified".\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2A — GENERATION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_SYSTEM = """\
Think briefly. You are RAQIFY, a freelance proposal writer. \
Turn a structured intent object into a professional client-facing proposal. Respond with ONLY valid JSON.

{
  "projectTitle": "3-5 word project name",
  "summary": "2-3 sentence executive summary",
  "goals": ["goal 1", "goal 2"],
  "ambiguities": ["unclear things the developer must confirm"],
  "followUpQuestions": ["one question per ambiguity — same length as ambiguities"],
  "formattedBrief": "Full proposal in Markdown — escape newlines as \\n, quotes as \\""
}

formattedBrief sections:
# [Title]
## Project Overview — 2-3 sentences, speak to client directly ("We will build...")
## What's Included — bullet list: **[Feature]**: value to client
## What We Need From You — assets/credentials/decisions needed (omit if none)
## Timeline — phased: **Phase N – [name]** (X weeks): deliverable
## Next Steps — 2-3 sentences on how to proceed

Rules: never invent features. followUpQuestions = same length as ambiguities. Tone: professional but warm. English only.\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2B — REFINEMENT PROMPT
# ══════════════════════════════════════════════════════════════════════════════

REFINEMENT_SYSTEM = """\
Think briefly. You are RAQIFY updating an existing proposal. You receive: current brief, version history, timeline, new intent.
Respond with ONLY valid JSON.

{
  "projectTitle": "same unless rename requested",
  "summary": "updated summary",
  "goals": ["full updated list"],
  "ambiguities": ["NEW ambiguities only — [] if none"],
  "followUpQuestions": ["one per new ambiguity — [] if none"],
  "formattedBrief": "FULL updated proposal in Markdown — not a diff"
}

Rules: modify only relevant sections. preserve unchanged content exactly. English only. escape newlines \\n, quotes \\".\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _chat(system: str, user: str, max_tokens: int = 4096) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\n\nRaw output:\n{raw}") from e


def _build_raw_input(text: str, attachments: list) -> str:
    parts = []
    if text:
        parts.append("[USER MESSAGE]\n" + text)
    for att in attachments:
        tag = (
            f"[IMAGE EXTRACTION — {att['name']}]"
            if att["type"] == "IMAGE"
            else f"[AUDIO TRANSCRIPTION — {att['name']}]"
        )
        parts.append(f"{tag}\n{att['extractedText']}")
    return "\n\n---\n\n".join(parts) if parts else "(no input provided)"


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_brief(raw_text: str, attachment_texts: list) -> dict:
    raw_input  = _build_raw_input(raw_text, attachment_texts)
    intent_raw = _chat(EXTRACTION_SYSTEM, raw_input, max_tokens=1000)
    intent     = _parse_json(intent_raw)

    brief_raw  = _chat(GENERATION_SYSTEM, json.dumps(intent, ensure_ascii=False), max_tokens=8192)
    result     = _parse_json(brief_raw)
    return result


def refine_brief(
    current_brief: str,
    current_version: int,
    version_history: list,
    timeline: list,
    raw_new_request: str,
    new_attachments: list,
) -> dict:
    raw_input  = _build_raw_input(raw_new_request, new_attachments)
    intent_raw = _chat(EXTRACTION_SYSTEM, raw_input, max_tokens=1000)
    intent = _parse_json(intent_raw)

    history_lines = "\n".join(f"- v{v['version']}: {v['summary']}" for v in version_history)
    convo_lines   = "\n".join(
        f"{'Client' if m['sender'] == 'CLIENT' else 'Developer'}: {m['message']}"
        for m in timeline
    )

    user_msg = (
        f"## Current Brief (v{current_version})\n{current_brief}\n\n"
        f"## Version History\n{history_lines or 'None'}\n\n"
        f"## Conversation Timeline\n{convo_lines or 'None'}\n\n"
        f"## New Extracted Intent\n{json.dumps(intent, ensure_ascii=False)}"
    )

    brief_raw = _chat(REFINEMENT_SYSTEM, user_msg, max_tokens=8192)
    result    = _parse_json(brief_raw)
    return result
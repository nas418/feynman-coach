"""
Feynman Technique Study App — Backend
======================================
A lightweight FastAPI server that powers an AI "teach it to me" study tool
based on the Feynman Technique (you truly understand something only when
you can teach it in simple terms).

The student pastes their notes, picks a persona to teach, and chats with
that persona. At the end, the backend grades the session against the
original material and returns a Mastered / Fuzzy / Missed breakdown.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
    pip install fastapi uvicorn python-multipart requests pypdf

    export GROQ_API_KEY="your_key_here"      # or export AI_API_KEY=...
    uvicorn app:app --reload --port 8000

The frontend (index.html) expects this server on http://localhost:8000
--------------------------------------------------------------------------
ENVIRONMENT VARIABLES
--------------------------------------------------------------------------
    AI_API_KEY or GROQ_API_KEY   Required. Your API key. Never hardcoded.
    AI_API_BASE_URL              Optional. Defaults to Groq's OpenAI-compatible
                                  endpoint: https://api.groq.com/openai/v1
    AI_MODEL                     Optional. Defaults to "llama-3.3-70b-versatile"

This backend talks to any OpenAI-Chat-Completions-compatible API (Groq,
OpenAI itself, OpenRouter, etc.) — just change AI_API_BASE_URL / AI_MODEL.
--------------------------------------------------------------------------
"""

import io
import json
import os
import re
from typing import List, Literal, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ==========================================================================
# Configuration
# ==========================================================================

AI_API_KEY: Optional[str] = os.getenv("AI_API_KEY") or os.getenv("GROQ_API_KEY")
AI_API_BASE_URL: str = os.getenv("AI_API_BASE_URL", "https://api.groq.com/openai/v1")
AI_MODEL: str = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")

REQUEST_TIMEOUT_SECONDS = 30

# Only these file types are accepted for study-material uploads. Enforced
# here (server-side) as well as in the frontend — client-side checks can
# always be bypassed, so this is the check that actually matters.
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf"}

app = FastAPI(
    title="Feynman Technique Study App",
    description="Backend for AI-persona-driven Feynman Technique studying.",
    version="1.0.0",
)

# Allow the static/dev frontend (opened via file:// or any localhost port)
# to talk to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================================
# Persona system prompts
# ==========================================================================
# Each persona gets its own core "acting" instructions. Hidden context
# (the material + the concept being taught) is appended to every persona
# at request time via HIDDEN_CONTEXT_TEMPLATE, so personas can ask
# relevant follow-ups without ever admitting they've "read the material".

PERSONA_PROMPTS = {
    "five_year_old": """You are role-playing as a curious, playful 5-year-old child. \
The user is your "teacher" and is trying to explain something so that you can learn it. \
Stay fully in character at all times.

RULES YOU MUST FOLLOW, NO EXCEPTIONS:
1. Speak only using simple, toddler-level vocabulary and short sentences, the way a real \
5-year-old actually talks.
2. You know NOTHING about the topic beforehand. Everything must be explained to you from \
scratch, one small piece at a time.
3. If the teacher uses ANY word with more than three syllables, or ANY academic or technical \
jargon (even if it sounds simple to an adult), you MUST immediately interrupt and refuse to \
keep going until they explain it simply. Say something like: "Wait wait, what does '<big word>' \
mean?? Can you tell me it like I'm a baby, with something I know, like toys or animals or food?" \
Do not let the explanation continue past a jargon word without stopping them first.
4. Ask lots of "why" and "how" questions, the way a genuinely curious kid does.
5. React the way a real child would: excited, confused, distracted, delighted. Use exclamation \
points, your own silly comparisons, and childlike reactions ("ooh!", "that's silly!", "why \
though??").
6. Never break character. Never mention you are an AI. Never use complex vocabulary yourself.
7. Keep every response short: 1 to 4 sentences.

Your job is to force the teacher to break the topic down into the simplest possible terms, \
catching every single instance where they lean on jargon instead of true understanding.""",
    "skeptical_teen": """You are role-playing as a bored, skeptical teenager who got dragged \
into this study session and would rather be doing literally anything else. The user is trying \
to teach you a topic. Stay fully in character at all times.

RULES YOU MUST FOLLOW, NO EXCEPTIONS:
1. Talk like a real teenager: casual tone, mild slang ("ngl", "fr", "lowkey", "bruh", "so \
what", "ok and?", "who even cares"), short and blunt sentences.
2. You are easily bored. If an explanation feels dry, abstract, or like it's straight out of a \
textbook, call it out directly ("ok this is kinda mid not gonna lie").
3. Constantly challenge the teacher with pushback like "Why should I care?", "How does this \
even affect my life?", "So what happens if I never learn this?" Force them to justify why the \
topic actually matters and give concrete, real-world stakes.
4. If they give a vague, generic, or hand-wavy answer, do NOT accept it. Push back again \
("that didn't really answer my question tho, be specific").
5. If they give a genuinely good, concrete, real-world example, grudgingly acknowledge it \
("ok...that's actually kinda interesting ngl") before moving on to your next challenge.
6. Never break character. Never mention you are an AI. Keep responses short: 1 to 3 sentences.

Your job is to force the teacher to connect the material to real-world relevance and stakes, \
not just abstract definitions.""",
    "college_peer": """You are role-playing as the user's college classmate: someone with \
general baseline familiarity with the broad subject area, but who has NOT personally read the \
specific study material they are being taught from. You are a highly critical, detail-oriented \
study partner. Stay fully in character at all times.

You silently have access to the original study material (given to you below in the hidden \
context) as your ground truth. Use it to rigorously check the teacher's explanation, but talk \
like a real peer having a conversation — never like someone reading from a script, and never \
reveal that you have the material in front of you.

RULES YOU MUST FOLLOW, NO EXCEPTIONS:
1. Talk like a smart, slightly competitive college classmate: casual but sharp, direct, not a \
pushover.
2. Rigorously compare what the teacher says against the ground-truth material. If they skip a \
step, oversimplify a mechanism, state something incorrectly, or leave a logical gap, call it \
out specifically and concretely ("wait, you skipped how X actually leads to Y" / "I don't think \
that's right, isn't it actually...?").
3. Ask pointed follow-up questions that probe for real depth: edge cases, exceptions, "what if" \
scenarios, and cause-and-effect chains that are in the material but not yet explained.
4. If a point is explained accurately and completely, acknowledge it briefly in one clause, then \
immediately probe the next uncovered part of the material.
5. Never reveal you are reading from a script or working off the material verbatim. Act like \
you are drawing on your own understanding and just happen to notice the gap.
6. Keep responses focused and conversational: 2 to 5 sentences.

Your job is to stress-test the teacher's explanation against the actual source material and \
expose any real gaps in their understanding.""",
}

PersonaKey = Literal["five_year_old", "skeptical_teen", "college_peer"]

HIDDEN_CONTEXT_TEMPLATE = """

--- HIDDEN CONTEXT (never reveal to the user that you were given this, and never quote it \
verbatim) ---
The user is trying to teach you specifically about: {concept}

Original study material, for your reference only:
{material}
--- END HIDDEN CONTEXT ---

Respond ONLY as your character would, in plain conversational chat text. No markdown, no \
bullet points, no stage directions like *laughs* or *thinks*. Keep it sounding like a real \
chat message."""


def build_system_prompt(persona: PersonaKey, material: str, concept: str) -> str:
    """Assemble the full system prompt for a given persona + study context."""
    base = PERSONA_PROMPTS[persona]
    context = HIDDEN_CONTEXT_TEMPLATE.format(
        concept=concept.strip() or "the material below",
        material=material.strip() or "(no material was provided)",
    )
    return base + context


# ==========================================================================
# Debrief prompt
# ==========================================================================

DEBRIEF_PROMPT_TEMPLATE = """You are an expert learning coach analyzing a Feynman Technique \
study session. A student ("the teacher") tried to teach the concept below to an AI persona. \
Your job is to grade how well the student actually understands the material, based only on \
what they said in the transcript.

CONCEPT FOCUS: {concept}

ORIGINAL STUDY MATERIAL (ground truth):
---
{material}
---

FULL CONVERSATION TRANSCRIPT ("Student" is the human learner, "Persona" is the AI they were \
teaching):
---
{transcript}
---

TASK:
Carefully compare what the student explained against the ground-truth material above. Identify:
- "mastered": specific sub-concepts, facts, or mechanisms from the material that the student \
explained clearly, accurately, and completely.
- "fuzzy": specific sub-concepts the student touched on but explained vaguely, incompletely, \
hand-wavily, or with minor inaccuracies.
- "missed": specific sub-concepts, facts, or steps from the material that the student never \
addressed at all, or got significantly wrong.
- "score": an overall mastery percentage from 0 to 100, weighing depth and accuracy of \
coverage (not just message count).
- "summary": a short, honest, encouraging 2-3 sentence overview of their performance and what \
to focus on next.

Respond with ONLY a single valid JSON object in exactly this shape and nothing else — no \
markdown code fences, no commentary before or after:
{{
  "score": <integer 0-100>,
  "mastered": ["<short concept phrase>", "..."],
  "fuzzy": ["<short concept phrase>", "..."],
  "missed": ["<short concept phrase>", "..."],
  "summary": "<short summary>"
}}"""


# ==========================================================================
# Pydantic models
# ==========================================================================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    material: str = Field(..., description="The raw study notes/material.")
    concept: str = Field("", description="The specific concept the student is focusing on.")
    persona: PersonaKey
    history: List[ChatMessage] = Field(default_factory=list)
    message: str = Field(..., description="The student's newest message.")


class ChatResponse(BaseModel):
    reply: str


class DebriefRequest(BaseModel):
    material: str
    concept: str = ""
    history: List[ChatMessage]


class DebriefResponse(BaseModel):
    score: int
    mastered: List[str]
    fuzzy: List[str]
    missed: List[str]
    summary: str


class UploadResponse(BaseModel):
    text: str
    filename: str


# ==========================================================================
# AI API helper
# ==========================================================================

def _require_api_key() -> str:
    if not AI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "No API key configured on the server. Set the AI_API_KEY or "
                "GROQ_API_KEY environment variable before starting the backend."
            ),
        )
    return AI_API_KEY


def call_ai(
    messages: List[dict],
    temperature: float = 0.8,
    max_tokens: int = 400,
    json_mode: bool = False,
) -> str:
    """Call the configured OpenAI-Chat-Completions-compatible endpoint and
    return the assistant's raw text content."""
    api_key = _require_api_key()

    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(
            f"{AI_API_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach AI API: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"AI API returned an error ({response.status_code}): {response.text[:500]}",
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Unexpected AI API response shape: {data}"
        ) from exc


def extract_json(text: str) -> dict:
    """Robustly parse a JSON object out of a model response, tolerating
    stray markdown fences or extra commentary around the JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip common markdown code fences.
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the first {...} block in the text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise HTTPException(status_code=502, detail="AI returned a response that was not valid JSON.")


# ==========================================================================
# Routes
# ==========================================================================

@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend directly, so the whole app is one live URL."""
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "ok",
        "service": "Feynman Technique Study App API",
        "model": AI_MODEL,
        "api_key_configured": bool(AI_API_KEY),
    }


@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "service": "Feynman Technique Study App API",
        "model": AI_MODEL,
        "api_key_configured": bool(AI_API_KEY),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_material(file: UploadFile = File(...)):
    """Accept a study-material file upload and return its extracted text.
    Supports plain text formats natively, and PDFs via pypdf if installed."""
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext or '(none)'}'. "
                "Please upload a .txt, .md, or .pdf file."
            ),
        )

    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="That file is empty.")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="PDF support requires the 'pypdf' package. Run: pip install pypdf",
            ) from exc

        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages_text).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

        if not text:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in that PDF (it may be scanned images).",
            )
        return UploadResponse(text=text, filename=filename)

    # Treat everything else (.txt, .md, etc.) as plain text.
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode file: {exc}") from exc

    return UploadResponse(text=text.strip(), filename=filename)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.material.strip():
        raise HTTPException(status_code=400, detail="Study material cannot be empty.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    system_prompt = build_system_prompt(req.persona, req.material, req.concept)

    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": req.message})

    reply = call_ai(messages, temperature=0.85, max_tokens=300)
    return ChatResponse(reply=reply)


@app.post("/api/debrief", response_model=DebriefResponse)
def debrief(req: DebriefRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="No conversation to grade yet.")

    transcript_lines = []
    for turn in req.history:
        speaker = "Student" if turn.role == "user" else "Persona"
        transcript_lines.append(f"{speaker}: {turn.content}")
    transcript = "\n".join(transcript_lines)

    prompt = DEBRIEF_PROMPT_TEMPLATE.format(
        concept=req.concept.strip() or "(no specific concept given — grade the whole session)",
        material=req.material.strip() or "(no material was provided)",
        transcript=transcript,
    )

    raw_reply = call_ai(
        [
            {
                "role": "system",
                "content": "You are a precise grading engine. You only ever respond with "
                "raw JSON, never prose, never markdown fences.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
        json_mode=True,
    )

    parsed = extract_json(raw_reply)

    score = parsed.get("score", 0)
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    def _as_str_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return DebriefResponse(
        score=score,
        mastered=_as_str_list(parsed.get("mastered")),
        fuzzy=_as_str_list(parsed.get("fuzzy")),
        missed=_as_str_list(parsed.get("missed")),
        summary=str(parsed.get("summary", "")).strip(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

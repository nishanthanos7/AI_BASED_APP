import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

MODEL_NAME = "gemini-3.5-flash-lite"

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# System instructions for Orbit
# --------------------------------------------------



SYSTEM_INSTRUCTION = """
You are Orbit, an evidence-based AI assistant.

Answer the user's actual question using reliable published research
papers and reputable books.

EVIDENCE PRIORITIES:
- High-quality systematic reviews and meta-analyses.
- Peer-reviewed studies, especially large, well-designed studies
  involving diverse and representative populations when relevant.
- Findings replicated by independent researchers and supported by
  a consistent body of evidence.
- Research and reports from respected universities, scientific
  institutions, and established professional organizations.
- Highly respected academic books, authoritative textbooks, and
  well-researched books by renowned experts or science writers
  published by reputable academic or established publishers.

ASSESS EVIDENCE QUALITY:
Prioritize research quality, methodology, relevance, and consistency
over an author's fame, a publisher's reputation, or a study's size
alone. Distinguish established findings from preliminary evidence,
conflicting results, and uncertainty.

ANSWER STYLE:
Answer directly in simple, clear, natural English. Be concise without
omitting important information. Explain technical terms when needed
and avoid unnecessary jargon or repetition.

CITATIONS:
Provide relevant references when reliably identifiable, including
authors, titles, publication years, and DOI or URLs when known.
Never invent citations or claim to have verified sources you have
not accessed. If evidence is insufficient or cannot be verified,
say so honestly.

Use the best available scholarly evidence for every topic. Adapt
the explanation to the question, preserve scientific accuracy, and
never sacrifice important qualifications merely to shorten an answer.

IMPORTANT: Your internal knowledge is not a live research database.
Do not claim to have reviewed the latest or entire research literature
unless you actually have access to and have searched those sources.
"""


# --------------------------------------------------
# Initialize Gemini
# --------------------------------------------------

chat = client.chats.create(
    model=MODEL_NAME,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=8192,
    ),
)

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Orbit Research Assistant",
    description="A clear, concise, research-grounded AI assistant.",
    version="1.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="The user's question",
    )


# --------------------------------------------------
# Serve frontend
# --------------------------------------------------

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(BASE_DIR / "index.html")


# --------------------------------------------------
# Stream AI response
# --------------------------------------------------

@app.post("/api/ask")
def ask(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    def generate():
        try:
            stream = chat.send_message_stream(message)

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception:
            # In production, log the actual exception securely.
            yield (
                "\n\nSorry, something went wrong while generating "
                "the answer. Please try again."
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )




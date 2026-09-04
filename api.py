import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import run_investigation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="AI Data Analyst API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    question: str


class AnalyzeResponse(BaseModel):
    question: str
    plan: list
    queries: list
    observations: list
    findings: list
    final_answer: dict
    charts: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    logger.info("Received analyze request: %.80s", req.question)
    try:
        state = run_investigation(req.question.strip())
        return {
            "question": req.question,
            "plan": state.get("plan", []),
            "queries": state.get("queries", []),
            "observations": state.get("observations", []),
            "findings": state.get("findings", []),
            "final_answer": state.get("final_answer", {}),
            "charts": state.get("charts", []),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Investigation failed: %s", e, exc_info=True)
        return {"error": type(e).__name__, "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)

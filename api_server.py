import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from contextlib import asynccontextmanager

# Import the main analyzer from the new pipeline
from pipeline.analyzer import PhishingAnalyzer

# Global analyzer instance
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    # Initialize the analyzer once at startup
    # It will automatically load the NLP models and ML weights
    analyzer = PhishingAnalyzer("config/settings.yaml")
    yield

app = FastAPI(
    title="Phishing Filter Microservice",
    description="Deep Content Inspection Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "phishing-filter"}

@app.post("/scan")
async def scan_eml(file: UploadFile = File(...)):
    """
    Accepts a .eml file upload, runs the 4-engine pipeline,
    and returns a JSON format that the Orchestrator expects.
    """
    temp_file_path = None
    try:
        # Create a secure temporary file to write raw upload contents
        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Read contents back as string for the Analyzer
        with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_email_str = f.read()

        # Run the full pipeline
        verdict = analyzer.analyze(raw_email_str)

        # Structure response strictly for the Orchestrator
        # Orchestrator expects: {"verdict": "PHISHING"|"CLEAN", "note": "...", "raw_report": {...}}
        final_verdict = verdict.label.value  # "MALICIOUS" or "CLEAN"
        # Map "MALICIOUS" to "PHISHING" for the orchestrator
        if final_verdict == "MALICIOUS":
            final_verdict = "PHISHING"
            
        score = int(round(verdict.final_score))
        
        # Compile explainable reasons for the note
        note = f"Score: {score}/100. Aggregator: {verdict.aggregator_used}."
        if verdict.nlp_result.trigger_phrases:
            note += f" Trigger phrases found: {', '.join(verdict.nlp_result.trigger_phrases[:3])}."

        # Convert Pydantic results to deeply nested JSON dictionaries
        details = {
            "confidence": verdict.confidence,
            "analysis_time_ms": verdict.analysis_duration_ms,
            "total_flags": verdict.feature_vector.total_flags_triggered,
            "engine_1_header": verdict.header_result.model_dump(),
            "engine_2_structure": verdict.structural_result.model_dump(),
            "engine_3_nlp": {
                "score": verdict.nlp_result.score,
                "phishing_probability": verdict.nlp_result.phishing_probability,
                "predicted_intent": verdict.nlp_result.predicted_intent.value if verdict.nlp_result.predicted_intent else None,
                "urgency_score": verdict.nlp_result.urgency_score,
                "credential_harvesting": verdict.nlp_result.credential_harvesting_score,
                "financial_fraud": verdict.nlp_result.financial_fraud_score,
                "trigger_phrases": verdict.nlp_result.trigger_phrases
            },
            "engine_4_links": verdict.link_result.model_dump(),
            "feature_vector": verdict.feature_vector.model_dump()
        }

        return {
            "verdict": final_verdict,
            "score": score,
            "note": note,
            "details": details,
            "raw_report": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phishing scan failed: {str(e)}")
        
    finally:
        # Securely clean up the temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8002)

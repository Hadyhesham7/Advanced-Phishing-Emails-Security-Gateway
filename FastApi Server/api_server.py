import os
import sys
import tempfile
import logging
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path so we can import the pipeline module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pipeline.analyzer import PhishingAnalyzer

app = FastAPI(title="Phishing Filter API", description="Member 2 - Phishing Filter for EPG")

# Global analyzer instance
analyzer: PhishingAnalyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    logger.info("Initializing Phishing Analyzer...")
    
    # Temporarily change working directory to parent_dir to allow config/settings.yaml to load correctly
    original_cwd = os.getcwd()
    os.chdir(parent_dir)
    try:
        # Load ML models ONCE at startup as requested
        analyzer = PhishingAnalyzer("config/settings.yaml")
        logger.info("Phishing Analyzer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PhishingAnalyzer: {e}", exc_info=True)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return {"status": "healthy", "service": "phishing-filter"}

@app.post("/scan")
async def scan_email(file: UploadFile = File(...)) -> Dict[str, Any]:
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
    tmp_path = None
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Read back as string since analyzer.analyze expects raw email string
        raw_email = content.decode('utf-8', errors='replace')
        
        # Change working directory to parent temporarily so models can be found during inference if needed
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        try:
            # Run detection logic
            verdict = analyzer.analyze(raw_email)
        finally:
            os.chdir(original_cwd)
        
        # The EPG orchestrator expects: {"verdict": "PHISHING" or "CLEAN", "score": 0-100, "note": "reason", "details": {}}
        is_phishing = verdict.action == "DROP" or verdict.final_score >= 60.0
        verdict_str = "PHISHING" if is_phishing else "CLEAN"
        
        h = verdict.header_result
        s = verdict.structural_result
        n = verdict.nlp_result
        l = verdict.link_result

        return {
            "verdict": verdict_str,
            "score": float(verdict.final_score),
            "note": f"Action: {verdict.action}, Confidence: {verdict.confidence:.1%}",
            "details": {
                "engine_1_header": {
                    "score": float(h.score),
                    "display_name_spoofing": h.display_name_spoofing_detected,
                    "claimed_name": h.display_name_claimed,
                    "actual_domain": h.display_name_actual_domain,
                    "reply_to_mismatch": h.reply_to_mismatch_detected,
                    "reply_to_address": h.reply_to_address,
                    "suspicious_xmailer": h.xmailer_anomaly_detected,
                    "auth_failure": h.auth_failure_detected
                },
                "engine_2_structure": {
                    "score": float(s.score),
                    "hidden_text": s.hidden_text_detected,
                    "hidden_text_methods": s.hidden_text_methods,
                    "zero_width_chars_detected": s.zero_width_chars_detected,
                    "zero_width_chars_count": s.zero_width_char_count,
                    "brand_impersonation": s.brand_impersonation_detected,
                    "impersonated_brand": s.impersonated_brand,
                    "credential_form": s.credential_form_detected,
                    "credential_details": s.credential_form_details,
                    "macro_detected": s.macro_detected
                },
                "engine_3_nlp": {
                    "score": float(n.score),
                    "phishing_probability": float(n.phishing_probability),
                    "predicted_intent": n.predicted_intent,
                    "urgency_score": float(n.urgency_coercion_score),
                    "credential_harvesting": float(n.credential_harvesting_score),
                    "financial_fraud_bec": float(n.financial_fraud_bec_score),
                    "oauth_consent_score": float(n.oauth_consent_score),
                    "trigger_phrases": n.trigger_phrases
                },
                "engine_4_links": {
                    "score": float(l.score),
                    "total_links": l.total_links_found,
                    "href_mismatch": l.href_mismatch_detected,
                    "mismatch_details": l.mismatched_links,
                    "url_obfuscation": l.url_obfuscation_detected,
                    "suspicious_tlds": l.suspicious_tlds,
                    "image_wrapped_links": l.image_wrapped_link_detected,
                    "login_url_pattern": l.login_url_pattern_detected
                },
                "aggregator_summary": {
                    "aggregator_used": verdict.aggregator_used,
                    "total_flags_triggered": verdict.feature_vector.total_flags_triggered,
                    "analysis_duration_ms": float(verdict.analysis_duration_ms)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error processing email: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

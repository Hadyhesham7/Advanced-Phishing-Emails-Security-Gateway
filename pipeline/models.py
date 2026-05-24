# ============================================================
# Phishing Detection Pipeline - Data Models
# Pydantic schemas for inter-engine communication
# ============================================================

from __future__ import annotations

from enum import Enum
from dataclasses import field
from typing import Optional
from pydantic import BaseModel, Field


# ── Verdict Enums ───────────────────────────────────────────

class VerdictLabel(str, Enum):
    CLEAN = "CLEAN"
    MALICIOUS = "MALICIOUS"


class VerdictAction(str, Enum):
    DELIVER = "DELIVER"
    DROP = "DROP"


class NLPIntentCategory(str, Enum):
    URGENCY_COERCION = "urgency_coercion"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    FINANCIAL_FRAUD_BEC = "financial_fraud_bec"
    OAUTH_CONSENT_PHISHING = "oauth_consent_phishing"
    GENERIC_PHISHING = "generic_phishing"
    LEGITIMATE = "legitimate"


# ── Engine Result Models ────────────────────────────────────

class HeaderAnalysisResult(BaseModel):
    """Output from Engine 1: Advanced Header Analysis."""

    score: float = Field(0.0, ge=0, le=25, description="Weighted header risk score")

    # Individual detection flags & sub-scores
    display_name_spoofing_detected: bool = False
    display_name_spoofing_score: float = 0.0
    display_name_claimed: Optional[str] = None
    display_name_actual_domain: Optional[str] = None

    reply_to_mismatch_detected: bool = False
    reply_to_mismatch_score: float = 0.0
    reply_to_address: Optional[str] = None

    xmailer_anomaly_detected: bool = False
    xmailer_anomaly_score: float = 0.0
    xmailer_value: Optional[str] = None

    domain_age_days: Optional[int] = None
    new_domain_flag: bool = False
    new_domain_score: float = 0.0

    typosquatting_detected: bool = False
    typosquatting_score: float = 0.0
    typosquatting_target_brand: Optional[str] = None
    typosquatting_similarity: Optional[float] = None

    # Authentication headers (SPF/DKIM/DMARC)
    auth_failure_detected: bool = False
    auth_failure_score: float = 0.0

    raw_signals: dict = Field(default_factory=dict)


class StructuralAnalysisResult(BaseModel):
    """Output from Engine 2: Structural & HTML Analysis."""

    score: float = Field(0.0, ge=0, le=20, description="Weighted structural risk score")

    hidden_text_detected: bool = False
    hidden_text_score: float = 0.0
    hidden_text_methods: list[str] = Field(default_factory=list)
    hidden_text_content: Optional[str] = None

    zero_width_chars_detected: bool = False
    zero_width_chars_score: float = 0.0
    zero_width_char_count: int = 0
    cleaned_text: Optional[str] = None

    image_only_email: bool = False
    image_only_score: float = 0.0
    text_to_image_ratio: Optional[float] = None
    ocr_extracted_text: Optional[str] = None

    brand_impersonation_detected: bool = False
    brand_impersonation_score: float = 0.0
    impersonated_brand: Optional[str] = None

    # Credential form detection
    credential_form_detected: bool = False
    credential_form_score: float = 0.0
    credential_form_details: list[str] = Field(default_factory=list)

    # Macro attachment detection
    macro_detected: bool = False
    macro_score: float = 0.0
    macro_details: list[str] = Field(default_factory=list)

    raw_signals: dict = Field(default_factory=dict)


class NLPAnalysisResult(BaseModel):
    """Output from Engine 3: NLP & Semantic Analysis."""

    score: float = Field(0.0, ge=0, le=30, description="NLP-derived risk score")

    phishing_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Raw phishing probability from the transformer model"
    )
    predicted_intent: NLPIntentCategory = NLPIntentCategory.LEGITIMATE
    intent_confidence: float = 0.0

    urgency_coercion_score: float = 0.0
    credential_harvesting_score: float = 0.0
    financial_fraud_bec_score: float = 0.0

    oauth_consent_score: float = 0.0

    # Key phrases that triggered detection
    trigger_phrases: list[str] = Field(default_factory=list)

    raw_signals: dict = Field(default_factory=dict)


class LinkAnalysisResult(BaseModel):
    """Output from Engine 4: Static Link & Payload Analysis."""

    score: float = Field(0.0, ge=0, le=25, description="Weighted link risk score")

    total_links_found: int = 0

    href_mismatch_detected: bool = False
    href_mismatch_score: float = 0.0
    mismatched_links: list[dict] = Field(default_factory=list)

    url_obfuscation_detected: bool = False
    url_obfuscation_score: float = 0.0
    obfuscated_urls: list[str] = Field(default_factory=list)
    shortener_urls: list[str] = Field(default_factory=list)

    homograph_attack_detected: bool = False
    homograph_attack_score: float = 0.0
    homograph_urls: list[dict] = Field(default_factory=list)

    suspicious_tlds: list[str] = Field(default_factory=list)

    # Additional detection flags
    suspicious_tld_score: float = 0.0

    image_wrapped_link_detected: bool = False
    image_wrapped_link_score: float = 0.0

    login_url_pattern_detected: bool = False
    login_url_pattern_score: float = 0.0

    raw_signals: dict = Field(default_factory=dict)


# ── Aggregated Pipeline Result ──────────────────────────────

class AggregatorFeatureVector(BaseModel):
    """Feature vector fed into the ML Aggregator (XGBoost)."""

    # Engine raw scores (continuous)
    header_score: float = 0.0
    structural_score: float = 0.0
    nlp_score: float = 0.0
    link_score: float = 0.0

    # Binary detection flags
    display_name_spoofing: int = 0
    reply_to_mismatch: int = 0
    xmailer_anomaly: int = 0
    new_domain: int = 0
    typosquatting: int = 0
    hidden_text: int = 0
    zero_width_chars: int = 0
    image_only: int = 0
    brand_impersonation: int = 0
    href_mismatch: int = 0
    url_obfuscation: int = 0
    homograph_attack: int = 0

    # Continuous confidence values
    nlp_phishing_probability: float = 0.0
    nlp_intent_confidence: float = 0.0
    urgency_score: float = 0.0
    credential_harvesting_score: float = 0.0
    financial_fraud_score: float = 0.0

    # Cross-engine interaction features
    header_x_link_risk: float = 0.0       # header_score * link_score
    nlp_x_structure_risk: float = 0.0     # nlp_score * structural_score
    total_flags_triggered: int = 0        # Count of binary flags = 1

    # New features (v2)
    credential_form_detected: int = 0
    image_wrapped_links: int = 0
    suspicious_tld_count: int = 0
    login_url_pattern: int = 0
    oauth_consent_score: float = 0.0
    auth_failure_detected: int = 0

    def to_feature_array(self) -> list[float]:
        """Convert to ordered numeric array for ML model input."""
        return [
            self.header_score,
            self.structural_score,
            self.nlp_score,
            self.link_score,
            float(self.display_name_spoofing),
            float(self.reply_to_mismatch),
            float(self.xmailer_anomaly),
            float(self.new_domain),
            float(self.typosquatting),
            float(self.hidden_text),
            float(self.zero_width_chars),
            float(self.image_only),
            float(self.brand_impersonation),
            float(self.href_mismatch),
            float(self.url_obfuscation),
            float(self.homograph_attack),
            self.nlp_phishing_probability,
            self.nlp_intent_confidence,
            self.urgency_score,
            self.credential_harvesting_score,
            self.financial_fraud_score,
            self.header_x_link_risk,
            self.nlp_x_structure_risk,
            float(self.total_flags_triggered),
            float(self.credential_form_detected),
            float(self.image_wrapped_links),
            float(self.suspicious_tld_count),
            float(self.login_url_pattern),
            self.oauth_consent_score,
            float(self.auth_failure_detected),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered feature names matching to_feature_array()."""
        return [
            "header_score", "structural_score", "nlp_score", "link_score",
            "display_name_spoofing", "reply_to_mismatch", "xmailer_anomaly",
            "new_domain", "typosquatting", "hidden_text", "zero_width_chars",
            "image_only", "brand_impersonation", "href_mismatch",
            "url_obfuscation", "homograph_attack", "nlp_phishing_probability",
            "nlp_intent_confidence", "urgency_score",
            "credential_harvesting_score", "financial_fraud_score",
            "header_x_link_risk", "nlp_x_structure_risk",
            "total_flags_triggered",
            "credential_form_detected", "image_wrapped_links",
            "suspicious_tld_count", "login_url_pattern",
            "oauth_consent_score", "auth_failure_detected",
        ]


class PipelineVerdict(BaseModel):
    """Final verdict from the Phishing Detection Pipeline."""

    final_score: float = Field(..., ge=0, le=100)
    label: VerdictLabel
    action: VerdictAction
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Per-engine breakdown
    header_result: HeaderAnalysisResult
    structural_result: StructuralAnalysisResult
    nlp_result: NLPAnalysisResult
    link_result: LinkAnalysisResult

    # ML Aggregator details
    feature_vector: AggregatorFeatureVector
    aggregator_used: str = "heuristic"  # "xgboost" | "heuristic"

    # Actionable metadata
    route_to_sandbox: bool = False
    extract_iocs: bool = False

    # Audit trail
    analysis_duration_ms: float = 0.0
    engine_timings: dict[str, float] = Field(default_factory=dict)

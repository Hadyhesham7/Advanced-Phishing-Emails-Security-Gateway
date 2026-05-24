# ============================================================
# Engine 3: NLP & Semantic Analysis (Intent Recognition)
# ============================================================
# Detection Targets:
#   - Urgency & Coercion
#   - Credential Harvesting Intent
#   - Financial Fraud / BEC (Business Email Compromise)
#
# ML Requirement: ★★★ HEAVY — This is the primary ML engine
#   - Requires: Fine-tuned BERT/RoBERTa transformer model
#   - Dataset:  Text corpus (phishing + legitimate emails)
#   - Training: Supervised BINARY classification (phishing vs legitimate)
#   - Fallback: Keyword/regex heuristics when model unavailable
# ============================================================

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from pipeline.models import NLPAnalysisResult, NLPIntentCategory


# ── Urgency & Coercion Keyword Patterns ─────────────────────
# Used as FALLBACK when the transformer model is unavailable,
# and as supplementary signal boosters alongside the model.
URGENCY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:urgent|immediately|right\s+away|asap)\b", re.I),
    re.compile(r"\b(?:account\s+(?:suspended|locked|disabled|terminated))\b", re.I),
    re.compile(r"\b(?:within\s+\d+\s+(?:hours?|minutes?|days?))\b", re.I),
    re.compile(r"\b(?:act\s+now|don'?t\s+delay|time\s+(?:is\s+)?limited)\b", re.I),
    re.compile(r"\b(?:final\s+(?:warning|notice|attempt))\b", re.I),
    re.compile(r"\b(?:failure\s+to\s+(?:respond|verify|confirm|act))\b", re.I),
    re.compile(r"\b(?:unauthorized\s+(?:access|activity|login|transaction))\b", re.I),
    re.compile(r"\b(?:security\s+(?:alert|breach|incident|threat))\b", re.I),
    re.compile(r"\b(?:your\s+account\s+(?:will|has)\s+be(?:en)?\s+(?:closed|locked))\b", re.I),
    re.compile(r"\b(?:verify\s+your\s+identity)\b", re.I),
]

# ── Credential Harvesting Patterns ──────────────────────────
CREDENTIAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:(?:enter|confirm|verify|update)\s+(?:your\s+)?(?:password|credentials|login))\b", re.I),
    re.compile(r"\b(?:sign\s+in\s+(?:to\s+)?(?:your|the)\s+account)\b", re.I),
    re.compile(r"\b(?:reset\s+(?:your\s+)?password)\b", re.I),
    re.compile(r"\b(?:click\s+(?:here|below|the\s+link)\s+to\s+(?:verify|confirm|update|secure))\b", re.I),
    re.compile(r"\b(?:log\s*in\s+(?:to\s+)?(?:your|the))\b", re.I),
    re.compile(r"\b(?:(?:ssn|social\s+security|tax\s+id|credit\s+card)\s+(?:number)?)\b", re.I),
    re.compile(r"\b(?:(?:user\s*name|email)\s+and\s+password)\b", re.I),
    re.compile(r"\b(?:two[\s-]?factor|2fa|authentication\s+code|otp)\b", re.I),
    re.compile(r"\b(?:security\s+question|mother'?s\s+maiden\s+name)\b", re.I),
    re.compile(r"approve\s+(the\s+)?(notification|prompt|request|sign.?in)\s+(on|from)\s+(your\s+)?(phone|device|app|mobile)", re.I),
    re.compile(r"(enter|forward|share|send)\s+(the\s+)?(verification|security|authentication|one.?time)\s+(code|pin|token)", re.I),
    re.compile(r"(backup|recovery|emergency)\s+(access\s+)?codes?", re.I),
    re.compile(r"(re-?register|re-?enroll|set\s+up)\s+(your\s+)?authenticat", re.I),
    re.compile(r"(verify|confirm|update)\s+(your\s+)?(phone|mobile|cell)\s+(number|#)", re.I),
    re.compile(r"(trust|register|authorize)\s+this\s+(browser|device|computer)", re.I),
    re.compile(r"(session|token|cookie)\s+(has\s+)?(expired|invalid|revoked|timed\s+out)", re.I),
    re.compile(r"re-?authenticat|re-?verify\s+(your\s+)?(identity|account|session)", re.I),
    re.compile(r"scan\s+(this|the)\s+QR\s+code\s+to\s+(verify|login|authenticat|access)", re.I),
    re.compile(r"(mailbox|inbox|email\s+account)\s+(will\s+be|has\s+been)\s+(suspended|deactivated|deleted|closed|locked)", re.I),
]

# ── Financial Fraud / BEC Patterns ──────────────────────────
FINANCIAL_FRAUD_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:wire\s+transfer|bank\s+transfer|money\s+transfer)\b", re.I),
    re.compile(r"\b(?:gift\s+card|itunes\s+card|google\s+play\s+card|amazon\s+card)\b", re.I),
    re.compile(r"\b(?:invoice\s+(?:attached|enclosed|payment))\b", re.I),
    re.compile(r"\b(?:(?:ceo|cfo|director|president)\s+(?:request|asked|needs))\b", re.I),
    re.compile(r"\b(?:urgent\s+(?:payment|transfer|transaction))\b", re.I),
    re.compile(r"\b(?:change\s+(?:the\s+)?(?:bank|payment|account)\s+(?:details|information))\b", re.I),
    re.compile(r"\b(?:(?:new|updated)\s+(?:bank|payment|routing)\s+(?:details|account|info))\b", re.I),
    re.compile(r"\b(?:bitcoin|btc|cryptocurrency|crypto\s+wallet)\b", re.I),
    re.compile(r"\b(?:western\s+union|moneygram)\b", re.I),
    re.compile(r"\b(?:keep\s+this\s+(?:confidential|between\s+us|private|quiet))\b", re.I),
]

OAUTH_PATTERNS = [
    re.compile(r"grant\s+(access|permission)|authorize\s+(this\s+)?(app|application)", re.I),
    re.compile(r"sign\s+in\s+with\s+(Microsoft|Google|Apple|Facebook|your\s+organization)", re.I),
    re.compile(r"(review|manage)\s+(app\s+)?permissions", re.I),
    re.compile(r"(consent|approval)\s+(required|needed)|admin\s+approval", re.I),
    re.compile(r"(connect|link)\s+your\s+(account|email|calendar|mailbox)", re.I),
    re.compile(r"this\s+(app|application)\s+(needs|requires|requests)\s+access\s+to", re.I),
]


class NLPAnalysisEngine:
    """
    Engine 3 — Deep NLP analysis for phishing intent recognition.

    This engine has TWO OPERATING MODES:

    1. TRANSFORMER MODE (Primary):
       Uses a fine-tuned RoBERTa/BERT model to classify email text.
       Requires:
         - A trained model at config's `fine_tuned_path`
         - GPU recommended for production throughput
         - Dataset: ~10K-50K labeled phishing/legitimate emails

    2. HEURISTIC FALLBACK MODE:
       Uses regex pattern matching against known phishing language.
       No ML model required — purely rule-based.
       Less accurate but provides baseline detection.

    The engine always runs BOTH modes and combines signals:
    - Transformer probability provides the primary NLP score
    - Regex patterns add supplementary confidence and identify
      specific intent categories even when the model outputs
      a generic "phishing" label
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._device = None

    def _load_model(self) -> bool:
        """
        Attempt to load the fine-tuned transformer model.

        Returns True if model loaded successfully, False otherwise.
        When False, the engine falls back to heuristic-only mode.

        MODEL DETAILS:
        ─────────────────────────────────────────────────────────
        Architecture:  RoBERTa-base (125M params) or BERT-large
        Task:          Sequence Classification (num_labels=2)
        Labels:        legitimate (0), phishing (1)
        Input:         Tokenized email body text (max 512 tokens)
        Output:        Softmax probabilities over 2 classes

        TRAINING REQUIREMENTS:
        ─────────────────────────────────────────────────────────
        Dataset:       Text corpus of labeled emails
                       - Recommended: Nazario Corpus (phishing) +
                         Enron Corpus (legitimate) as baseline
                       - Augment with Kaggle phishing datasets
                       - Target: 10K-50K samples, balanced classes
        Format:        CSV/JSONL with columns: [text, label]
        Preprocessing: NO stemming/lemmatization (tokenizer handles)
        Optimizer:     AdamW, lr=2e-5, warmup_steps=500
        Epochs:        3-5 (early stopping on validation F1)
        Batch Size:    8-16 (gradient accumulation if GPU-limited)
        Metrics:       Optimize for Recall & F1 (minimize FN)
        Hardware:      GPU recommended (RTX 3090+ or A100)
                       CPU fine for inference only
        ─────────────────────────────────────────────────────────
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            import torch

            model_name = self.config.get("model", {}).get("name", "roberta-base")
            fine_tuned_path = self.config.get("model", {}).get("fine_tuned_path")
            device_config = self.config.get("model", {}).get("device", "auto")

            # Determine device
            if device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = device_config

            # Load fine-tuned model if available, otherwise base model
            model_path = fine_tuned_path or model_name

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,  # Binary: legitimate vs phishing
            ).to(self._device)
            self._model.eval()
            self._model_loaded = True

            logger.info(
                f"[Engine 3 - NLP] Loaded model from '{model_path}' "
                f"on device '{self._device}'"
            )
            return True

        except Exception as e:
            logger.warning(
                f"[Engine 3 - NLP] Model loading failed: {e}. "
                f"Falling back to heuristic mode."
            )
            self._model_loaded = False
            return False

    def analyze(self, email_text: str) -> NLPAnalysisResult:
        """
        Run NLP analysis on email body text.

        Combines transformer-based classification with regex
        pattern matching for robust intent detection.

        Args:
            email_text: The email body text (HTML-stripped, 
                        zero-width chars removed by Engine 2).

        Returns:
            NLPAnalysisResult with phishing probability, intent
            classification, and trigger phrases.
        """
        result = NLPAnalysisResult()

        if not email_text or not email_text.strip():
            return result

        # ── Transformer Classification ──────────────────────
        if not self._model_loaded:
            self._load_model()

        if self._model_loaded:
            self._run_transformer_classification(email_text, result)

        # ── Regex Pattern Matching (always runs) ────────────
        self._run_heuristic_patterns(email_text, result)

        # ── Compute Final NLP Score ─────────────────────────
        score_multiplier = self.config.get("score_multiplier", 30)
        max_score = self.config.get("max_score", 30)

        if self._model_loaded:
            # Primary: transformer probability × multiplier
            raw_score = result.phishing_probability * score_multiplier
        else:
            # Fallback: heuristic-derived probability × multiplier
            heuristic_prob = self._compute_heuristic_probability(result)
            result.phishing_probability = heuristic_prob
            raw_score = heuristic_prob * score_multiplier

        result.score = min(raw_score, max_score)

        logger.info(
            f"[Engine 3 - NLP] Score: {result.score:.1f}/{max_score} | "
            f"P(phishing)={result.phishing_probability:.3f} | "
            f"Intent={result.predicted_intent.value} | "
            f"Mode={'transformer' if self._model_loaded else 'heuristic'}"
        )

        return result

    # ────────────────────────────────────────────────────────
    # Transformer Classification
    # ────────────────────────────────────────────────────────

    def _run_transformer_classification(
        self, text: str, result: NLPAnalysisResult
    ) -> None:
        """
        Run the fine-tuned transformer model on email text.

        The model outputs softmax probabilities over 2 classes:
        - 0 = legitimate
        - 1 = phishing

        LABEL MAPPING (configured during fine-tuning):
            0 → legitimate
            1 → phishing

        The regex heuristic patterns still provide sub-intent
        categorization (urgency, credential, BEC) for explainability.
        """
        try:
            import torch

            max_length = self.config.get("model", {}).get(
                "max_sequence_length", 512
            )

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(self._device)

            # Inference (no gradient computation)
            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            probs = probabilities.cpu().numpy()

            # Binary output: P(phishing) = probs[1]
            result.phishing_probability = float(probs[1])
            result.intent_confidence = float(max(probs))

            # Determine if model thinks it's phishing
            if probs[1] > probs[0]:
                # Mark as generic phishing — regex will refine the sub-intent
                result.predicted_intent = NLPIntentCategory.GENERIC_PHISHING
            else:
                result.predicted_intent = NLPIntentCategory.LEGITIMATE

            result.raw_signals["transformer_probs"] = {
                "legitimate": float(probs[0]),
                "phishing": float(probs[1]),
            }

        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")

    # ────────────────────────────────────────────────────────
    # Heuristic Pattern Matching
    # ────────────────────────────────────────────────────────

    def _run_heuristic_patterns(
        self, text: str, result: NLPAnalysisResult
    ) -> None:
        """
        Run regex pattern matching for intent-specific language.

        This runs IN ADDITION to the transformer model to:
        1. Provide interpretable trigger phrases for analysts
        2. Detect specific intent subcategories
        3. Act as standalone detection when model is unavailable

        NO ML TRAINING NEEDED — pure regex heuristics.
        """
        trigger_phrases = []

        # ── Urgency & Coercion ──────────────────────────────
        urgency_hits = 0
        for pattern in URGENCY_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                urgency_hits += len(matches)
                trigger_phrases.extend(matches[:3])

        if urgency_hits > 0:
            # Supplement transformer score
            boost = min(urgency_hits * 0.1, 0.5)
            result.urgency_coercion_score = max(
                result.urgency_coercion_score, boost
            )

        # ── Credential Harvesting ───────────────────────────
        credential_hits = 0
        for pattern in CREDENTIAL_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                credential_hits += len(matches)
                trigger_phrases.extend(matches[:3])

        if credential_hits > 0:
            boost = min(credential_hits * 0.15, 0.6)
            result.credential_harvesting_score = max(
                result.credential_harvesting_score, boost
            )

        # ── Financial Fraud / BEC ───────────────────────────
        fraud_hits = 0
        for pattern in FINANCIAL_FRAUD_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                fraud_hits += len(matches)
                trigger_phrases.extend(matches[:3])

        if fraud_hits > 0:
            boost = min(fraud_hits * 0.15, 0.6)
            result.financial_fraud_bec_score = max(
                result.financial_fraud_bec_score, boost
            )

        # ── OAuth / Consent Phishing ──────────────────────────
        oauth_hits = 0
        for pattern in OAUTH_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                oauth_hits += len(matches)
                for m in matches[:3]:
                    match_text = m if isinstance(m, str) else m[0] if m else ""
                    if match_text and match_text not in result.trigger_phrases:
                        result.trigger_phrases.append(match_text)

        result.oauth_consent_score = max(
            result.oauth_consent_score,
            min(oauth_hits * 0.15, 0.6),
        )

        # Store trigger phrases (deduplicated)
        result.trigger_phrases = list(set(trigger_phrases))[:15]

        # Determine dominant intent if model didn't classify
        if result.predicted_intent == NLPIntentCategory.LEGITIMATE:
            max_score = max(
                urgency_hits, credential_hits, fraud_hits, oauth_hits
            )
            if max_score > 0:
                if urgency_hits == max_score:
                    result.predicted_intent = NLPIntentCategory.URGENCY_COERCION
                elif credential_hits == max_score:
                    result.predicted_intent = NLPIntentCategory.CREDENTIAL_HARVESTING
                elif fraud_hits == max_score:
                    result.predicted_intent = NLPIntentCategory.FINANCIAL_FRAUD_BEC
                elif oauth_hits == max_score:
                    result.predicted_intent = NLPIntentCategory.OAUTH_CONSENT_PHISHING

        result.raw_signals["heuristic_hits"] = {
            "urgency": urgency_hits,
            "credential": credential_hits,
            "financial_fraud": fraud_hits,
            "oauth": oauth_hits,
        }

        # ── Triad Compound Scoring ────────────────────────────
        active_categories = sum([
            result.urgency_coercion_score > 0,
            result.credential_harvesting_score > 0,
            result.financial_fraud_bec_score > 0,
            result.oauth_consent_score > 0,
        ])
        if active_categories >= 2:
            triad_boost = 0.15 * active_categories
            result.phishing_probability = min(
                1.0, result.phishing_probability + triad_boost
            )
            result.raw_signals["triad_boost_applied"] = triad_boost
            result.raw_signals["active_threat_categories"] = active_categories
            logger.warning(
                f"Triad compound scoring: {active_categories} categories active, "
                f"+{triad_boost:.2f} probability boost applied"
            )

    def _compute_heuristic_probability(
        self, result: NLPAnalysisResult
    ) -> float:
        """
        Compute a phishing probability from heuristic signals alone.

        Used as fallback when the transformer model is unavailable.
        Combines urgency, credential, and fraud sub-scores into
        a single probability estimate.
        """
        max_component = max(
            result.urgency_coercion_score,
            result.credential_harvesting_score,
            result.financial_fraud_bec_score,
            result.oauth_consent_score,
        )

        # Weighted combination: max drives, others add marginally
        combined = max_component + 0.2 * (
            result.urgency_coercion_score
            + result.credential_harvesting_score
            + result.financial_fraud_bec_score
            + result.oauth_consent_score
            - max_component
        )

        # Clamp to [0, 1]
        return min(max(combined, 0.0), 1.0)

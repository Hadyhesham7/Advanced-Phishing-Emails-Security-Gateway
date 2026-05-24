# ============================================================
# PhishingAnalyzer: Main Pipeline Orchestrator
# ============================================================
# The primary entry point for the Deep Content Inspection Pipeline.
# Orchestrates all 4 engines + ML Aggregator in sequence.
# ============================================================

from __future__ import annotations

import email
import time
from typing import Optional

import yaml
from loguru import logger

from pipeline.models import (
    PipelineVerdict,
    VerdictLabel,
)
from pipeline.engine_header import HeaderAnalysisEngine
from pipeline.engine_structure import StructuralAnalysisEngine
from pipeline.engine_nlp import NLPAnalysisEngine
from pipeline.engine_links import LinkAnalysisEngine
from pipeline.aggregator import MLAggregator


class PhishingAnalyzer:
    """
    Enterprise Phishing Detection Pipeline Orchestrator.

    Processes a raw email through 4 specialized analysis engines
    and produces a final verdict via the ML Aggregator.

    ┌─────────────────────────────────────────────────────────┐
    │                    Raw Email Input                       │
    └────────────────────────┬────────────────────────────────┘
                             │
                ┌────────────▼────────────────┐
                │      Email Parsing          │
                │  (headers, body, attachments)│
                └────────────┬────────────────┘
                             │
          ┌──────────────────┼──────────────────────┐
          │                  │                      │
    ┌─────▼─────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
    │  Engine 1  │  │    Engine 2     │  │      Engine 4       │
    │  Header    │  │   Structure    │  │      Links          │
    │ Analysis   │  │   & HTML       │  │    Analysis         │
    │ (heuristic)│  │  (heuristic    │  │   (heuristic)       │
    │            │  │   + OCR)       │  │                     │
    └─────┬─────┘  └────────┬────────┘  └──────────┬──────────┘
          │                 │                       │
          │         ┌───────▼────────┐              │
          │         │  Engine 3      │              │
          │         │  NLP & Semantic│              │
          │         │  (BERT/RoBERTa │              │
          │         │   or heuristic)│              │
          │         └───────┬────────┘              │
          │                 │                       │
    ┌─────▼─────────────────▼───────────────────────▼─────┐
    │              ML Aggregator                           │
    │  (XGBoost / Weighted Heuristic Fallback)             │
    │  24-feature vector → Final Score 0-100               │
    └─────────────────────────┬────────────────────────────┘
                              │
                ┌─────────────▼──────────────┐
                │   Binary Verdict Mapping    │
                │  0-50:   CLEAN → Deliver    │
                │  51-100: MALICIOUS → Drop   │
                └─────────────────────────────┘

    Usage:
        analyzer = PhishingAnalyzer("config/settings.yaml")
        verdict = analyzer.analyze(raw_email_string)
        print(verdict.label)      # VerdictLabel.MALICIOUS
        print(verdict.action)     # VerdictAction.DROP
        print(verdict.final_score)  # 67.5
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the PhishingAnalyzer with optional config file.

        Args:
            config_path: Path to YAML configuration file.
                         If None, uses default settings.
        """
        self.config = self._load_config(config_path)

        # Initialize engines
        self.engine_header = HeaderAnalysisEngine(
            config=self.config.get("engine_header", {})
        )
        self.engine_structure = StructuralAnalysisEngine(
            config=self.config.get("engine_structure", {})
        )
        self.engine_nlp = NLPAnalysisEngine(
            config=self.config.get("engine_nlp", {})
        )
        self.engine_links = LinkAnalysisEngine(
            config=self.config.get("engine_links", {})
        )

        # Initialize ML Aggregator
        self.aggregator = MLAggregator(
            config=self.config.get("aggregator", {})
        )

        logger.info(
            f"[PhishingAnalyzer] Initialized with "
            f"{self._count_enabled_engines()} engines enabled"
        )

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path:
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config or {}
            except FileNotFoundError:
                logger.warning(
                    f"Config file not found: {config_path}. "
                    f"Using defaults."
                )
            except yaml.YAMLError as e:
                logger.error(f"Config parsing error: {e}. Using defaults.")

        return {}

    def _count_enabled_engines(self) -> int:
        """Count how many engines are enabled in config."""
        count = 0
        for key in ["engine_header", "engine_structure", "engine_nlp", "engine_links"]:
            if self.config.get(key, {}).get("enabled", True):
                count += 1
        return count

    # ────────────────────────────────────────────────────────
    # Main Analysis Pipeline
    # ────────────────────────────────────────────────────────

    def analyze(self, raw_email: str) -> PipelineVerdict:
        """
        Execute the full phishing detection pipeline on a raw email.

        This is the primary entry point. It:
        1. Parses the raw email into components
        2. Runs all 4 analysis engines
        3. Feeds results into the ML Aggregator
        4. Returns the final verdict

        Args:
            raw_email: Complete raw email string (RFC 5322 format),
                       including headers and body.

        Returns:
            PipelineVerdict containing:
              - final_score (0-100)
              - label (CLEAN or MALICIOUS)
              - action (DELIVER or DROP)
              - Per-engine detailed results
              - Feature vector used for ML aggregation
              - Actionable flags (sandbox, IoC extraction)
        """
        pipeline_start = time.perf_counter()
        engine_timings: dict[str, float] = {}

        # ── Step 1: Parse Email ─────────────────────────────
        parsed = self._parse_email(raw_email)
        html_body = parsed.get("html_body")
        plain_text = parsed.get("plain_text")
        attachments = parsed.get("attachments", [])

        logger.info(
            f"[PhishingAnalyzer] Analyzing email from: "
            f"{parsed.get('from', 'unknown')}, "
            f"subject: {parsed.get('subject', 'N/A')}"
        )

        # ── Step 2: Engine 1 — Header Analysis ──────────────
        t0 = time.perf_counter()
        header_result = self.engine_header.analyze(raw_email)
        engine_timings["engine_header"] = (time.perf_counter() - t0) * 1000

        # ── Step 3: Engine 2 — Structural Analysis ──────────
        t0 = time.perf_counter()
        structural_result = self.engine_structure.analyze(
            html_body=html_body,
            plain_text=plain_text,
            attachments=attachments,
        )
        engine_timings["engine_structure"] = (time.perf_counter() - t0) * 1000

        # ── Step 4: Engine 3 — NLP Analysis ─────────────────
        # Use cleaned text from Engine 2 if zero-width chars were found
        nlp_input_text = structural_result.cleaned_text or plain_text or ""

        # If we extracted OCR text from an image-only email, use it
        if structural_result.ocr_extracted_text:
            nlp_input_text += " " + structural_result.ocr_extracted_text

        # Strip HTML if still present
        if nlp_input_text and "<" in nlp_input_text:
            nlp_input_text = self._strip_html(nlp_input_text)

        t0 = time.perf_counter()
        nlp_result = self.engine_nlp.analyze(nlp_input_text)
        engine_timings["engine_nlp"] = (time.perf_counter() - t0) * 1000

        # ── Step 5: Engine 4 — Link Analysis ────────────────
        t0 = time.perf_counter()
        link_result = self.engine_links.analyze(html_body, plain_text)
        engine_timings["engine_links"] = (time.perf_counter() - t0) * 1000

        # ── Step 6: ML Aggregator ───────────────────────────
        t0 = time.perf_counter()

        # Build 24-feature vector
        feature_vector = self.aggregator.build_feature_vector(
            header=header_result,
            structure=structural_result,
            nlp=nlp_result,
            links=link_result,
        )

        # Predict final score
        final_score, confidence = self.aggregator.predict(feature_vector)

        # Map to verdict
        verdict = self.aggregator.determine_verdict(
            score=final_score,
            confidence=confidence,
            header=header_result,
            structure=structural_result,
            nlp=nlp_result,
            links=link_result,
            feature_vector=feature_vector,
        )

        engine_timings["aggregator"] = (time.perf_counter() - t0) * 1000

        # ── Step 7: Finalize ────────────────────────────────
        total_time = (time.perf_counter() - pipeline_start) * 1000
        verdict.analysis_duration_ms = round(total_time, 2)
        verdict.engine_timings = {
            k: round(v, 2) for k, v in engine_timings.items()
        }

        self._log_verdict(verdict)

        return verdict

    # ────────────────────────────────────────────────────────
    # Email Parsing
    # ────────────────────────────────────────────────────────

    def _parse_email(self, raw_email: str) -> dict:
        """
        Parse raw email into structured components.

        Extracts:
        - Headers (From, To, Subject, Reply-To, X-Mailer, etc.)
        - HTML body (if multipart)
        - Plain text body
        - Attachments (with content type and raw bytes)
        """
        msg = email.message_from_string(raw_email)

        parsed = {
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "subject": msg.get("Subject", ""),
            "reply_to": msg.get("Reply-To", ""),
            "x_mailer": msg.get("X-Mailer", ""),
            "date": msg.get("Date", ""),
            "html_body": None,
            "plain_text": None,
            "attachments": [],
        }

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                if content_type == "text/html" and "attachment" not in disposition:
                    try:
                        parsed["html_body"] = part.get_payload(decode=True).decode(
                            part.get_content_charset() or "utf-8",
                            errors="replace",
                        )
                    except Exception:
                        parsed["html_body"] = part.get_payload()

                elif content_type == "text/plain" and "attachment" not in disposition:
                    try:
                        parsed["plain_text"] = part.get_payload(decode=True).decode(
                            part.get_content_charset() or "utf-8",
                            errors="replace",
                        )
                    except Exception:
                        parsed["plain_text"] = part.get_payload()

                elif "attachment" in disposition or content_type.startswith("image/"):
                    try:
                        parsed["attachments"].append({
                            "filename": part.get_filename() or "unknown",
                            "content_type": content_type,
                            "data": part.get_payload(decode=True) or b"",
                        })
                    except Exception:
                        pass
        else:
            content_type = msg.get_content_type()
            try:
                payload = msg.get_payload(decode=True).decode(
                    msg.get_content_charset() or "utf-8",
                    errors="replace",
                )
            except Exception:
                payload = msg.get_payload()

            if content_type == "text/html":
                parsed["html_body"] = payload
            else:
                parsed["plain_text"] = payload

        return parsed

    @staticmethod
    def _strip_html(text: str) -> str:
        """Strip HTML tags from text content."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(text, "html.parser").get_text(
                separator=" ", strip=True
            )
        except ImportError:
            import re
            return re.sub(r"<[^>]+>", " ", text).strip()

    # ────────────────────────────────────────────────────────
    # Logging & Reporting
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _log_verdict(verdict: PipelineVerdict) -> None:
        """Log the final verdict with summary."""
        severity_emoji = {
            VerdictLabel.CLEAN: "✅",
            VerdictLabel.MALICIOUS: "🔴",
        }

        emoji = severity_emoji.get(verdict.label, "❓")

        logger.info(
            f"\n{'═' * 60}\n"
            f"{emoji} VERDICT: {verdict.label.value}\n"
            f"   Score:      {verdict.final_score}/100\n"
            f"   Action:     {verdict.action.value}\n"
            f"   Confidence: {verdict.confidence:.1%}\n"
            f"   Aggregator: {verdict.aggregator_used}\n"
            f"   ────────────────────────────────────\n"
            f"   Engine Scores:\n"
            f"     Header:    {verdict.header_result.score}/25\n"
            f"     Structure: {verdict.structural_result.score}/20\n"
            f"     NLP:       {verdict.nlp_result.score:.1f}/30\n"
            f"     Links:     {verdict.link_result.score}/25\n"
            f"   ────────────────────────────────────\n"
            f"   Flags: {verdict.feature_vector.total_flags_triggered}/12\n"
            f"   Duration: {verdict.analysis_duration_ms:.1f}ms\n"
            f"{'═' * 60}"
        )

    # ────────────────────────────────────────────────────────
    # Verdict Report Generation
    # ────────────────────────────────────────────────────────

    @staticmethod
    def generate_report(verdict: PipelineVerdict) -> str:
        """
        Generate a human-readable analysis report from a verdict.

        Intended for SOC analyst review and incident documentation.
        """
        lines = [
            "=" * 70,
            "  PHISHING ANALYSIS REPORT",
            "=" * 70,
            "",
            f"  Final Score:    {verdict.final_score}/100",
            f"  Verdict:        {verdict.label.value}",
            f"  Action:         {verdict.action.value}",
            f"  Confidence:     {verdict.confidence:.1%}",
            f"  Aggregator:     {verdict.aggregator_used}",
            f"  Analysis Time:  {verdict.analysis_duration_ms:.1f}ms",
            "",
            "-" * 70,
            "  ENGINE 1 — HEADER ANALYSIS",
            "-" * 70,
            f"  Score: {verdict.header_result.score}/25",
        ]

        h = verdict.header_result
        if h.display_name_spoofing_detected:
            lines.append(
                f"  ⚠ Display Name Spoofing: "
                f"'{h.display_name_claimed}' from {h.display_name_actual_domain}"
            )
        if h.reply_to_mismatch_detected:
            lines.append(f"  ⚠ Reply-To Mismatch: {h.reply_to_address}")
        if h.xmailer_anomaly_detected:
            lines.append(f"  ⚠ Suspicious X-Mailer: {h.xmailer_value}")
        if h.new_domain_flag:
            lines.append(f"  ⚠ New Domain: {h.domain_age_days} days old")
        if h.typosquatting_detected:
            lines.append(
                f"  ⚠ Typosquatting: mimics '{h.typosquatting_target_brand}' "
                f"(similarity: {h.typosquatting_similarity:.0%})"
            )

        lines.extend([
            "",
            "-" * 70,
            "  ENGINE 2 — STRUCTURAL ANALYSIS",
            "-" * 70,
            f"  Score: {verdict.structural_result.score}/20",
        ])

        s = verdict.structural_result
        if s.hidden_text_detected:
            lines.append(
                f"  ⚠ Hidden Text: methods={s.hidden_text_methods}"
            )
        if s.zero_width_chars_detected:
            lines.append(
                f"  ⚠ Zero-Width Characters: {s.zero_width_char_count} found"
            )
        if s.image_only_email:
            lines.append("  ⚠ Image-Only Email detected")
            if s.ocr_extracted_text:
                lines.append(
                    f"    OCR Text: {s.ocr_extracted_text[:100]}..."
                )
        if s.brand_impersonation_detected:
            lines.append(
                f"  ⚠ Brand Impersonation: {s.impersonated_brand}"
            )

        lines.extend([
            "",
            "-" * 70,
            "  ENGINE 3 — NLP ANALYSIS",
            "-" * 70,
            f"  Score: {verdict.nlp_result.score:.1f}/30",
            f"  Phishing Probability: {verdict.nlp_result.phishing_probability:.1%}",
            f"  Predicted Intent: {verdict.nlp_result.predicted_intent.value}",
        ])

        n = verdict.nlp_result
        if n.trigger_phrases:
            lines.append(f"  Trigger Phrases: {', '.join(n.trigger_phrases[:5])}")

        lines.extend([
            "",
            "-" * 70,
            "  ENGINE 4 — LINK ANALYSIS",
            "-" * 70,
            f"  Score: {verdict.link_result.score}/25",
            f"  Total Links: {verdict.link_result.total_links_found}",
        ])

        l = verdict.link_result
        if l.href_mismatch_detected:
            for mismatch in l.mismatched_links[:3]:
                display = mismatch.get('displayed_domain', mismatch.get('brand_mentioned', 'unknown'))
                actual = mismatch.get('actual_domain', mismatch.get('actual_href', 'unknown'))
                lines.append(
                    f"  ⚠ Href Mismatch: displays '{display}' "
                    f"→ links to '{actual}'"
                )
        if l.url_obfuscation_detected:
            lines.append(
                f"  ⚠ URL Obfuscation: {len(l.obfuscated_urls)} obfuscated URLs"
            )
        if l.homograph_attack_detected:
            for hg in l.homograph_urls[:3]:
                lines.append(
                    f"  ⚠ Homograph Attack: '{hg.get('original', '')}' "
                    f"→ looks like '{hg.get('latin_equivalent', '')}'"
                )

        lines.extend([
            "",
            "-" * 70,
            "  FEATURE VECTOR SUMMARY",
            "-" * 70,
            f"  Total Flags Triggered: "
            f"{verdict.feature_vector.total_flags_triggered}/12",
            f"  Cross-Engine Risk (Header×Link): "
            f"{verdict.feature_vector.header_x_link_risk:.1f}",
            f"  Cross-Engine Risk (NLP×Structure): "
            f"{verdict.feature_vector.nlp_x_structure_risk:.1f}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

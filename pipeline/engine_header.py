# ============================================================
# Engine 1: Advanced Header Analysis (Identity & Routing)
# ============================================================
# Detection Targets:
#   - Display Name Spoofing
#   - Reply-To Mismatch
#   - X-Mailer / User-Agent Anomalies
#   - Domain Age (WHOIS) & Typosquatting
#
# ML Requirement: NONE — Pure heuristic engine
# ============================================================

from __future__ import annotations

import re
import email
import email.utils
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from pipeline.models import HeaderAnalysisResult


# ── Known Brand Names for Spoofing Detection ────────────────
# In production, load from config/data/brand_names.txt
DEFAULT_BRAND_NAMES: set[str] = {
    "paypal", "microsoft", "apple", "google", "amazon", "netflix",
    "facebook", "meta", "instagram", "linkedin", "twitter", "chase",
    "wells fargo", "bank of america", "citibank", "hsbc", "dropbox",
    "docusign", "adobe", "zoom", "slack", "salesforce", "stripe",
    "coinbase", "binance", "dhl", "fedex", "ups", "usps",
}

# ── Freemail Domains ────────────────────────────────────────
FREEMAIL_DOMAINS: set[str] = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "proton.me", "mail.com", "yandex.com", "gmx.com",
    "zoho.com", "icloud.com", "tutanota.com", "fastmail.com",
}

# ── Suspicious X-Mailer Patterns ────────────────────────────
SUSPICIOUS_MAILER_PATTERNS: list[re.Pattern] = [
    re.compile(r"PHPMailer", re.IGNORECASE),
    re.compile(r"Kali", re.IGNORECASE),
    re.compile(r"swaks", re.IGNORECASE),
    re.compile(r"sendemail", re.IGNORECASE),
    re.compile(r"Python", re.IGNORECASE),
    re.compile(r"curl", re.IGNORECASE),
    re.compile(r"perl", re.IGNORECASE),
    re.compile(r"ruby", re.IGNORECASE),
    re.compile(r"Go-http-client", re.IGNORECASE),
    re.compile(r"PowerShell", re.IGNORECASE),
]

# ── Homoglyph Map for Typosquatting ─────────────────────────
# Maps visually similar characters used in domain impersonation
HOMOGLYPH_MAP: dict[str, str] = {
    "rn": "m",   # rnicrosoft → microsoft
    "cl": "d",   # clocusign → docusign
    "vv": "w",   # amavvs → amaws (contrived)
    "1": "l",    # pay1pal → paypal
    "0": "o",    # g00gle → google
    "!": "i",    # l!nkedin → linkedin
}


class HeaderAnalysisEngine:
    """
    Engine 1 — Analyzes email headers for identity spoofing and
    routing anomalies using deterministic heuristic rules.

    NO ML MODELS REQUIRED. All logic is rule-based.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.brand_names = DEFAULT_BRAND_NAMES
        self.freemail_domains = FREEMAIL_DOMAINS
        self._load_custom_brands()

    def _load_custom_brands(self) -> None:
        """Load additional brand names from config file if specified."""
        brand_file = self.config.get("display_name_spoofing", {}).get(
            "brand_names_file"
        )
        if brand_file:
            try:
                with open(brand_file, "r") as f:
                    custom_brands = {
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    }
                    self.brand_names |= custom_brands
                    logger.info(
                        f"Loaded {len(custom_brands)} custom brand names"
                    )
            except FileNotFoundError:
                logger.warning(f"Brand names file not found: {brand_file}")

    def analyze(self, raw_email: str) -> HeaderAnalysisResult:
        """
        Run all header analysis checks and return aggregated result.

        Args:
            raw_email: The complete raw email string (headers + body).

        Returns:
            HeaderAnalysisResult with all flags and weighted score.
        """
        result = HeaderAnalysisResult()
        msg = email.message_from_string(raw_email)

        # ── Check 1: Display Name Spoofing ──────────────────
        self._check_display_name_spoofing(msg, result)

        # ── Check 2: Reply-To Mismatch ──────────────────────
        self._check_reply_to_mismatch(msg, result)

        # ── Check 3: X-Mailer / User-Agent Anomalies ────────
        self._check_xmailer_anomaly(msg, result)

        # ── Check 4: Domain Age (Newly Registered) ──────────
        self._check_domain_age(msg, result)

        # ── Check 5: Typosquatting ──────────────────────────
        self._check_typosquatting(msg, result)

        # ── Check 6: Authentication Header Analysis ─────────
        self._check_authentication_results(msg, result)

        # ── Compute Aggregate Score (capped at max_score) ───
        max_score = self.config.get("max_score", 25)
        raw_total = (
            result.display_name_spoofing_score
            + result.reply_to_mismatch_score
            + result.xmailer_anomaly_score
            + result.new_domain_score
            + result.typosquatting_score
            + result.auth_failure_score
        )

        # ── Compound Scoring: Display Name Spoofing + Freemail Reply-To ──
        if result.display_name_spoofing_detected and result.reply_to_mismatch_detected:
            freemail_reply = result.raw_signals.get("reply_to_is_freemail", False)
            if freemail_reply:
                compound_bonus = 5
                result.raw_signals["compound_spoof_freemail"] = True
                logger.warning(
                    f"Compound threat: Display name spoofing + freemail Reply-To — +{compound_bonus} penalty"
                )
                raw_total += compound_bonus

        result.score = min(raw_total, max_score)

        logger.info(
            f"[Engine 1 - Header] Score: {result.score}/{max_score} | "
            f"Spoofing={result.display_name_spoofing_detected}, "
            f"ReplyTo={result.reply_to_mismatch_detected}, "
            f"XMailer={result.xmailer_anomaly_detected}, "
            f"NewDomain={result.new_domain_flag}, "
            f"Typosquat={result.typosquatting_detected}, "
            f"AuthFailure={result.auth_failure_detected}"
        )

        return result

    # ────────────────────────────────────────────────────────
    # Detection Methods
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _robust_parseaddr(header_val: str) -> tuple[str, str]:
        """
        Fallback parser for malformed headers designed to break standard RFC parsing.
        e.g., 'Microsoft team ,_<no-reply@bad.com>' breaks email.utils.parseaddr.
        """
        display_name, addr = email.utils.parseaddr(header_val)
        if not display_name and not addr and "<" in header_val and ">" in header_val:
            import re
            match = re.search(r'(.*?)\s*<([^>]+)>', header_val)
            if match:
                display_name = match.group(1).strip(' ,_"\'\t\n\r')
                addr = match.group(2).strip()
        return display_name, addr

    def _check_display_name_spoofing(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Compare Display Name against From address domain.
        
        Flags when a known brand name appears in the display name
        but the actual sender domain doesn't belong to that brand.
        
        Example:
            "PayPal Support" <hacker@randomdomain.com>  → FLAGGED
            "PayPal Support" <noreply@paypal.com>        → OK
        """
        from_header = msg.get("From", "")
        display_name, from_addr = self._robust_parseaddr(from_header)

        if not display_name or not from_addr:
            return

        display_lower = display_name.lower()
        display_normalized = display_lower.replace(" ", "")
        from_domain = from_addr.split("@")[-1].lower() if "@" in from_addr else ""

        result.display_name_claimed = display_name
        result.display_name_actual_domain = from_domain

        for brand in self.brand_names:
            brand_clean = brand.replace(" ", "")
            if brand_clean in display_normalized:
                # Check if the from domain legitimately contains the brand
                if brand_clean not in from_domain:
                    result.display_name_spoofing_detected = True
                    weight = self.config.get(
                        "display_name_spoofing", {}
                    ).get("weight", 15)
                    result.display_name_spoofing_score = weight
                    result.raw_signals["spoofed_brand"] = brand
                    result.raw_signals["claimed_display"] = display_name
                    result.raw_signals["actual_domain"] = from_domain
                    logger.warning(
                        f"Display name spoofing: '{display_name}' "
                        f"from domain '{from_domain}'"
                    )
                    break

    def _check_reply_to_mismatch(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Detect Reply-To pointing to a different domain than From.
        
        Especially flags when Reply-To uses a freemail provider
        while From uses a corporate-looking domain, suggesting
        the attacker wants responses routed to their personal inbox.
        """
        from_header = msg.get("From", "")
        reply_to_header = msg.get("Reply-To", "")

        if not reply_to_header:
            return

        _, from_addr = self._robust_parseaddr(from_header)
        _, reply_to_addr = self._robust_parseaddr(reply_to_header)

        if not from_addr or not reply_to_addr:
            return

        from_domain = from_addr.split("@")[-1].lower() if "@" in from_addr else ""
        reply_domain = (
            reply_to_addr.split("@")[-1].lower() if "@" in reply_to_addr else ""
        )

        result.reply_to_address = reply_to_addr

        if from_domain != reply_domain:
            # Extra penalty if reply-to is a freemail domain
            is_freemail = reply_domain in self.freemail_domains
            result.reply_to_mismatch_detected = True
            weight = self.config.get("reply_to_mismatch", {}).get("weight", 10)
            result.reply_to_mismatch_score = weight
            result.raw_signals["reply_to_domain"] = reply_domain
            result.raw_signals["reply_to_is_freemail"] = is_freemail

            if reply_domain in self.freemail_domains:
                result.raw_signals["reply_to_is_freemail"] = True
                result.reply_to_mismatch_score += 5
                logger.warning(
                    f"Reply-To uses freemail domain: {reply_domain} — additional penalty applied"
                )

    def _check_xmailer_anomaly(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Flag emails with suspicious X-Mailer / User-Agent headers.
        
        Legitimate enterprise emails come from Outlook, Thunderbird,
        or corporate MTAs. Emails from scripting tools (PHPMailer,
        Python, curl) are strong phishing indicators.
        """
        xmailer = msg.get("X-Mailer", "") or msg.get("User-Agent", "")

        if not xmailer:
            return

        result.xmailer_value = xmailer

        for pattern in SUSPICIOUS_MAILER_PATTERNS:
            if pattern.search(xmailer):
                result.xmailer_anomaly_detected = True
                weight = self.config.get("xmailer_anomaly", {}).get("weight", 10)
                result.xmailer_anomaly_score = weight
                result.raw_signals["xmailer_matched_pattern"] = pattern.pattern
                logger.warning(f"Suspicious X-Mailer detected: '{xmailer}'")
                break

    def _check_domain_age(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Flag newly registered sender domains (< 30 days old).
        
        Uses python-whois for domain creation date lookup.
        In production, this should be backed by a WHOIS cache
        to avoid rate limiting and latency.
        
        NOTE: This is a heuristic check — no ML training needed.
        The WHOIS lookup is a real-time API call, not a model.
        """
        from_header = msg.get("From", "")
        _, from_addr = email.utils.parseaddr(from_header)

        if not from_addr or "@" not in from_addr:
            return

        domain = from_addr.split("@")[-1].lower()
        threshold_days = self.config.get("domain_analysis", {}).get(
            "new_domain_threshold_days", 30
        )

        try:
            domain_age = self._get_domain_age_days(domain)
            result.domain_age_days = domain_age

            if domain_age is not None and domain_age < threshold_days:
                result.new_domain_flag = True
                weight = self.config.get("domain_analysis", {}).get(
                    "weight_new_domain", 15
                )
                result.new_domain_score = weight
                result.raw_signals["domain_age_days"] = domain_age
                logger.warning(
                    f"Newly registered domain: {domain} "
                    f"({domain_age} days old)"
                )
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {domain}: {e}")

    def _get_domain_age_days(self, domain: str) -> Optional[int]:
        """
        Query WHOIS for domain creation date.
        
        Returns the age in days, or None if unavailable.
        In production, wrap this with a Redis/local cache.
        """
        try:
            import whois

            w = whois.whois(domain)
            creation_date = w.creation_date

            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            if creation_date:
                age = (datetime.now(timezone.utc) - creation_date.replace(
                    tzinfo=timezone.utc
                )).days
                return max(age, 0)
        except Exception:
            pass

        return None

    def _check_typosquatting(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Detect typosquatted domains mimicking known brands.
        
        Uses two approaches:
        1. Levenshtein edit distance (fuzzy string matching)
        2. Homoglyph substitution (rn→m, 0→o, 1→l, etc.)
        
        Example: rnicrosoft.com → microsoft.com (Levenshtein=1 after
                 homoglyph normalization)
        """
        from_header = msg.get("From", "")
        _, from_addr = email.utils.parseaddr(from_header)

        if not from_addr or "@" not in from_addr:
            return

        domain = from_addr.split("@")[-1].lower()
        domain_name = domain.split(".")[0]  # Extract base domain

        similarity_threshold = self.config.get("domain_analysis", {}).get(
            "typosquatting_similarity", 0.80
        )

        best_match = None
        best_similarity = 0.0

        for brand in self.brand_names:
            brand_clean = brand.replace(" ", "")

            # Skip exact matches (legitimate)
            if domain_name == brand_clean:
                continue

            # Approach 1: Direct Levenshtein similarity
            similarity = self._levenshtein_ratio(domain_name, brand_clean)

            # Approach 2: Normalize homoglyphs, then compare
            normalized = self._normalize_homoglyphs(domain_name)
            if normalized != domain_name:
                homo_similarity = self._levenshtein_ratio(normalized, brand_clean)
                similarity = max(similarity, homo_similarity)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = brand

        if best_match and best_similarity >= similarity_threshold:
            result.typosquatting_detected = True
            result.typosquatting_target_brand = best_match
            result.typosquatting_similarity = best_similarity
            weight = self.config.get("domain_analysis", {}).get(
                "weight_typosquatting", 20
            )
            result.typosquatting_score = weight
            result.raw_signals["typosquat_domain"] = domain
            result.raw_signals["typosquat_brand"] = best_match
            result.raw_signals["typosquat_similarity"] = best_similarity
            logger.warning(
                f"Typosquatting detected: '{domain}' mimics "
                f"'{best_match}' (similarity={best_similarity:.2f})"
            )

    def _check_authentication_results(
        self, msg: email.message.Message, result: HeaderAnalysisResult
    ) -> None:
        """
        Analyze Authentication-Results header for SPF/DKIM/DMARC failures.

        Flags when 2+ authentication mechanisms report failure,
        or when the header is entirely missing from an external sender.
        """
        auth_header = msg.get("Authentication-Results", "")
        weight = self.config.get("authentication", {}).get("weight", 10)

        if auth_header:
            auth_lower = auth_header.lower()
            failures = 0
            mechanisms_failed = []

            for mechanism in ("spf", "dkim", "dmarc"):
                if f"{mechanism}=fail" in auth_lower:
                    failures += 1
                    mechanisms_failed.append(mechanism)

            if failures >= 2:
                result.auth_failure_detected = True
                result.auth_failure_score = weight
                result.raw_signals["auth_failures"] = mechanisms_failed
                result.raw_signals["auth_failure_count"] = failures
                logger.warning(
                    f"Authentication failures detected: {', '.join(mechanisms_failed)}"
                )
        else:
            # Missing Authentication-Results — suspicious for external mail
            from_header = msg.get("From", "")
            _, from_addr = self._robust_parseaddr(from_header)
            if from_addr and "@" in from_addr:
                from_domain = from_addr.split("@")[-1].lower()
                internal_domains = self.config.get(
                    "authentication", {}
                ).get("internal_domains", [])
                if from_domain not in internal_domains:
                    result.auth_failure_score = 5
                    result.raw_signals["auth_header_missing"] = True
                    logger.info(
                        f"Authentication-Results header missing for external domain: {from_domain}"
                    )

    # ────────────────────────────────────────────────────────
    # Utility Methods
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Compute Levenshtein similarity ratio between two strings.
        Returns value between 0.0 (completely different) and 1.0 (identical).
        
        Uses dynamic programming if python-Levenshtein is unavailable.
        """
        try:
            from Levenshtein import ratio
            return ratio(s1, s2)
        except ImportError:
            # Fallback: manual DP implementation
            return HeaderAnalysisEngine._manual_levenshtein_ratio(s1, s2)

    @staticmethod
    def _manual_levenshtein_ratio(s1: str, s2: str) -> float:
        """Pure Python Levenshtein distance fallback."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost,
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    @staticmethod
    def _normalize_homoglyphs(text: str) -> str:
        """
        Replace known homoglyph sequences with their intended characters.
        
        This handles visual substitutions attackers use:
            rn → m  (rnicrosoft → microsoft)
            0  → o  (g00gle → google)
            1  → l  (pay1pal → paypal)
        """
        normalized = text
        for fake, real in HOMOGLYPH_MAP.items():
            normalized = normalized.replace(fake, real)
        return normalized

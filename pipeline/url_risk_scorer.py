# ============================================================
# URL Risk Scorer — ML-based URL phishing probability
# ============================================================
# Uses an XGBoost model trained on the LegitPhish dataset
# (101K URLs with 17 features) to score URL phishing risk.
# ============================================================

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs
from typing import Optional

from loguru import logger


# Common TLDs considered "popular" (legitimate)
POPULAR_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "co", "io", "us", "uk", "ca", "au", "de", "fr",
    "jp", "cn", "ru", "br", "in", "it", "nl", "se",
    "no", "es", "pt", "pl", "ch", "at", "be", "dk",
    "fi", "ie", "nz", "za", "mx", "ar", "cl", "co.uk",
}

# Suspicious file extensions in URLs
SUSPICIOUS_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".scr", ".pif", ".com",
    ".vbs", ".js", ".wsh", ".wsf", ".msi", ".jar",
    ".php", ".asp", ".aspx", ".cgi", ".pl",
}


class URLRiskScorer:
    """
    Score URLs for phishing risk using an XGBoost model
    trained on the LegitPhish dataset.
    """

    def __init__(self, model_path: str = "models/url_risk_model.json"):
        self._model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        """Load the XGBoost URL risk model."""
        try:
            import xgboost as xgb

            self._model = xgb.XGBClassifier()
            self._model.load_model(self._model_path)
            logger.debug(f"[URLRiskScorer] Loaded model from {self._model_path}")
        except Exception as e:
            logger.debug(f"[URLRiskScorer] Model not available: {e}")
            self._model = None

    def is_available(self) -> bool:
        """Check if the ML model is loaded."""
        return self._model is not None

    def score_url(self, url: str) -> float:
        """
        Score a single URL for phishing risk.

        Args:
            url: The URL string to analyze.

        Returns:
            Float between 0.0 (safe) and 1.0 (phishing).
        """
        features = self.extract_features(url)

        if self._model is not None:
            try:
                import numpy as np

                feature_values = [
                    features["url_length"],
                    features["has_ip_address"],
                    features["dot_count"],
                    features["https_flag"],
                    features["url_entropy"],
                    features["token_count"],
                    features["subdomain_count"],
                    features["query_param_count"],
                    features["tld_length"],
                    features["path_length"],
                    features["has_hyphen_in_domain"],
                    features["number_of_digits"],
                    features["tld_popularity"],
                    features["suspicious_file_extension"],
                    features["domain_name_length"],
                    features["percentage_numeric_chars"],
                ]
                X = np.array([feature_values])
                proba = self._model.predict_proba(X)[0]
                # Return probability of phishing (class 0 in LegitPhish)
                return float(proba[0]) if len(proba) > 1 else float(proba[0])
            except Exception as e:
                logger.debug(f"[URLRiskScorer] Prediction error: {e}")

        # Fallback heuristic scoring
        return self._heuristic_score(features)

    @staticmethod
    def extract_features(url: str) -> dict:
        """
        Extract the 17 URL features matching the LegitPhish dataset.
        """
        try:
            parsed = urlparse(url if "://" in url else f"http://{url}")
        except Exception:
            return URLRiskScorer._empty_features()

        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        scheme = parsed.scheme or ""

        # 1. url_length
        url_length = len(url)

        # 2. has_ip_address
        ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
        has_ip_address = 1 if ip_pattern.match(hostname) else 0

        # 3. dot_count
        dot_count = url.count(".")

        # 4. https_flag
        https_flag = 1 if scheme.lower() == "https" else 0

        # 5. url_entropy
        url_entropy = URLRiskScorer._calculate_entropy(url)

        # 6. token_count (split by special chars)
        tokens = re.split(r"[/\-_.?&=:@#]", url)
        token_count = len([t for t in tokens if t])

        # 7. subdomain_count
        parts = hostname.split(".")
        subdomain_count = max(0, len(parts) - 2)

        # 8. query_param_count
        query_params = parse_qs(query)
        query_param_count = len(query_params) + (1 if query else 0)

        # 9. tld_length
        tld = parts[-1] if parts else ""
        tld_length = len(tld)

        # 10. path_length
        path_length = len(path)

        # 11. has_hyphen_in_domain
        domain_part = ".".join(parts[:-1]) if len(parts) > 1 else hostname
        has_hyphen_in_domain = 1 if "-" in domain_part else 0

        # 12. number_of_digits
        number_of_digits = sum(1 for c in url if c.isdigit())

        # 13. tld_popularity
        tld_clean = tld.lower().strip(".")
        tld_popularity = 1 if tld_clean in POPULAR_TLDS else 0

        # 14. suspicious_file_extension
        suspicious_file_extension = 0
        for ext in SUSPICIOUS_EXTENSIONS:
            if path.lower().endswith(ext):
                suspicious_file_extension = 1
                break

        # 15. domain_name_length
        domain_name_length = len(hostname)

        # 16. percentage_numeric_chars
        total_chars = len(url) if len(url) > 0 else 1
        percentage_numeric_chars = (number_of_digits / total_chars) * 100

        return {
            "url_length": url_length,
            "has_ip_address": has_ip_address,
            "dot_count": dot_count,
            "https_flag": https_flag,
            "url_entropy": url_entropy,
            "token_count": token_count,
            "subdomain_count": subdomain_count,
            "query_param_count": query_param_count,
            "tld_length": tld_length,
            "path_length": path_length,
            "has_hyphen_in_domain": has_hyphen_in_domain,
            "number_of_digits": number_of_digits,
            "tld_popularity": tld_popularity,
            "suspicious_file_extension": suspicious_file_extension,
            "domain_name_length": domain_name_length,
            "percentage_numeric_chars": percentage_numeric_chars,
        }

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        counter = Counter(text)
        length = len(text)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _empty_features() -> dict:
        """Return empty feature dict."""
        return {k: 0 for k in [
            "url_length", "has_ip_address", "dot_count", "https_flag",
            "url_entropy", "token_count", "subdomain_count",
            "query_param_count", "tld_length", "path_length",
            "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
            "suspicious_file_extension", "domain_name_length",
            "percentage_numeric_chars",
        ]}

    @staticmethod
    def _heuristic_score(features: dict) -> float:
        """Fallback heuristic scoring when model isn't available."""
        score = 0.0

        if features["has_ip_address"]:
            score += 0.3
        if not features["https_flag"]:
            score += 0.1
        if features["url_length"] > 75:
            score += 0.15
        if features["subdomain_count"] > 2:
            score += 0.1
        if features["suspicious_file_extension"]:
            score += 0.2
        if not features["tld_popularity"]:
            score += 0.1
        if features["has_hyphen_in_domain"]:
            score += 0.05
        if features["url_entropy"] > 4.5:
            score += 0.1
        if features["percentage_numeric_chars"] > 20:
            score += 0.1

        return min(score, 1.0)

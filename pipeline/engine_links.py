# ============================================================
# Engine 4: Static Link & Payload Analysis (Call-to-Action)
# ============================================================
# Detection Targets:
#   - Href vs. Display Text Mismatch
#   - URL Obfuscation (shorteners, deep paths)
#   - Punycode / Homograph Attacks
#
# ML Requirement: NONE — Pure heuristic engine
#   All detection is rule-based: URL parsing, string comparison,
#   Unicode codepoint analysis, and domain reputation lookups.
# ============================================================

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, unquote
from typing import Optional

from loguru import logger

from pipeline.models import LinkAnalysisResult


# ── Known URL Shortener Domains ─────────────────────────────
URL_SHORTENER_DOMAINS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "tiny.cc", "lnkd.in", "db.tt",
    "qr.ae", "tr.im", "bl.ink", "rb.gy", "shorturl.at",
    "cutt.ly", "v.gd", "s.id", "rebrand.ly", "t.ly",
    "short.io", "clck.ru", "clk.sh", "su.pr",
}

# ── Suspicious TLDs ─────────────────────────────────────────
SUSPICIOUS_TLDS: set[str] = {
    ".xyz", ".top", ".club", ".work", ".click", ".info",
    ".buzz", ".icu", ".rest", ".surf", ".monster", ".site",
    ".online", ".space", ".fun", ".gq", ".ml", ".cf", ".ga", ".tk",
}

# ── Cyrillic → Latin Homoglyph Mapping ──────────────────────
# Characters that look identical to Latin letters but are Cyrillic
CYRILLIC_HOMOGLYPHS: dict[str, str] = {
    "\u0430": "a",   # а → a
    "\u0435": "e",   # е → e
    "\u043E": "o",   # о → o
    "\u0440": "p",   # р → p
    "\u0441": "c",   # с → c
    "\u0443": "y",   # у → y
    "\u0445": "x",   # х → x
    "\u042C": "b",   # Ь → soft sign (looks like b)
    "\u0456": "i",   # і → i
    "\u0458": "j",   # ј → j
    "\u043A": "k",   # к → k (depending on font)
    "\u043D": "h",   # н → h (depending on font)
}

# Greek homoglyphs
GREEK_HOMOGLYPHS: dict[str, str] = {
    "\u03BF": "o",   # ο → o
    "\u03B1": "a",   # α → a
    "\u03C1": "p",   # ρ → p
    "\u03B5": "e",   # ε → e
    "\u03C4": "t",   # τ → t
}

BRAND_DOMAINS: dict[str, list[str]] = {
    "paypal": ["paypal.com"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com", "office365.com"],
    "google": ["google.com", "gmail.com", "googleapis.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "aws.amazon.com"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "wells fargo": ["wellsfargo.com"],
    "bank of america": ["bankofamerica.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "linkedin": ["linkedin.com"],
    "dropbox": ["dropbox.com"],
    "docusign": ["docusign.com", "docusign.net"],
    "onedrive": ["onedrive.com", "sharepoint.com", "microsoft.com"],
    "sharepoint": ["sharepoint.com", "microsoft.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "twitter": ["twitter.com", "x.com"],
}

BARE_DOMAIN_RE = re.compile(
    r'\b([a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:com|org|net|edu|gov|io|co|co\.uk|app|dev))\b'
)


class LinkAnalysisEngine:
    """
    Engine 4 — Pre-sandbox analysis of embedded URLs and links.

    Detects visual deception in hyperlinks without executing them.
    NO ML MODELS REQUIRED — all analysis is string/Unicode-based.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._shortener_domains = URL_SHORTENER_DOMAINS
        self._load_custom_shorteners()

    def _load_custom_shorteners(self) -> None:
        """Load additional URL shortener domains from config."""
        shortener_file = self.config.get("url_obfuscation", {}).get(
            "shortener_domains_file"
        )
        if shortener_file:
            try:
                with open(shortener_file, "r") as f:
                    custom = {
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    }
                    self._shortener_domains |= custom
            except FileNotFoundError:
                logger.warning(f"Shortener domains file not found: {shortener_file}")

    def analyze(
        self,
        html_body: Optional[str] = None,
        plain_text: Optional[str] = None,
    ) -> LinkAnalysisResult:
        """
        Analyze all links found in the email body (HTML and/or plain text).

        Args:
            html_body: HTML content of the email.
            plain_text: Plain text content of the email.

        Returns:
            LinkAnalysisResult with all flags and weighted score.
        """
        result = LinkAnalysisResult()

        if not html_body and not plain_text:
            return result

        # ── Extract links from HTML <a> tags ────────────────
        links = []
        if html_body:
            links = self._extract_links(html_body)

        # ── Also extract plain-text URLs from both sources ──
        # This catches URLs that are NOT inside <a> tags
        plain_urls = set()
        if plain_text:
            plain_urls.update(self._extract_plain_text_urls(plain_text))
        if html_body:
            # Extract URLs from HTML text content (not just <a> tags)
            plain_urls.update(self._extract_plain_text_urls(html_body))

        # Add plain-text URLs that weren't already found in <a> tags
        existing_hrefs = {link.get("href", "").lower() for link in links}
        for url in plain_urls:
            if url.lower() not in existing_hrefs:
                links.append({"href": url, "display_text": url})

        result.total_links_found = len(links)

        if not links:
            return result

        # ── URL Risk Scoring (ML-based if model available) ──
        url_risk_score = 0.0
        try:
            from pipeline.url_risk_scorer import URLRiskScorer
            scorer = URLRiskScorer()
            if scorer.is_available():
                scores = []
                for link in links:
                    href = link.get("href", "")
                    if href:
                        s = scorer.score_url(href)
                        scores.append(s)
                if scores:
                    url_risk_score = max(scores)  # Highest risk URL
                    result.raw_signals["url_risk_ml_score"] = url_risk_score
                    result.raw_signals["url_risk_ml_count"] = len(
                        [s for s in scores if s > 0.5]
                    )
        except (ImportError, Exception) as e:
            logger.debug(f"URL risk scorer not available: {e}")

        for link in links:
            href = link.get("href", "")
            display_text = link.get("display_text", "")

            # ── Check 1: Href vs Display Text Mismatch ──────
            self._check_href_mismatch(href, display_text, result)

            # ── Check 2: URL Obfuscation ────────────────────
            self._check_url_obfuscation(href, result)

            # ── Check 3: Punycode / Homograph Attacks ───────
            self._check_homograph_attack(href, result)

            # ── Check 4: Suspicious TLDs ────────────────────
            self._check_suspicious_tld(href, result)

            # ── Check 5: Image-Wrapped Links ─────────────────
            self._check_image_wrapped_links(link, result)

            # ── Check 6: Login URL Patterns ──────────────────
            self._check_login_url_patterns(href, result)

        # ── Compute Aggregate Score ─────────────────────
        max_score = self.config.get("max_score", 25)
        raw_total = (
            result.href_mismatch_score
            + result.url_obfuscation_score
            + result.homograph_attack_score
            + min(result.suspicious_tld_score, 10)
            + result.image_wrapped_link_score
            + result.login_url_pattern_score
        )

        # Add ML URL risk boost (scaled to max 10 points)
        if url_risk_score > 0.7:
            raw_total += min(url_risk_score * 10, 10)

        result.score = min(raw_total, max_score)

        logger.info(
            f"[Engine 4 - Links] Score: {result.score}/{max_score} | "
            f"Links={result.total_links_found}, "
            f"HrefMismatch={result.href_mismatch_detected}, "
            f"Obfuscation={result.url_obfuscation_detected}, "
            f"Homograph={result.homograph_attack_detected}"
        )

        return result

    # ────────────────────────────────────────────────────────
    # Link Extraction
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_links(html_body: str) -> list[dict]:
        """
        Extract all <a href="..."> links with their display text.

        Returns list of dicts: [{'href': '...', 'display_text': '...'}, ...]
        """
        links = []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_body, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                display_text = a_tag.get_text(strip=True)

                # Skip mailto: and javascript: links
                if href.lower().startswith(("mailto:", "javascript:", "tel:", "#")):
                    continue

                link_entry = {
                    "href": href,
                    "display_text": display_text,
                }

                if not display_text and a_tag.find("img"):
                    link_entry["display_text"] = "[IMAGE_LINK]"
                    link_entry["is_image_wrapped"] = True

                links.append(link_entry)

        except ImportError:
            # Fallback regex extraction
            pattern = re.compile(
                r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            for match in pattern.finditer(html_body):
                href = match.group(1).strip()
                display_text = re.sub(r"<[^>]+>", "", match.group(2)).strip()

                if not href.lower().startswith(("mailto:", "javascript:")):
                    links.append({
                        "href": href,
                        "display_text": display_text,
                    })

        return links

    @staticmethod
    def _extract_plain_text_urls(text: str) -> list[str]:
        """
        Extract all URLs from plain text content using regex.

        Catches URLs that aren't wrapped in <a> tags, such as:
        - https://example.com/path
        - http://192.168.1.1/login
        - www.example.com/page
        """
        if not text:
            return []

        # Match http/https URLs
        url_pattern = re.compile(
            r'https?://[^\s<>"\')\]]+',
            re.IGNORECASE,
        )
        urls = url_pattern.findall(text)

        # Also match www. URLs without scheme
        www_pattern = re.compile(
            r'(?<!://)www\.[^\s<>"\')\]]+',
            re.IGNORECASE,
        )
        for match in www_pattern.findall(text):
            urls.append("http://" + match)

        # Clean trailing punctuation
        cleaned = []
        for url in urls:
            # Remove trailing dots, commas, semicolons, parentheses
            url = url.rstrip('.,;:!?)\"\'')
            if len(url) > 10:  # Skip very short "URLs"
                cleaned.append(url)

        return cleaned

    # ────────────────────────────────────────────────────────
    # Detection Methods
    # ────────────────────────────────────────────────────────

    def _check_href_mismatch(
        self,
        href: str,
        display_text: str,
        result: LinkAnalysisResult,
    ) -> None:
        """
        Detect when the display text shows one URL/domain/brand but
        the href points to a completely different domain.

        Three tiers of detection:
        - Tier 1: Display text contains a full URL
        - Tier 2: Display text contains a bare domain (e.g. paypal.com)
        - Tier 3: Display text contains a known brand name
        """
        if not display_text or not href:
            return

        try:
            actual_domain = self._extract_domain(href)
            if not actual_domain:
                return
            actual_base = self._get_base_domain(actual_domain)

            # Tier 1: Display text contains a full URL
            url_pattern = re.compile(
                r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
                re.IGNORECASE,
            )
            url_match = url_pattern.search(display_text)

            if url_match:
                displayed_url = url_match.group(0)
                displayed_domain = self._extract_domain(displayed_url)
                if displayed_domain:
                    displayed_base = self._get_base_domain(displayed_domain)
                    if displayed_base != actual_base:
                        result.href_mismatch_detected = True
                        weight = self.config.get("href_mismatch", {}).get("weight", 25)
                        result.href_mismatch_score = weight
                        result.mismatched_links.append({
                            "displayed_url": displayed_url,
                            "actual_href": href,
                            "displayed_domain": displayed_domain,
                            "actual_domain": actual_domain,
                            "tier": 1,
                        })
                        logger.warning(
                            f"Href mismatch (Tier 1): displays '{displayed_domain}' "
                            f"but links to '{actual_domain}'"
                        )
                return

            display_lower = display_text.lower()

            # Tier 2: Display text contains a bare domain
            bare_match = BARE_DOMAIN_RE.search(display_lower)
            if bare_match:
                displayed_bare_domain = bare_match.group(1)
                displayed_base = self._get_base_domain(displayed_bare_domain)
                if displayed_base != actual_base:
                    result.href_mismatch_detected = True
                    weight = self.config.get("href_mismatch", {}).get("weight", 25)
                    result.href_mismatch_score = weight
                    result.mismatched_links.append({
                        "displayed_domain": displayed_bare_domain,
                        "actual_href": href,
                        "actual_domain": actual_domain,
                        "tier": 2,
                    })
                    logger.warning(
                        f"Href mismatch (Tier 2): displays bare domain '{displayed_bare_domain}' "
                        f"but links to '{actual_domain}'"
                    )
                return

            # Tier 3: Display text contains a known brand name
            for brand, domains in BRAND_DOMAINS.items():
                if brand in display_lower:
                    brand_matches_href = any(
                        actual_base == self._get_base_domain(d) for d in domains
                    )
                    if not brand_matches_href:
                        result.href_mismatch_detected = True
                        weight = self.config.get("href_mismatch", {}).get("weight", 25)
                        result.href_mismatch_score = weight
                        result.mismatched_links.append({
                            "brand_mentioned": brand,
                            "expected_domains": domains,
                            "actual_href": href,
                            "actual_domain": actual_domain,
                            "tier": 3,
                        })
                        logger.warning(
                            f"Href mismatch (Tier 3): mentions brand '{brand}' "
                            f"but links to '{actual_domain}'"
                        )
                    break

        except Exception as e:
            logger.debug(f"Href mismatch check error: {e}")

    def _check_url_obfuscation(
        self,
        href: str,
        result: LinkAnalysisResult,
    ) -> None:
        """
        Detect URL obfuscation techniques:

        1. URL shorteners (bit.ly, tinyurl.com, etc.) that hide
           the true destination.
        2. Excessively deep URL paths used to bury the real domain
           in a wall of path segments.
        3. IP address URLs (http://192.168.1.1/login).
        4. URL encoding abuse (%68%74%74%70...).
        5. @ symbol abuse (http://paypal.com@evil.com).
        """
        if not href:
            return

        parsed = urlparse(href)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""

        obfuscation_detected = False

        # Check 1: URL Shortener
        for shortener in self._shortener_domains:
            if hostname == shortener or hostname.endswith(f".{shortener}"):
                result.shortener_urls.append(href)
                obfuscation_detected = True
                break

        # Check 2: Excessive path depth
        # BUT: Skip for known legitimate email tracking domains.
        # Marketing platforms (SendGrid, Mailchimp, Uber, etc.) use
        # long Base64-encoded tracking paths that look "deep" but are benign.
        known_tracking_patterns = [
            ".mgm.", ".sendgrid.", ".mailchimp.", ".campaign-archive.",
            ".list-manage.", ".exact-target.", ".salesforce.",
            ".hubspot.", ".mailgun.", ".constantcontact.",
            ".pardot.", ".marketo.", ".eloqua.", ".moengage.",
            ".braze.", ".iterable.", ".sendinblue.", ".klaviyo.",
            ".sng.link", ".branch.io", ".app.link",
            ".tiktok.com", ".uber.com",
        ]
        is_tracking_domain = any(
            pattern in hostname or hostname.endswith(pattern.lstrip("."))
            for pattern in known_tracking_patterns
        )

        image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico")
        is_image_asset = path.lower().endswith(image_extensions) or "cloudfront.net" in hostname

        if not is_tracking_domain and not is_image_asset:
            max_depth = self.config.get("url_obfuscation", {}).get(
                "max_path_depth", 5
            )
            path_segments = [s for s in path.split("/") if s]
            if len(path_segments) > max_depth:
                obfuscation_detected = True
                result.raw_signals["deep_path_url"] = href

        # Check 3: IP address as hostname
        ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
        if ip_pattern.match(hostname):
            obfuscation_detected = True
            result.raw_signals["ip_address_url"] = href

        # Check 4: Excessive URL encoding
        decoded = unquote(href)
        encoding_ratio = 1 - (len(decoded) / len(href)) if len(href) > 0 else 0
        if encoding_ratio > 0.3:  # >30% was URL-encoded
            obfuscation_detected = True
            result.raw_signals["heavy_encoding_url"] = href

        # Check 5: @ symbol abuse (Basic Auth in URL)
        if parsed.username is not None and not href.startswith("mailto:"):
            obfuscation_detected = True
            result.raw_signals["at_symbol_abuse"] = href

        if obfuscation_detected:
            result.url_obfuscation_detected = True
            result.obfuscated_urls.append(href)
            weight = self.config.get("url_obfuscation", {}).get("weight", 15)
            result.url_obfuscation_score = max(result.url_obfuscation_score, weight)

    def _check_homograph_attack(
        self,
        href: str,
        result: LinkAnalysisResult,
    ) -> None:
        """
        Detect Punycode / IDN Homograph attacks.

        Attackers register domains using characters from other scripts
        (Cyrillic, Greek) that are visually identical to Latin characters.

        Example: "аpple.com" — the 'а' is Cyrillic U+0430, not Latin 'a'.
        In a browser, this displays as "xn--pple-43d.com" (Punycode).

        Detection methods:
        1. Check for mixed-script characters in the domain
        2. Check for Punycode (xn--) encoded domains
        3. Map individual characters through homoglyph tables
        """
        if not href:
            return

        parsed = urlparse(href)
        hostname = parsed.hostname or ""

        if not hostname:
            return

        # Check 1: Punycode-encoded domain (xn-- prefix)
        if "xn--" in hostname.lower():
            try:
                decoded_hostname = hostname.encode("ascii").decode("idna")
                result.homograph_attack_detected = True
                result.homograph_urls.append({
                    "punycode": hostname,
                    "decoded": decoded_hostname,
                    "full_url": href,
                })
                weight = self.config.get("homograph", {}).get("weight", 20)
                result.homograph_attack_score = max(
                    result.homograph_attack_score, weight
                )
                logger.warning(
                    f"Punycode domain detected: {hostname} → {decoded_hostname}"
                )
            except Exception:
                pass
            return

        # Check 2: Mixed-script detection
        has_non_ascii = any(ord(c) > 127 for c in hostname)
        if not has_non_ascii:
            return

        # Analyze character scripts
        scripts_found = set()
        homoglyph_chars = []

        for char in hostname:
            if char in (".", "-"):
                continue

            # Check against known homoglyph maps
            if char in CYRILLIC_HOMOGLYPHS:
                homoglyph_chars.append({
                    "char": char,
                    "codepoint": f"U+{ord(char):04X}",
                    "looks_like": CYRILLIC_HOMOGLYPHS[char],
                    "script": "Cyrillic",
                })
                scripts_found.add("Cyrillic")
            elif char in GREEK_HOMOGLYPHS:
                homoglyph_chars.append({
                    "char": char,
                    "codepoint": f"U+{ord(char):04X}",
                    "looks_like": GREEK_HOMOGLYPHS[char],
                    "script": "Greek",
                })
                scripts_found.add("Greek")
            else:
                try:
                    script = unicodedata.name(char, "").split()[0]
                    scripts_found.add(script)
                except (ValueError, IndexError):
                    pass

        # Mixed scripts = homograph attack
        if len(scripts_found) > 1 or homoglyph_chars:
            # Reconstruct what the domain "looks like" in Latin
            latin_version = ""
            for char in hostname:
                if char in CYRILLIC_HOMOGLYPHS:
                    latin_version += CYRILLIC_HOMOGLYPHS[char]
                elif char in GREEK_HOMOGLYPHS:
                    latin_version += GREEK_HOMOGLYPHS[char]
                else:
                    latin_version += char

            result.homograph_attack_detected = True
            result.homograph_urls.append({
                "original": hostname,
                "latin_equivalent": latin_version,
                "scripts_detected": list(scripts_found),
                "homoglyph_chars": homoglyph_chars,
                "full_url": href,
            })
            weight = self.config.get("homograph", {}).get("weight", 20)
            result.homograph_attack_score = max(
                result.homograph_attack_score, weight
            )
            logger.warning(
                f"Homograph attack: '{hostname}' looks like "
                f"'{latin_version}' (scripts: {scripts_found})"
            )

    def _check_suspicious_tld(
        self,
        href: str,
        result: LinkAnalysisResult,
    ) -> None:
        """Flag URLs with commonly abused TLDs."""
        try:
            parsed = urlparse(href)
            hostname = (parsed.hostname or "").lower()

            for tld in SUSPICIOUS_TLDS:
                if hostname.endswith(tld):
                    result.suspicious_tlds.append(hostname)
                    result.suspicious_tld_score += 5
                    result.raw_signals.setdefault("suspicious_tld_urls", [])
                    result.raw_signals["suspicious_tld_urls"].append(href)
                    break
        except Exception:
            pass

    def _check_image_wrapped_links(
        self,
        link: dict,
        result: LinkAnalysisResult,
    ) -> None:
        """Detect links that wrap an image instead of visible text."""
        if not link.get("is_image_wrapped"):
            return

        result.image_wrapped_link_detected = True
        result.image_wrapped_link_score = 10
        result.raw_signals.setdefault("image_wrapped_links", [])
        result.raw_signals["image_wrapped_links"].append(link.get("href", ""))

        logger.warning(
            f"Image-wrapped link detected: {link.get('href', '')}"
        )

    def _check_login_url_patterns(
        self,
        href: str,
        result: LinkAnalysisResult,
    ) -> None:
        """Detect login-related keywords in URL paths."""
        if not href:
            return

        try:
            parsed = urlparse(href)
            path = (parsed.path or "").lower()
            hostname = (parsed.hostname or "").lower()

            login_keywords = [
                "/login", "/signin", "/sign-in", "/verify",
                "/account/update", "/secure", "/auth",
                "/password", "/credential", "/confirm-identity",
            ]

            has_login_path = any(kw in path for kw in login_keywords)
            if not has_login_path:
                return

            is_suspicious_tld = any(
                hostname.endswith(tld) for tld in SUSPICIOUS_TLDS
            )
            is_ip = re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)

            if is_suspicious_tld or is_ip:
                result.login_url_pattern_detected = True
                result.login_url_pattern_score = 5
                result.raw_signals.setdefault("login_url_patterns", [])
                result.raw_signals["login_url_patterns"].append(href)

                logger.warning(
                    f"Login URL pattern on suspicious domain: {href}"
                )
        except Exception:
            pass

    # ────────────────────────────────────────────────────────
    # Utility Methods
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        """Extract the hostname from a URL string."""
        if not url:
            return None

        # Add scheme if missing
        if not re.match(r"https?://", url, re.IGNORECASE):
            url = "http://" + url

        try:
            parsed = urlparse(url)
            return (parsed.hostname or "").lower()
        except Exception:
            return None

    @staticmethod
    def _get_base_domain(hostname: str) -> str:
        """
        Extract the registerable base domain.
        
        Uses tldextract for accurate TLD-aware extraction.
        Falls back to simple last-two-segments extraction.
        
        Example: "login.secure.paypal.com" → "paypal.com"
        """
        try:
            import tldextract

            extracted = tldextract.extract(hostname)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}"
            return hostname
        except ImportError:
            # Fallback: naive extraction
            parts = hostname.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return hostname

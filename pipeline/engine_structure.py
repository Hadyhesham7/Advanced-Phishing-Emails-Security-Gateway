# ============================================================
# Engine 2: Structural & HTML Analysis (Visual Evasion)
# ============================================================
# Detection Targets:
#   - Hidden Text / Bayesian Poisoning
#   - Zero-Width Characters
#   - Image-Only Emails (with OCR)
#   - Brand Impersonation (CSS/Logo analysis)
#
# ML Requirement: PARTIAL
#   - OCR uses pre-trained models (Tesseract/EasyOCR) — no custom training
#   - Brand logo matching could use a CNN, but we start with perceptual hashing
#   - Hidden text & zero-width detection are pure heuristics
# ============================================================

from __future__ import annotations

import re
from typing import Optional
from io import BytesIO

from loguru import logger

from pipeline.models import StructuralAnalysisResult


# ── Zero-Width Character Definitions ────────────────────────
ZERO_WIDTH_CHARS: set[str] = {
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner
    "\u200D",  # Zero-width joiner
    "\uFEFF",  # Zero-width no-break space (BOM)
    "\u2060",  # Word joiner
    "\u180E",  # Mongolian vowel separator
    "\u00AD",  # Soft hyphen
}

# ── CSS Properties Indicating Hidden Content ────────────────
HIDDEN_CSS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("display_none", re.compile(r"display\s*:\s*none", re.IGNORECASE)),
    ("visibility_hidden", re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)),
    ("zero_font_size", re.compile(r"font-size\s*:\s*0", re.IGNORECASE)),
    ("negative_positioning", re.compile(
        r"(?:left|top|margin-left|margin-top)\s*:\s*-\d{3,}",
        re.IGNORECASE
    )),
    ("zero_opacity", re.compile(r"opacity\s*:\s*0(?:[;\s]|$)", re.IGNORECASE)),
    ("overflow_hidden_zero_height", re.compile(
        r"(?:height|max-height)\s*:\s*0.*overflow\s*:\s*hidden",
        re.IGNORECASE | re.DOTALL
    )),
]


class StructuralAnalysisEngine:
    """
    Engine 2 — Analyzes HTML structure for visual evasion tactics.

    Primarily heuristic-based, with pre-trained OCR models for
    image-only email extraction (no custom training required).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._ocr_engine = None

    def analyze(
        self,
        html_body: Optional[str],
        plain_text: Optional[str],
        attachments: Optional[list[dict]] = None,
    ) -> StructuralAnalysisResult:
        """
        Run all structural analysis checks on the email body.

        Args:
            html_body: The HTML content of the email (if available).
            plain_text: Plain text content (if available).
            attachments: List of attachment dicts with keys
                         'filename', 'content_type', 'data' (bytes).

        Returns:
            StructuralAnalysisResult with all flags and weighted score.
        """
        result = StructuralAnalysisResult()

        if html_body:
            # ── Check 1: Hidden Text / Bayesian Poisoning ───
            self._check_hidden_text(html_body, result)

            # ── Check 2: Zero-Width Characters ──────────────
            self._check_zero_width_chars(html_body, plain_text, result)

            # ── Check 3: Image-Only Email Detection ─────────
            self._check_image_only(html_body, plain_text, attachments, result)

            # ── Check 4: Brand Impersonation ────────────────
            self._check_brand_impersonation(html_body, result)

            # ── Check 5: Credential Form Detection ──────────
            self._check_credential_forms(html_body, result)

        elif plain_text:
            # Still check for zero-width in plain text
            self._check_zero_width_chars(None, plain_text, result)

        # ── Check 6: Macro Attachment Detection ─────────
        # Runs regardless of body type — macros are in attachments
        if attachments:
            self._check_macro_attachments(attachments, result)

        # ── Compute Aggregate Score (capped at max_score) ───
        max_score = self.config.get("max_score", 20)
        raw_total = (
            result.hidden_text_score
            + result.zero_width_chars_score
            + result.image_only_score
            + result.brand_impersonation_score
            + result.credential_form_score
            + result.macro_score
        )
        result.score = min(raw_total, max_score)

        logger.info(
            f"[Engine 2 - Structure] Score: {result.score}/{max_score} | "
            f"HiddenText={result.hidden_text_detected}, "
            f"ZeroWidth={result.zero_width_chars_detected}, "
            f"ImageOnly={result.image_only_email}, "
            f"BrandImpersonation={result.brand_impersonation_detected}, "
            f"CredentialForm={result.credential_form_detected}, "
            f"Macro={result.macro_detected}"
        )

        return result

    # ────────────────────────────────────────────────────────
    # Detection Methods
    # ────────────────────────────────────────────────────────

    def _check_hidden_text(
        self, html_body: str, result: StructuralAnalysisResult
    ) -> None:
        """
        Detect hidden text used for Bayesian poisoning.

        Attackers embed invisible text (white-on-white, display:none,
        zero font-size) to confuse Bayesian spam filters by injecting
        "clean" words that lower the spam probability.

        Detection methods:
        1. CSS display:none / visibility:hidden
        2. font-size:0
        3. Matching foreground/background colors
        4. Extreme negative positioning (off-screen)
        5. Zero opacity
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed, skipping hidden text check")
            return

        soup = BeautifulSoup(html_body, "html.parser")
        detected_methods = []
        hidden_content_parts = []

        # Check inline styles for hidden CSS
        for tag in soup.find_all(style=True):
            style = tag.get("style", "")
            for method_name, pattern in HIDDEN_CSS_PATTERNS:
                if pattern.search(style):
                    detected_methods.append(method_name)
                    text = tag.get_text(strip=True)
                    if text:
                        hidden_content_parts.append(text)

        # Check for matching foreground/background colors
        self._check_matching_colors(soup, detected_methods, hidden_content_parts)

        # Check <style> blocks for class-based hiding
        for style_tag in soup.find_all("style"):
            style_content = style_tag.string or ""
            for method_name, pattern in HIDDEN_CSS_PATTERNS:
                if pattern.search(style_content):
                    detected_methods.append(f"{method_name}_in_stylesheet")

        # Check for <iframe>, <object>, <embed> tags
        for tag_name in ("iframe", "object", "embed"):
            for tag in soup.find_all(tag_name):
                src = tag.get("src") or tag.get("data") or ""
                detected_methods.append(f"hidden_{tag_name}")
                if src:
                    hidden_content_parts.append(f"{tag_name}:{src}")

        if detected_methods:
            result.hidden_text_detected = True
            result.hidden_text_methods = list(set(detected_methods))
            result.hidden_text_content = " | ".join(hidden_content_parts[:5])
            weight = self.config.get("hidden_text", {}).get("weight", 15)
            result.hidden_text_score = weight
            result.raw_signals["hidden_text_methods"] = result.hidden_text_methods

    def _check_matching_colors(
        self,
        soup,
        detected_methods: list[str],
        hidden_content: list[str],
    ) -> None:
        """
        Detect text hidden via matching foreground and background colors.
        
        E.g., white text on a white background: 
            <span style="color: #FFFFFF; background: #FFFFFF">hidden text</span>
        """
        color_pattern = re.compile(
            r"(?:^|;)\s*color\s*:\s*([^;]+)",
            re.IGNORECASE,
        )
        bg_pattern = re.compile(
            r"(?:^|;)\s*background(?:-color)?\s*:\s*([^;]+)",
            re.IGNORECASE,
        )

        for tag in soup.find_all(style=True):
            style = tag.get("style", "")
            color_match = color_pattern.search(style)
            bg_match = bg_pattern.search(style)

            if color_match and bg_match:
                fg_color = self._normalize_color(color_match.group(1).strip())
                bg_color = self._normalize_color(bg_match.group(1).strip())

                if fg_color and bg_color and fg_color == bg_color:
                    detected_methods.append("matching_fg_bg_color")
                    text = tag.get_text(strip=True)
                    if text:
                        hidden_content.append(text)

    def _check_zero_width_chars(
        self,
        html_body: Optional[str],
        plain_text: Optional[str],
        result: StructuralAnalysisResult,
    ) -> None:
        """
        Detect zero-width characters inserted to break keyword matching.

        Attackers insert invisible Unicode characters to split
        blacklisted words:
            P\u200Bay\u200Bpal → displays as "PayPal" but won't match
            the string "PayPal" in simple filters.

        We count and strip them to produce cleaned text for downstream
        NLP analysis.
        """
        content = html_body or plain_text or ""

        # Strip HTML tags if analyzing HTML
        if html_body:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_body, "html.parser")
                
                # Strip hidden nodes to prevent Bayesian Poisoning of NLP
                for tag in soup.find_all(style=True):
                    # Skip if tag was already decomposed via a parent
                    if getattr(tag, 'attrs', None) is None:
                        continue
                        
                    style = str(tag.get("style", "")).lower()
                    # Strip basic inline hiding techniques
                    if ("display: none" in style or 
                        "visibility: hidden" in style or 
                        "opacity: 0" in style or 
                        "font-size: 0" in style or
                        "display:none" in style or
                        "visibility:hidden" in style or
                        "opacity:0" in style or
                        "font-size:0" in style):
                        tag.decompose()
                
                content = soup.get_text(separator=" ")
            except ImportError:
                content = re.sub(r"<[^>]+>", "", content)

        zwc_count = sum(1 for char in content if char in ZERO_WIDTH_CHARS)

        if zwc_count > 0:
            result.zero_width_chars_detected = True
            result.zero_width_char_count = zwc_count

            # Graduated scoring based on ZWC density
            if zwc_count <= 5:
                result.zero_width_chars_score = 5
            elif zwc_count <= 20:
                result.zero_width_chars_score = 10
            else:
                weight = self.config.get("zero_width_chars", {}).get("weight", 15)
                result.zero_width_chars_score = weight

            # Produce cleaned text for downstream engines
            cleaned = "".join(
                char for char in content if char not in ZERO_WIDTH_CHARS
            )
            result.cleaned_text = cleaned
            result.raw_signals["zwc_count"] = zwc_count

            logger.warning(
                f"Zero-width characters detected: {zwc_count} instances "
                f"(score={result.zero_width_chars_score})"
            )

    def _check_image_only(
        self,
        html_body: str,
        plain_text: Optional[str],
        attachments: Optional[list[dict]],
        result: StructuralAnalysisResult,
    ) -> None:
        """
        Detect image-only emails that lack meaningful body text.

        Many phishing emails embed the entire message as an image
        to evade text-based filters. We check the text-to-image
        ratio and optionally run OCR on embedded images.

        OCR Model: Pre-trained Tesseract or EasyOCR — NO custom
        training required. These are off-the-shelf OCR engines.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return

        soup = BeautifulSoup(html_body, "html.parser")

        # Extract visible text
        visible_text = soup.get_text(strip=True)
        text_length = len(visible_text)

        # Count image elements
        images = soup.find_all("img")
        inline_images = soup.find_all(
            lambda tag: tag.name and tag.get("style", "")
            and "background-image" in tag.get("style", "")
        )
        total_images = len(images) + len(inline_images)

        # Calculate text-to-image ratio
        text_to_image_ratio = (
            text_length / (text_length + total_images * 500)
            if total_images > 0 else 1.0
        )
        min_ratio = self.config.get("image_only", {}).get("min_text_ratio", 0.10)

        # Heuristic: flag if low text-to-image ratio with sanity guard
        if total_images > 0 and text_to_image_ratio < min_ratio and text_length < 100:
            result.image_only_email = True
            result.text_to_image_ratio = text_to_image_ratio
            weight = self.config.get("image_only", {}).get("weight", 10)
            result.image_only_score = weight

            # Attempt OCR on attachments
            if attachments:
                ocr_text = self._run_ocr_on_images(attachments)
                if ocr_text:
                    result.ocr_extracted_text = ocr_text

            result.raw_signals["total_images"] = total_images
            result.raw_signals["visible_text_length"] = text_length

            logger.warning(
                f"Image-only email detected: {total_images} images, "
                f"{text_length} chars of text"
            )

    def _run_ocr_on_images(self, attachments: list[dict]) -> Optional[str]:
        """
        Run OCR on image attachments to extract hidden text.

        Uses Tesseract (pytesseract) as the primary OCR engine
        with EasyOCR as fallback.

        These are PRE-TRAINED models — no custom dataset needed.
        """
        extracted_texts = []
        ocr_engine = self.config.get("image_only", {}).get(
            "ocr_engine", "tesseract"
        )

        for attachment in attachments:
            content_type = attachment.get("content_type", "")
            if not content_type.startswith("image/"):
                continue

            image_data = attachment.get("data", b"")
            if not image_data:
                continue

            try:
                if ocr_engine == "tesseract":
                    text = self._ocr_tesseract(image_data)
                else:
                    text = self._ocr_easyocr(image_data)

                if text and text.strip():
                    extracted_texts.append(text.strip())

            except Exception as e:
                logger.debug(f"OCR failed for attachment: {e}")

        return " ".join(extracted_texts) if extracted_texts else None

    @staticmethod
    def _ocr_tesseract(image_data: bytes) -> Optional[str]:
        """Extract text using Tesseract OCR (pre-trained)."""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(BytesIO(image_data))
            return pytesseract.image_to_string(image)
        except ImportError:
            logger.debug("pytesseract not installed")
            return None

    @staticmethod
    def _ocr_easyocr(image_data: bytes) -> Optional[str]:
        """Extract text using EasyOCR (pre-trained neural model)."""
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_data)
            return " ".join([text for _, text, _ in results])
        except ImportError:
            logger.debug("easyocr not installed")
            return None

    def _check_brand_impersonation(
        self, html_body: str, result: StructuralAnalysisResult
    ) -> None:
        """
        Detect brand impersonation via CSS analysis and known brand markers.

        Checks for:
        1. Known brand color schemes in CSS
        2. Brand-specific CSS class names or IDs
        3. References to brand logos from non-brand domains
        4. Favicon/logo URLs pointing to legitimate brands
           but email sent from unrelated domain

        NOTE: Full logo matching (CNN-based) would require a trained
        model. This initial implementation uses perceptual hashing
        and CSS heuristics, which are rule-based.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return

        soup = BeautifulSoup(html_body, "html.parser")

        # Known brand CSS patterns (simplified)
        brand_indicators = {
            "paypal": {
                "colors": ["#003087", "#009cde", "#012169"],
                "keywords": ["paypal", "pp-header", "pp-button"],
            },
            "microsoft": {
                "colors": ["#0078d4", "#00bcf2", "#737373"],
                "keywords": ["microsoft", "ms-", "office365"],
            },
            "google": {
                "colors": ["#4285f4", "#34a853", "#fbbc05", "#ea4335"],
                "keywords": ["google", "gmail", "g-suite"],
            },
            "apple": {
                "colors": ["#333333", "#0071e3"],
                "keywords": ["apple", "icloud", "apple-id"],
            },
            "amazon": {
                "colors": ["#ff9900", "#232f3e", "#146eb4"],
                "keywords": ["amazon", "aws", "prime"],
            },
        }

        html_lower = html_body.lower()
        detected_brand = None

        for brand, indicators in brand_indicators.items():
            # Check for brand-specific keywords in HTML
            keyword_matches = sum(
                1 for kw in indicators["keywords"] if kw in html_lower
            )
            # Check for brand colors in CSS
            color_matches = sum(
                1 for color in indicators["colors"] if color in html_lower
            )

            # If multiple indicators match, likely impersonation
            if keyword_matches >= 2 or (keyword_matches >= 1 and color_matches >= 1):
                detected_brand = brand
                break

        # Cross-reference: if brand detected in HTML but sender domain
        # doesn't match, flag as impersonation
        # (The actual cross-reference with sender domain happens in
        #  the aggregator where we have access to Engine 1 results)
        if detected_brand:
            # Check for external logo references
            for img in soup.find_all("img", src=True):
                src = img["src"].lower()
                if detected_brand in src and "http" in src:
                    result.brand_impersonation_detected = True
                    result.impersonated_brand = detected_brand
                    weight = self.config.get("brand_impersonation", {}).get(
                        "weight", 15
                    )
                    result.brand_impersonation_score = weight
                    result.raw_signals["impersonated_brand"] = detected_brand
                    result.raw_signals["brand_logo_url"] = src
                    logger.warning(
                        f"Brand impersonation detected: {detected_brand}"
                    )
                    break

    def _check_credential_forms(
        self, html_body: str, result: StructuralAnalysisResult
    ) -> None:
        """
        Detect credential harvesting forms embedded in the email body.

        Flags password fields, credential-related inputs, and forms
        that POST to external URLs.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return

        soup = BeautifulSoup(html_body, "html.parser")
        details = []
        credential_keywords = {
            "password", "passwd", "pass", "email",
            "username", "login", "user", "credential",
        }

        # Check <input> tags for password fields or credential-related names
        for inp in soup.find_all("input"):
            input_type = (inp.get("type") or "").lower()
            input_name = (inp.get("name") or "").lower()
            input_placeholder = (inp.get("placeholder") or "").lower()

            if input_type == "password":
                details.append(f"password_field: name={inp.get('name', '')}")
                continue

            for kw in credential_keywords:
                if kw in input_name or kw in input_placeholder:
                    details.append(f"credential_input: name={inp.get('name', '')} placeholder={inp.get('placeholder', '')}")
                    break

        # Check <form> tags with external action URLs
        for form in soup.find_all("form", action=True):
            action = form["action"].strip()
            if action.lower().startswith("http"):
                details.append(f"external_form_action: {action}")

        if details:
            result.credential_form_detected = True
            result.credential_form_score = 15
            result.credential_form_details = details
            result.raw_signals["credential_form_details"] = details
            logger.warning(
                f"Credential form detected: {len(details)} indicator(s) — "
                f"{', '.join(details[:3])}"
            )

    def _check_macro_attachments(
        self, attachments: list[dict], result: StructuralAnalysisResult
    ) -> None:
        """
        Detect VBA macros in Office document attachments.

        Checks for:
        1. OLE2 files (.doc, .xls, .ppt) — uses olefile to detect VBA streams
        2. OOXML files (.docm, .xlsm, .pptm) — checks for vbaProject.bin in ZIP
        3. Macro-enabled file extensions as a fallback heuristic

        Scoring:
        - +20 for confirmed VBA macro content detected in binary
        - +15 for macro-enabled file extension (without binary confirmation)
        """
        details = []

        # Macro-enabled extensions (high risk)
        MACRO_EXTENSIONS = {
            ".docm", ".xlsm", ".pptm",      # OOXML macro-enabled
            ".dotm", ".xltm", ".potm",      # OOXML macro-enabled templates
            ".doc", ".xls", ".ppt",          # Legacy OLE2 (can contain macros)
            ".xlsb",                          # Excel binary (macro-capable)
        }

        # Additional suspicious attachment types
        DANGEROUS_EXTENSIONS = {
            ".hta", ".vbs", ".vbe", ".js", ".jse",
            ".wsf", ".wsh", ".scr", ".bat", ".cmd",
            ".ps1", ".lnk", ".iso", ".img",
        }

        for attachment in attachments:
            filename = (attachment.get("filename") or "unknown").lower()
            data = attachment.get("data", b"")
            content_type = (attachment.get("content_type") or "").lower()

            # Extract extension
            ext = ""
            if "." in filename:
                ext = "." + filename.rsplit(".", 1)[-1]

            # ── Check 1: Dangerous non-Office extensions ────
            if ext in DANGEROUS_EXTENSIONS:
                details.append(f"dangerous_extension: {filename}")
                continue

            # ── Check 2: Macro-enabled extension heuristic ──
            is_macro_ext = ext in MACRO_EXTENSIONS
            macro_confirmed = False

            # ── Check 3: OLE2 binary analysis (.doc, .xls, .ppt) ──
            if data and len(data) > 8:
                # OLE2 magic bytes: D0 CF 11 E0 A1 B1 1A E1
                if data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                    macro_confirmed = self._check_ole2_macros(data, filename)
                    if macro_confirmed:
                        details.append(f"ole2_vba_macro: {filename}")

                # ── Check 4: OOXML ZIP analysis (.docm, .xlsm, .pptm) ──
                elif data[:2] == b'PK':
                    macro_confirmed = self._check_ooxml_macros(data, filename)
                    if macro_confirmed:
                        details.append(f"ooxml_vba_macro: {filename}")

            # ── Fallback: flag macro-enabled extension ──────
            if not macro_confirmed and is_macro_ext:
                details.append(f"macro_enabled_extension: {filename}")

        if details:
            # Determine score based on confirmation level
            has_confirmed = any(
                d.startswith("ole2_vba") or d.startswith("ooxml_vba")
                for d in details
            )
            has_dangerous = any(d.startswith("dangerous_ext") for d in details)

            result.macro_detected = True
            result.macro_score = 20 if (has_confirmed or has_dangerous) else 15
            result.macro_details = details
            result.raw_signals["macro_details"] = details
            result.raw_signals["macro_confirmed_in_binary"] = has_confirmed
            logger.warning(
                f"Macro attachment detected: {len(details)} indicator(s) — "
                f"{', '.join(details[:3])} "
                f"(confirmed={'yes' if has_confirmed else 'extension_only'})"
            )

    @staticmethod
    def _check_ole2_macros(data: bytes, filename: str) -> bool:
        """
        Check OLE2 compound files for VBA macro streams.

        Uses olefile if available, otherwise falls back to scanning
        for known VBA stream signatures in the raw binary.
        """
        try:
            import olefile
            ole = olefile.OleFileIO(data)
            # VBA macros live in streams like 'Macros/VBA' or '_VBA_PROJECT_CUR'
            for stream in ole.listdir():
                stream_path = "/".join(stream).lower()
                if "vba" in stream_path or "macro" in stream_path:
                    ole.close()
                    return True
            ole.close()
            return False
        except ImportError:
            # Fallback: scan raw bytes for VBA signatures
            vba_signatures = [
                b'_VBA_PROJECT',
                b'VBA',
                b'Attribute VB_',
                b'ThisDocument',
                b'Auto_Open',
                b'AutoOpen',
                b'Document_Open',
                b'Workbook_Open',
                b'Auto_Close',
                b'AutoExec',
            ]
            for sig in vba_signatures:
                if sig in data:
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _check_ooxml_macros(data: bytes, filename: str) -> bool:
        """
        Check OOXML ZIP files for vbaProject.bin.

        OOXML macro-enabled documents (.docm, .xlsm, .pptm) are ZIP
        archives. VBA macros are stored in 'word/vbaProject.bin',
        'xl/vbaProject.bin', or 'ppt/vbaProject.bin'.
        """
        import zipfile
        from io import BytesIO

        try:
            with zipfile.ZipFile(BytesIO(data), "r") as zf:
                for name in zf.namelist():
                    name_lower = name.lower()
                    if "vbaproject.bin" in name_lower:
                        return True
                    if "vba" in name_lower and name_lower.endswith(".bin"):
                        return True
            return False
        except (zipfile.BadZipFile, Exception):
            return False

    # ────────────────────────────────────────────────────────
    # Utility Methods
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_color(color_str: str) -> Optional[str]:
        """
        Normalize CSS color values for comparison.
        Converts named colors and rgb() to hex.
        """
        color_str = color_str.strip().lower()

        # Named colors (subset)
        named_colors = {
            "white": "#ffffff", "black": "#000000",
            "red": "#ff0000", "blue": "#0000ff",
            "green": "#008000", "transparent": None,
        }
        if color_str in named_colors:
            return named_colors[color_str]

        # Already hex
        if color_str.startswith("#"):
            hex_val = color_str.lstrip("#")
            if len(hex_val) == 3:
                hex_val = "".join(c * 2 for c in hex_val)
            return f"#{hex_val.lower()}"

        # rgb(r, g, b) format
        rgb_match = re.match(
            r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str
        )
        if rgb_match:
            r, g, b = (int(x) for x in rgb_match.groups())
            return f"#{r:02x}{g:02x}{b:02x}"

        return color_str

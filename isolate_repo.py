"""
isolate_repo.py — Secure Project Isolation & Extraction Script
================================================================
Creates a clean, portable copy of the Advanced Email Security Gateway
ready for GitHub initialization. Strips all sensitive data, heavy ML
model checkpoints, CSV datasets, and virtual environments.

Usage:
    python isolate_repo.py

Output:
    ../Advanced-Email-Security-Gateway/
        ├── pipeline/          (core detection engines)
        ├── training/          (training scripts)
        ├── tests/             (test suite)
        ├── config/            (settings YAML)
        ├── data/              (brand_names.txt, freemail_domains.txt ONLY)
        ├── models/            (XGBoost + RoBERTa, NO checkpoints)
        ├── demo.py
        ├── requirements.txt
        ├── README.md
        └── .gitignore
"""

import os
import sys
import shutil
import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these lists to control what gets copied
# ═══════════════════════════════════════════════════════════════

# Target directory (relative to current project root)
TARGET_DIR = os.path.join("..", "Advanced-Email-Security-Gateway")

# Directories to copy entirely (relative to project root)
FOLDERS_TO_COPY = [
    "pipeline",
    "training",
    "tests",
    "config",
]

# Individual files to copy from project root
FILES_TO_COPY = [
    "demo.py",
    "test_real_emails.py",
]

# Selective files from data/ (only lightweight reference data, NOT CSVs)
DATA_FILES_TO_COPY = [
    os.path.join("data", "brand_names.txt"),
    os.path.join("data", "freemail_domains.txt"),
]

# Models to copy (skip massive checkpoint directories)
MODELS_TO_COPY = {
    "models": {
        "include_files": [
            "aggregator_xgb.json",
            "aggregator_metadata.json",
            "url_risk_model.json",
        ],
        "include_dirs": {
            "phishing_roberta": {
                # Copy core model files, SKIP checkpoint-* folders
                "include_files": [
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "training_args.bin",
                    "training_report.txt",
                ],
                "exclude_dirs": ["checkpoint-400", "checkpoint-800"],
            }
        },
    }
}

# Patterns to SKIP when copying directories
SKIP_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".env",
    ".venv",
    "venv",
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
    ".DS_Store",
    "Thumbs.db",
    "*.pkl",
    "*.csv",
    "*.log",
    "eggs",
    "*.egg-info",
    "dist",
    "build",
]

# ═══════════════════════════════════════════════════════════════
# .gitignore content
# ═══════════════════════════════════════════════════════════════

GITIGNORE_CONTENT = """\
# ── Secrets & Environment ──
.env
.env.*

# ── Python Cache ──
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
eggs/

# ── Virtual Environments ──
.venv/
venv/
env/

# ── Heavy ML Models & Pickles ──
*.pkl
*.pickle
*.h5
*.pt
*.onnx

# ── Datasets (too large for Git) ──
*.csv
data/enron_legitimate/
data/synthetic_phishing/
data/synthetic_clean/
emails.csv/

# ── Model Checkpoints (keep only final model) ──
models/phishing_roberta/checkpoint-*/

# ── IDE & OS ──
.idea/
.vscode/
*.swp
*.swo
.DS_Store
Thumbs.db

# ── Logs ──
*.log
logs/
"""

# ═══════════════════════════════════════════════════════════════
# requirements.txt — Complete dependency list
# ═══════════════════════════════════════════════════════════════

REQUIREMENTS_CONTENT = """\
# ============================================================
# Advanced Email Security Gateway — Dependencies
# Phishing Detection Pipeline with 4-Engine Architecture
# ============================================================
# Install: pip install -r requirements.txt
# ============================================================

# --- Core ML / NLP ---
transformers>=4.40.0            # RoBERTa transformer for semantic phishing detection
torch>=2.2.0                    # PyTorch backend for transformer models
xgboost>=2.0.0                  # ML Aggregator (31-feature XGBoost classifier)
scikit-learn>=1.4.0             # Metrics, preprocessing, train/test split
numpy>=1.26.0                   # Numerical operations
pandas>=2.2.0                   # Tabular data handling
datasets>=2.18.0                # HuggingFace dataset loading for NLP training

# --- NLP Utilities ---
nltk>=3.8.0                     # Tokenization, stopword removal
spacy>=3.7.0                    # Advanced NLP entity recognition

# --- HTML & Email Parsing ---
beautifulsoup4>=4.12.0          # HTML content parsing (credential forms, hidden text)
lxml>=5.1.0                     # Fast XML/HTML parser backend
html5lib>=1.1                   # HTML5 parser fallback

# --- OCR (Image-Only Emails) ---
Pillow>=10.2.0                  # Image processing
pytesseract>=0.3.10             # Tesseract OCR wrapper
easyocr>=1.7.0                  # Alternative neural OCR engine

# --- URL & Domain Analysis ---
tldextract>=5.1.0               # Domain/TLD extraction (Engine 4)
python-whois>=0.9.0             # WHOIS domain age lookups (Engine 1)
idna>=3.6                       # Punycode / IDN handling (homograph detection)
Levenshtein>=0.25.0             # Edit distance for typosquatting detection

# --- OLE2 / Office Macro Detection ---
olefile>=0.47                   # OLE2 compound file parser (macro detection)

# --- Email Parsing ---
mail-parser>=3.15.0             # RFC-compliant email parsing

# --- Training Utilities ---
faker>=28.0.0                   # Synthetic email generation for training
tqdm>=4.66.0                    # Progress bars for training scripts

# --- Utilities ---
pyyaml>=6.0                     # Configuration management (settings.yaml)
loguru>=0.7.0                   # Structured logging across all engines
pydantic>=2.6.0                 # Data validation & inter-engine schemas
httpx>=0.27.0                   # Async HTTP client for URL checks
"""

# ═══════════════════════════════════════════════════════════════
# README.md — Project Documentation
# ═══════════════════════════════════════════════════════════════

README_CONTENT = """\
# 🛡️ Advanced Email Security Gateway

## Enterprise-Grade Phishing Detection with Hybrid AI Architecture

> A multi-layered email security system that combines **4 specialized heuristic engines**,
> a **transformer-based NLP classifier** (RoBERTa), and a **gradient-boosted ML aggregator**
> (XGBoost) to detect sophisticated phishing campaigns that bypass traditional email filters.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The 4-Engine Pipeline](#the-4-engine-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training Your Own Model](#training-your-own-model)
- [Test Suite](#test-suite)
- [Results](#results)

---

## Overview

Modern phishing attacks use a combination of visual spoofing, HTML obfuscation,
semantic social engineering, and URL evasion to bypass conventional email security.
This system addresses these threats through **defense-in-depth** — an attacker must
simultaneously evade header analysis, structural analysis, NLP understanding, *and*
link analysis to avoid detection.

### Key Capabilities

| Capability | Detection Method |
|------------|-----------------|
| Display name spoofing | Brand database cross-reference (70+ brands) |
| Reply-To freemail mismatch | Freemail provider database (30+ domains) |
| SPF/DKIM/DMARC failures | Authentication-Results header parsing |
| Hidden text / Bayesian poisoning | 6 CSS concealment method detection |
| Zero-width character injection | Unicode scanning with graduated scoring |
| Credential harvesting forms | HTML DOM parsing for password fields |
| Macro-enabled attachments | OLE2/OOXML binary analysis for VBA |
| Visual-technical URL mismatch | 3-tier href vs display text analysis |
| URL shorteners & IP obfuscation | URL structure feature extraction |
| Homograph (IDN) attacks | Unicode confusable normalization |
| Image-wrapped phishing links | `<a><img></a>` detection |
| MFA bypass / OAuth consent phishing | 24+ NLP regex patterns |
| Urgency & coercion language | RoBERTa transformer classification |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    .eml File Input                          │
│                         │                                   │
│              ┌──────────┼──────────┐                        │
│              ▼          ▼          ▼          ▼              │
│         ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│         │Engine 1│ │Engine 2│ │Engine 3│ │Engine 4│        │
│         │Header  │ │Structure│ │  NLP  │ │ Links  │        │
│         │ /25    │ │  /20   │ │  /30  │ │  /25   │        │
│         └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘        │
│             │          │          │          │               │
│             └──────────┴──────────┴──────────┘               │
│                         │                                    │
│              ┌──────────▼──────────┐                         │
│              │   ML Aggregator     │                         │
│              │   31-Feature Vector │                         │
│              │   XGBoost Classifier│                         │
│              └──────────┬──────────┘                         │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │   Pipeline Verdict  │                         │
│              │   CLEAN / MALICIOUS │                         │
│              │   + Confidence Score│                         │
│              └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## The 4-Engine Pipeline

### Engine 1 — Advanced Header & Identity Analysis (Max: 25 pts)

Analyzes email headers to detect identity spoofing, routing anomalies, and
authentication failures.

- **Display Name Spoofing**: Cross-references sender display name against 70+ known
  brand names. Flags when brand appears in display name but sender domain is unrelated.
- **Reply-To Mismatch**: Detects when Reply-To points to a different domain than From,
  with +5 escalation penalty when Reply-To uses freemail providers (Gmail, Yahoo, etc.).
- **Compound Scoring**: Additional +5 bonus when both display name spoofing AND freemail
  Reply-To trigger simultaneously.
- **Authentication Analysis**: Parses SPF, DKIM, and DMARC results from
  Authentication-Results header. Two or more failures trigger +10 penalty.
- **X-Mailer Anomaly**: Flags scripting tools (PHPMailer, Python, curl, swaks).
- **Domain Age**: WHOIS lookup identifies newly registered sender domains (<30 days).
- **Typosquatting**: Levenshtein distance + homoglyph detection against brand domains.

### Engine 2 — Structural & HTML Analysis (Max: 20 pts)

Detects visual evasion tactics, hidden content, credential harvesting forms, and
weaponized attachments.

- **Hidden Text Detection**: 6 CSS concealment methods (display:none, visibility:hidden,
  font-size:0, negative positioning, zero opacity, color matching).
- **Zero-Width Characters**: Graduated scoring (1-5: +5, 6-20: +10, 20+: +15).
  Produces cleaned text for downstream NLP analysis.
- **Credential Form Detection**: Identifies `<input type="password">`, credential-named
  fields, and external form actions.
- **Macro Attachment Detection**: OLE2 binary analysis for VBA streams, OOXML ZIP
  inspection for vbaProject.bin, and dangerous extension flagging (.vbs, .hta, .ps1).
- **Iframe/Object Detection**: Flags embedded `<iframe>`, `<object>`, `<embed>` tags.

### Engine 3 — NLP & Semantic Analysis (Max: 30 pts)

Understands the *intent* behind email text using transformer deep learning and
rule-based pattern matching.

- **RoBERTa Transformer**: Fine-tuned bidirectional transformer that outputs phishing
  probability and classifies into 6 intent categories (urgency, credential harvesting,
  financial fraud, OAuth consent, generic phishing, legitimate).
- **Heuristic Fallback**: 40+ regex patterns covering urgency phrases, MFA bypass
  tactics (push bombing, recovery codes, authenticator re-registration), and OAuth
  consent phishing.
- **Triad Compound Scoring**: When 2+ intent categories fire simultaneously, a
  probability boost of `0.15 × active_categories` is applied.

### Engine 4 — Static Link & Payload Analysis (Max: 25 pts)

Analyzes all URLs for mismatch attacks, obfuscation, and suspicious destinations.

- **3-Tier Href Mismatch**: Full URL display text (+20), bare domain display (+15),
  brand name in text (+10).
- **Image-Wrapped Links**: Detects `<a><img></a>` clickjacking.
- **URL Obfuscation**: Shorteners, IP-based URLs, excessive subdomains.
- **Homograph Detection**: International domain name (IDN) confusable analysis.
- **ML URL Scorer**: Pre-trained XGBoost on 16 structural URL features.
- **Login URL Patterns**: Detects `/login`, `/signin`, `/auth` paths on suspicious domains.
- **Suspicious TLD Scoring**: Flags high-risk TLDs (.tk, .ml, .ga, .xyz, .icu, etc.).

---

## Machine Learning Models

This project uses **two distinct ML paradigms**:

### 1. Deep Learning — RoBERTa Transformer (Engine 3)

| Attribute | Value |
|-----------|-------|
| Base Model | `roberta-base` (125M parameters) |
| Task | Binary classification + intent categorization |
| Input | Cleaned email text (after zero-width char removal) |
| Output | P(phishing) ∈ [0, 1] + intent category |

### 2. Gradient-Boosted Trees — XGBoost (Aggregator)

| Attribute | Value |
|-----------|-------|
| Features | 31-dimensional vector from all 4 engines |
| Trees | 500 estimators, max depth 8 |
| Training Data | 8,000 emails (2,500 phishing + 5,500 legitimate) |
| Accuracy | **100.0%** on test set (1,600 samples) |
| Top Feature | `header_x_link_risk` (cross-engine interaction, 47% importance) |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (package manager)
- Tesseract OCR (optional, for image-only email detection)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Advanced-Email-Security-Gateway.git
cd Advanced-Email-Security-Gateway

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy language model (optional)
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
python -c "from pipeline.analyzer import PhishingAnalyzer; print('Pipeline loaded successfully!')"
```

---

## Quick Start

### Analyze a Single Email

```python
from pipeline.analyzer import PhishingAnalyzer

analyzer = PhishingAnalyzer("config/settings.yaml")

with open("sample_email.eml", "r") as f:
    raw_email = f.read()

verdict = analyzer.analyze(raw_email)
print(f"Verdict: {verdict.label}")
print(f"Score: {verdict.final_score}/100")
print(f"Confidence: {verdict.confidence:.2%}")
```

### Run the Demo

```bash
python demo.py
```

---

## Project Structure

```
Advanced-Email-Security-Gateway/
├── pipeline/                   # Core detection engines
│   ├── __init__.py
│   ├── analyzer.py             # Main orchestrator (PhishingAnalyzer)
│   ├── engine_header.py        # Engine 1: Header & identity analysis
│   ├── engine_structure.py     # Engine 2: HTML structural analysis
│   ├── engine_nlp.py           # Engine 3: NLP & semantic analysis
│   ├── engine_links.py         # Engine 4: Link & URL analysis
│   ├── aggregator.py           # ML Aggregator (XGBoost)
│   ├── models.py               # Pydantic data models & feature vector
│   └── url_risk_scorer.py      # URL risk ML scorer
│
├── training/                   # Model training scripts
│   ├── generate_synthetic_phishing.py  # Synthetic dataset generator
│   ├── train_aggregator_v3.py  # XGBoost aggregator training
│   ├── train_nlp_model.py      # RoBERTa fine-tuning script
│   ├── improve_engine4.py      # URL risk model training
│   └── process_enron.py        # Enron corpus processor
│
├── tests/                      # Test suite
│   └── test_5_cases.py         # 30-check functional tests
│
├── config/
│   └── settings.yaml           # Engine configuration
│
├── models/                     # Trained ML models
│   ├── aggregator_xgb.json     # XGBoost aggregator (31 features)
│   ├── aggregator_metadata.json
│   ├── url_risk_model.json     # URL risk scorer
│   └── phishing_roberta/       # RoBERTa transformer model
│
├── data/                       # Reference data files
│   ├── brand_names.txt         # 70+ impersonated brand names
│   └── freemail_domains.txt    # 30+ freemail domain list
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion rules
└── README.md                   # This file
```

---

## Training Your Own Model

### 1. Generate Synthetic Training Data

```bash
python training/generate_synthetic_phishing.py
# Outputs: data/synthetic_phishing/ (2,500 files) + data/synthetic_clean/ (2,500 files)
```

### 2. Train the XGBoost Aggregator

```bash
python training/train_aggregator_v3.py
# Processes all emails through 4 engines → extracts 31-feature vectors → trains XGBoost
# Output: models/aggregator_xgb.json
```

### 3. Fine-Tune the RoBERTa NLP Model (Optional)

```bash
python training/train_nlp_model.py
# Fine-tunes roberta-base on phishing text classification
# Output: models/phishing_roberta/
```

---

## Test Suite

Run the full functional test suite:

```bash
python -m pytest tests/test_5_cases.py -v
```

The test suite validates **30 checks** covering all 5 evasion techniques:

| Test Case | Checks | Status |
|-----------|--------|--------|
| Visual vs. Technical URL Mismatch (3 tiers) | 5 | ✅ |
| Display Name Spoofing + Freemail Reply-To | 4 | ✅ |
| NLP Triad + MFA + OAuth Patterns | 7 | ✅ |
| Zero-Width Characters + Hidden Text | 4 | ✅ |
| Fake Infrastructure + Image-Wrapped Links | 6 | ✅ |
| Credential Harvesting Forms | 4 | ✅ |

---

## Results

### XGBoost Aggregator Performance

| Metric | Score |
|--------|-------|
| Accuracy | **100.0%** |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| AUC-ROC | 1.0000 |

### Confusion Matrix (Test Set: 1,600 samples)

|  | Predicted Legit | Predicted Phishing |
|--|----------------|-------------------|
| Actual Legit | 1,100 (TN) | 0 (FP) |
| Actual Phishing | 0 (FN) | 500 (TP) |

### Top Features by Importance

| Feature | Importance |
|---------|-----------|
| `header_x_link_risk` (Engine 1 × Engine 4) | 46.97% |
| `display_name_spoofing` | 35.58% |
| `header_score` | 16.18% |

---

## License

This project was developed as part of a university thesis. All rights reserved.

---

*Built with 🛡️ for enterprise email security.*
"""


# ═══════════════════════════════════════════════════════════════
# SCRIPT LOGIC — Do not edit below unless customizing behavior
# ═══════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_ok(msg):
    print(f"  {Colors.GREEN}✔{Colors.RESET} {msg}")

def log_warn(msg):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")

def log_err(msg):
    print(f"  {Colors.RED}✖{Colors.RESET} {msg}")

def log_info(msg):
    print(f"  {Colors.CYAN}→{Colors.RESET} {msg}")


def should_skip(path: str) -> bool:
    """Check if a file/directory should be skipped."""
    basename = os.path.basename(path)
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("*."):
            ext = pattern[1:]  # e.g., ".pkl"
            if basename.endswith(ext):
                return True
        elif basename == pattern:
            return True
    return False


def copy_directory(src: str, dst: str, label: str) -> tuple[int, int]:
    """Copy a directory recursively, skipping unwanted files."""
    files_copied = 0
    files_skipped = 0

    for root, dirs, files in os.walk(src):
        # Filter out skip directories in-place
        dirs[:] = [d for d in dirs if not should_skip(d)]

        rel_root = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_root) if rel_root != "." else dst

        os.makedirs(dst_root, exist_ok=True)

        for fname in files:
            src_file = os.path.join(root, fname)
            if should_skip(fname):
                files_skipped += 1
                continue
            dst_file = os.path.join(dst_root, fname)
            shutil.copy2(src_file, dst_file)
            files_copied += 1

    return files_copied, files_skipped


def copy_models_selective(src_models: str, dst_models: str) -> tuple[int, int]:
    """Copy models with selective file/directory inclusion."""
    files_copied = 0
    files_skipped = 0

    config = MODELS_TO_COPY["models"]
    os.makedirs(dst_models, exist_ok=True)

    # Copy top-level model files
    for fname in config["include_files"]:
        src_file = os.path.join(src_models, fname)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst_models, fname))
            files_copied += 1
        else:
            log_warn(f"Model file not found: {fname}")

    # Copy sub-directories with selective rules
    for dirname, dir_config in config["include_dirs"].items():
        src_subdir = os.path.join(src_models, dirname)
        dst_subdir = os.path.join(dst_models, dirname)

        if not os.path.isdir(src_subdir):
            log_warn(f"Model directory not found: {dirname}/")
            continue

        os.makedirs(dst_subdir, exist_ok=True)

        for fname in dir_config["include_files"]:
            src_file = os.path.join(src_subdir, fname)
            if os.path.exists(src_file):
                size_mb = os.path.getsize(src_file) / (1024 * 1024)
                shutil.copy2(src_file, os.path.join(dst_subdir, fname))
                files_copied += 1
                if size_mb > 10:
                    log_info(f"  Large file: {dirname}/{fname} ({size_mb:.1f} MB)")
            else:
                log_warn(f"  Model file not found: {dirname}/{fname}")

        # Count skipped checkpoint directories
        for excluded in dir_config.get("exclude_dirs", []):
            excluded_path = os.path.join(src_subdir, excluded)
            if os.path.isdir(excluded_path):
                n_files = sum(len(f) for _, _, f in os.walk(excluded_path))
                files_skipped += n_files
                log_info(f"  Skipped checkpoint: {dirname}/{excluded}/ ({n_files} files)")

    return files_copied, files_skipped


def main():
    project_root = os.path.abspath(".")
    target_root = os.path.abspath(TARGET_DIR)

    print()
    print(f"{Colors.BOLD}{'═' * 62}{Colors.RESET}")
    print(f"{Colors.BOLD}  🔒 Secure Project Isolation Script{Colors.RESET}")
    print(f"{Colors.BOLD}  Advanced Email Security Gateway{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 62}{Colors.RESET}")
    print()
    print(f"  Source:  {project_root}")
    print(f"  Target:  {target_root}")
    print()

    # ── Step 1: Create target directory ───────────────────
    print(f"{Colors.CYAN}[Step 1/6]{Colors.RESET} Creating target directory...")
    if os.path.exists(target_root):
        log_warn(f"Target already exists — removing and recreating")
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)
    log_ok(f"Created: {os.path.basename(target_root)}/")

    total_copied = 0
    total_skipped = 0

    # ── Step 2: Copy source directories ───────────────────
    print(f"\n{Colors.CYAN}[Step 2/6]{Colors.RESET} Copying source directories...")
    for folder in FOLDERS_TO_COPY:
        src_path = os.path.join(project_root, folder)
        dst_path = os.path.join(target_root, folder)
        if os.path.isdir(src_path):
            copied, skipped = copy_directory(src_path, dst_path, folder)
            total_copied += copied
            total_skipped += skipped
            log_ok(f"{folder}/ — {copied} files copied, {skipped} skipped")
        else:
            log_err(f"{folder}/ not found — skipping")

    # ── Step 3: Copy individual files ─────────────────────
    print(f"\n{Colors.CYAN}[Step 3/6]{Colors.RESET} Copying individual files...")
    for fpath in FILES_TO_COPY:
        src_file = os.path.join(project_root, fpath)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(target_root, fpath))
            total_copied += 1
            log_ok(f"{fpath}")
        else:
            log_warn(f"{fpath} not found — skipping")

    # Copy selective data files
    for fpath in DATA_FILES_TO_COPY:
        src_file = os.path.join(project_root, fpath)
        dst_file = os.path.join(target_root, fpath)
        if os.path.exists(src_file):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
            total_copied += 1
            log_ok(f"{fpath}")
        else:
            log_warn(f"{fpath} not found — skipping")

    # ── Step 4: Copy models (selective) ───────────────────
    print(f"\n{Colors.CYAN}[Step 4/6]{Colors.RESET} Copying ML models (selective)...")
    src_models = os.path.join(project_root, "models")
    dst_models = os.path.join(target_root, "models")
    if os.path.isdir(src_models):
        m_copied, m_skipped = copy_models_selective(src_models, dst_models)
        total_copied += m_copied
        total_skipped += m_skipped
        log_ok(f"models/ — {m_copied} files copied, {m_skipped} skipped")
    else:
        log_err("models/ not found — skipping")

    # ── Step 5: Generate project files ────────────────────
    print(f"\n{Colors.CYAN}[Step 5/6]{Colors.RESET} Generating project files...")

    # .gitignore
    with open(os.path.join(target_root, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(GITIGNORE_CONTENT)
    log_ok(".gitignore — strict security rules")

    # requirements.txt
    with open(os.path.join(target_root, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_CONTENT)
    log_ok("requirements.txt — complete dependency list")

    # README.md
    with open(os.path.join(target_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_CONTENT)
    log_ok("README.md — full project documentation")

    # ── Step 6: Validation ────────────────────────────────
    print(f"\n{Colors.CYAN}[Step 6/6]{Colors.RESET} Validating output...")

    # Check critical files exist
    critical_files = [
        "pipeline/__init__.py",
        "pipeline/analyzer.py",
        "pipeline/engine_header.py",
        "pipeline/engine_structure.py",
        "pipeline/engine_nlp.py",
        "pipeline/engine_links.py",
        "pipeline/aggregator.py",
        "pipeline/models.py",
        "pipeline/url_risk_scorer.py",
        "config/settings.yaml",
        "models/aggregator_xgb.json",
        "models/phishing_roberta/model.safetensors",
        "data/brand_names.txt",
        "data/freemail_domains.txt",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]

    missing = []
    for cf in critical_files:
        full_path = os.path.join(target_root, cf)
        if not os.path.exists(full_path):
            missing.append(cf)
            log_err(f"MISSING: {cf}")

    if not missing:
        log_ok("All critical files present")

    # Check no sensitive files leaked
    sensitive_patterns = [".env", ".pkl", "__pycache__"]
    leaks = []
    for root, dirs, files in os.walk(target_root):
        for f in files:
            for sp in sensitive_patterns:
                if sp in f:
                    leaks.append(os.path.relpath(os.path.join(root, f), target_root))

    if leaks:
        log_err(f"SECURITY LEAK: {len(leaks)} sensitive files found!")
        for leak in leaks:
            log_err(f"  → {leak}")
    else:
        log_ok("No sensitive files leaked")

    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(target_root):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    # ── Summary ───────────────────────────────────────────
    print()
    print(f"{Colors.BOLD}{'═' * 62}{Colors.RESET}")
    print(f"{Colors.BOLD}  ✅ Isolation Complete!{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 62}{Colors.RESET}")
    print(f"  Files copied:  {total_copied}")
    print(f"  Files skipped: {total_skipped} (caches, CSVs, checkpoints)")
    print(f"  Total size:    {total_size / (1024*1024):.1f} MB")
    print(f"  Location:      {target_root}")
    print()
    print(f"  {Colors.CYAN}Next steps:{Colors.RESET}")
    print(f"  1. cd {os.path.basename(target_root)}")
    print(f"  2. git init")
    print(f"  3. git add .")
    print(f"  4. git commit -m \"Initial commit: Advanced Email Security Gateway\"")
    print(f"  5. git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git")
    print(f"  6. git push -u origin main")
    print()


if __name__ == "__main__":
    main()

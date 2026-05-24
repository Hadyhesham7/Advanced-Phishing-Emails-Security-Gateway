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
.venv\Scripts\activate
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

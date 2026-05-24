# ============================================================
# PhishGuard Pipeline — Demonstration Script
# ============================================================
# Tests the pipeline against crafted phishing email samples
# to demonstrate all 4 engines detecting real attack vectors.
# ============================================================

from pipeline.analyzer import PhishingAnalyzer


def build_sample_phishing_email() -> str:
    """
    Construct a realistic phishing email that triggers
    multiple detection engines simultaneously.

    Attack vectors embedded:
    ─────────────────────────────────────────────────────
    Engine 1 (Header):
      ✓ Display name spoofing ("PayPal Security" from attacker domain)
      ✓ Reply-To mismatch (freemail reply-to)
      ✓ Suspicious X-Mailer (PHPMailer)

    Engine 2 (Structure):
      ✓ Hidden text (display:none div with Bayesian poison words)
      ✓ Zero-width characters in "PayPal"

    Engine 3 (NLP):
      ✓ Urgency language ("suspended within 24 hours")
      ✓ Credential harvesting ("verify your account")

    Engine 4 (Links):
      ✓ Href vs display text mismatch (shows paypal.com, links to evil)
      ✓ URL shortener (bit.ly)
    ─────────────────────────────────────────────────────
    """
    return """From: "PayPal Security Team" <security@paypai-support.com>
To: victim@company.com
Subject: [URGENT] Your PayPal Account Has Been Compromised
Reply-To: paypal.helpdesk.2024@gmail.com
X-Mailer: PHPMailer 6.8.0
Date: Wed, 21 May 2025 14:30:00 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary_phish_demo"

--boundary_phish_demo
Content-Type: text/plain; charset="UTF-8"

Dear Valued Customer,

We have detected unauthorized access to your PayPal account. Your account will be suspended within 24 hours unless you verify your identity immediately.

Please click the link below to confirm your account details and restore access:

https://paypal.com/security/verify

If you do not act now, your account will be permanently locked and all funds will be frozen.

Thank you,
PayPal Security Team

--boundary_phish_demo
Content-Type: text/html; charset="UTF-8"

<html>
<head>
<style>
.hidden-poison { display: none; }
.pp-header { background-color: #003087; color: white; padding: 20px; }
.pp-button { background-color: #009cde; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; }
</style>
</head>
<body>
<div class="hidden-poison">
sunshine happiness flowers birthday congratulations wonderful
amazing beautiful perfect excellent outstanding magnificent
</div>
<div class="pp-header">
<img src="https://www.paypal.com/logo.png" alt="PayPal">
<h1>Account Security Alert</h1>
</div>
<div style="padding: 20px; font-family: Arial, sans-serif;">
<p>Dear Valued Customer,</p>
<p>We have detected <strong>unauthorized access</strong> to your
P\u200Ba\u200By\u200BP\u200Ba\u200Bl account.
Your account will be <span style="color: red; font-weight: bold;">
suspended within 24 hours</span> unless you verify your identity
immediately.</p>
<p>Please click the button below to confirm your account details
and restore full access to your account:</p>
<p style="text-align: center; margin: 30px 0;">
<a href="http://paypai-security-verify.com/login/confirm.php?id=8a3f2b"
   class="pp-button"
   style="background-color: #009cde; color: white; padding: 12px 24px;
          text-decoration: none; border-radius: 4px;">
https://www.paypal.com/security/verify
</a>
</p>
<p>Alternatively, you can copy and paste this secure link:
<a href="https://bit.ly/3xFk2Pm">https://paypal.com/restore-account</a></p>
<p style="color: #666;">If you do not act now, your account will be
permanently locked and all funds will be frozen.</p>
<p>Thank you,<br>PayPal Security Team</p>
</div>
</body>
</html>

--boundary_phish_demo--
"""


def build_sample_clean_email() -> str:
    """
    Construct a legitimate email that should pass clean.
    """
    return """From: "John Smith" <john.smith@company.com>
To: jane.doe@company.com
Subject: Q3 Budget Review Meeting
Reply-To: john.smith@company.com
X-Mailer: Microsoft Outlook 16.0
Date: Wed, 21 May 2025 10:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"

Hi Jane,

Just a reminder that our Q3 budget review meeting is scheduled for
this Friday at 2:00 PM in Conference Room B.

Please bring the updated spreadsheet we discussed last week.

Thanks,
John
"""


def build_sample_bec_email() -> str:
    """
    Construct a Business Email Compromise (BEC) / CEO fraud email.
    
    Attack vector: Financial fraud via wire transfer request,
    impersonating the CEO with urgency and confidentiality.
    """
    return """From: "Michael Reynolds - CEO" <m.reynolds@companycorp.com>
To: sarah.finance@companycorp.com
Subject: Urgent - Confidential Wire Transfer Needed
X-Mailer: Microsoft Outlook 16.0
Date: Wed, 21 May 2025 16:45:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"

Sarah,

I need you to process an urgent wire transfer today. This is
time-sensitive and must be completed before end of business.

Please transfer $47,500 to the following account:

Bank: First National Bank
Account: 2847193650
Routing: 021000021

This is for a confidential acquisition deal. Please keep this
between us for now - do not discuss with anyone else until the
deal is finalized.

I'm in meetings all day so please just process this and
confirm via email when done.

Thanks,
Michael Reynolds
CEO, CompanyCorp
"""


def main():
    """Run the pipeline demonstration."""
    print("=" * 70)
    print("  PhishGuard — Deep Content Inspection Pipeline Demo")
    print("=" * 70)
    print()

    # Initialize analyzer (no config file = defaults)
    analyzer = PhishingAnalyzer("config/settings.yaml")

    # ── Test 1: Phishing Email ──────────────────────────────
    print("\n" + "─" * 70)
    print("  TEST 1: Multi-Vector Phishing Email")
    print("─" * 70)
    phishing_email = build_sample_phishing_email()
    verdict1 = analyzer.analyze(phishing_email)
    print(PhishingAnalyzer.generate_report(verdict1))

    # ── Test 2: Clean Email ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  TEST 2: Legitimate Business Email")
    print("─" * 70)
    clean_email = build_sample_clean_email()
    verdict2 = analyzer.analyze(clean_email)
    print(PhishingAnalyzer.generate_report(verdict2))

    # ── Test 3: BEC / CEO Fraud ─────────────────────────────
    print("\n" + "─" * 70)
    print("  TEST 3: Business Email Compromise (CEO Fraud)")
    print("─" * 70)
    bec_email = build_sample_bec_email()
    verdict3 = analyzer.analyze(bec_email)
    print(PhishingAnalyzer.generate_report(verdict3))

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for i, (name, verdict) in enumerate([
        ("Multi-Vector Phishing", verdict1),
        ("Legitimate Email", verdict2),
        ("BEC / CEO Fraud", verdict3),
    ], 1):
        emoji = "🔴" if verdict.label.value == "MALICIOUS" else "✅"
        print(
            f"  {emoji} Test {i}: {name:30s} → "
            f"Score={verdict.final_score:5.1f} | "
            f"Verdict={verdict.label.value:10s} | "
            f"Action={verdict.action.value}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()

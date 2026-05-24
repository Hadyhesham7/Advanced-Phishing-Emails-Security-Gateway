"""
Synthetic Phishing & Legitimate Email Generator v2
===================================================
Generates realistic .eml files covering ALL 5 evasion techniques:
  1. Visual vs Technical Destination Mismatch (brand/bare-domain display text)
  2. Display Name Spoofing + Freemail Reply-To
  3. Urgency + Consequence + Action NLP Triad
  4. HTML Obfuscation & Zero-Width Characters + Credential Forms
  5. Fake Infrastructure Lure (image-wrapped links + suspicious TLDs)
"""

import os
import random
from email.message import EmailMessage
from faker import Faker

fake = Faker()

OUTPUT_DIR_PHISH = os.path.join("data", "synthetic_phishing")
OUTPUT_DIR_CLEAN = os.path.join("data", "synthetic_clean")

NUM_PHISH = 2500
NUM_CLEAN = 2500

BRANDS = ["PayPal", "Microsoft", "Apple", "Google", "Amazon", "Netflix",
          "Chase", "Bank of America", "Wells Fargo", "LinkedIn",
          "Dropbox", "DocuSign", "DHL", "FedEx", "Facebook"]

BRAND_DOMAINS = {
    "PayPal": "paypal.com", "Microsoft": "microsoft.com", "Apple": "apple.com",
    "Google": "google.com", "Amazon": "amazon.com", "Netflix": "netflix.com",
    "Chase": "chase.com", "Bank of America": "bankofamerica.com",
    "Wells Fargo": "wellsfargo.com", "LinkedIn": "linkedin.com",
    "Dropbox": "dropbox.com", "DocuSign": "docusign.com",
    "DHL": "dhl.com", "FedEx": "fedex.com", "Facebook": "facebook.com",
}

FREEMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                    "aol.com", "protonmail.com", "mail.com"]

SUSPICIOUS_XMAILERS = ["PHPMailer", "Python/3.x", "SendBlaster", "swaks",
                       "Gophish", "Go-http-client", "curl/7.88"]

LEGIT_XMAILERS = ["Microsoft Outlook 16.0", "Apple Mail (2.3654.120.0)",
                   "Google Mail", "Thunderbird 115.0"]

URL_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "cutt.ly"]

SUSPICIOUS_TLDS = [".xyz", ".top", ".club", ".work", ".click", ".buzz",
                   ".icu", ".rest", ".site", ".online", ".tk", ".ml", ".ga"]

URGENCY_SUBJECTS = [
    "Action Required: Your account has been suspended",
    "Security Alert: Unusual sign-in activity detected",
    "URGENT: Verify your billing information within 24 hours",
    "Final Notice: Invoice #{id} is overdue",
    "Your password expires today — update now",
    "Important: Unauthorized access to your account",
    "Critical Security Update Required",
    "Your {brand} account will be deactivated",
    "Confirm your identity to avoid account closure",
    "⚠️ Suspicious login attempt blocked",
]

MFA_SUBJECTS = [
    "Action Required: Approve the sign-in request on your device",
    "Verify your phone number to continue",
    "Re-register your authenticator app",
    "Trust this browser — verification needed",
    "Enter your verification code to proceed",
]

OAUTH_SUBJECTS = [
    "App permission request — action needed",
    "Grant access to continue using {brand}",
    "Admin consent required for new application",
    "Connect your {brand} account to continue",
    "Review application permissions",
]

# ── Template 1: Standard Phishing with Href Mismatch (Test Case 1 + 3) ──
TEMPLATE_HREF_MISMATCH = """<html><body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<h2>{brand} Security Alert</h2>
<p>Dear Customer,</p>
<p>We detected unusual activity on your account. Failure to verify your identity
within 24 hours will result in your account being permanently suspended.</p>
<p>Please click the link below to verify your account immediately:</p>
<p><a href="{phish_url}">{display_link}</a></p>
<p>If you did not initiate this action, click the link above to secure your account now.</p>
<p>Thank you,<br>{brand} Security Team</p>
{hidden_div}
</div>
</body></html>"""

# ── Template 2: Credential Harvesting Form (Test Case 4) ──
TEMPLATE_CREDENTIAL_FORM = """<html><body>
<div style="font-family: Arial, sans-serif; max-width: 500px; margin: 40px auto;">
<img src="http://{spoof_domain}/logo.png" alt="{brand}" width="150" />
<h2>Sign in to your {brand} account</h2>
<p>Your session has expired. Please re-authenticate to continue.</p>
<form action="{phish_url}" method="POST">
  <input type="text" name="email" placeholder="Enter your email address" style="width:100%; padding:10px; margin:5px 0;"/>
  <input type="password" name="password" placeholder="Enter your password" style="width:100%; padding:10px; margin:5px 0;"/>
  <button type="submit" style="background-color: #0070ba; color: white; padding: 12px 24px; border: none; width:100%; cursor:pointer;">Sign In</button>
</form>
<p style="font-size:11px; color:#666;">This is an automated security message from {brand}.</p>
{zero_width_div}
</div>
</body></html>"""

# ── Template 3: Image-Wrapped Link (Test Case 5) ──
TEMPLATE_IMAGE_WRAPPED = """<html><body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<p>{sender_name} shared a file with you via {brand}.</p>
<a href="{phish_url}"><img src="https://cdn.{legit_domain}/file-icon.png" width="600" alt="Document Preview" /></a>
<p><strong>{file_name}</strong> — Shared securely via {brand}.</p>
<p style="font-size:12px; color:#888;">This link will expire in 24 hours. Click the preview above to view the document.</p>
</div>
</body></html>"""

# ── Template 4: MFA Bypass (Test Case 3 variant) ──
TEMPLATE_MFA_BYPASS = """<html><body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<h2>{brand} Security</h2>
<p>A new sign-in was attempted on your account from an unrecognized device.</p>
<p><strong>Action Required:</strong> Approve the notification on your phone or enter the verification code sent to your device.</p>
<p>If you did not attempt to sign in, please trust this browser by clicking below and entering your recovery codes:</p>
<p><a href="{phish_url}">Verify My Identity</a></p>
<p>Your account will be locked if you do not re-authenticate within 1 hour.</p>
<p>{brand} Security Team</p>
</div>
</body></html>"""

# ── Template 5: OAuth Consent Phishing ──
TEMPLATE_OAUTH = """<html><body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<h2>Application Access Request</h2>
<p>The application <strong>"{app_name}"</strong> needs access to your {brand} account.</p>
<p>This app requires permission to:</p>
<ul>
<li>Read your email and calendar</li>
<li>Connect your mailbox</li>
<li>Manage app permissions</li>
</ul>
<p>Sign in with {brand} to review and grant access:</p>
<p><a href="{phish_url}" style="background:#0078d4; color:white; padding:10px 20px; text-decoration:none;">Grant Access</a></p>
<p>Admin approval may be required. If you did not request this, contact your IT administrator.</p>
</div>
</body></html>"""

# ── Template 6: BEC / Financial Fraud ──
TEMPLATE_BEC = """<html><body>
<p>Hi {recipient_name},</p>
<p>I need you to process an urgent wire transfer for me today. The amount is ${amount} to the vendor below.</p>
<p><strong>New bank details:</strong><br>
Account Name: {account_name}<br>
Routing: {routing}<br>
Account: {account}</p>
<p>Please keep this confidential between us until the transfer is complete. I need this done within the hour.</p>
<p>Thanks,<br>{ceo_name}<br>CEO</p>
</body></html>"""

FAKE_APP_NAMES = [
    "SharePoint Document Viewer", "Microsoft Teams Connector",
    "Calendar Sync Pro", "Email Migration Tool", "Cloud Backup Manager",
    "File Share Access", "IT Service Portal", "HR Document Portal",
]

FAKE_FILE_NAMES = [
    "Q3_Salary_Adjustments.pdf", "Board_Meeting_Minutes.docx",
    "Invoice_2026-05-001.xlsx", "Contract_Amendment_Final.pdf",
    "Benefits_Enrollment_2026.pdf", "Annual_Performance_Review.pdf",
    "Confidential_Merger_Update.pdf", "Tax_Documents_W2.pdf",
]


def _random_phish_url(brand, style="generic"):
    """Generate a phishing URL using various obfuscation techniques."""
    choice = random.choice(["domain", "shortener", "ip", "suspicious_tld"])
    if choice == "shortener":
        return f"https://{random.choice(URL_SHORTENERS)}/{fake.pystr(6, 8)}"
    elif choice == "ip":
        return f"http://{fake.ipv4()}/secure/auth/login.php"
    elif choice == "suspicious_tld":
        tld = random.choice(SUSPICIOUS_TLDS)
        slug = brand.lower().replace(" ", "-")
        return f"http://{slug}-security{tld}/verify/login"
    else:
        return f"http://{fake.domain_name()}/login.php?ref={fake.uuid4()[:8]}"


def _random_display_link(brand):
    """Generate a display link that LOOKS legitimate (for href mismatch)."""
    legit_domain = BRAND_DOMAINS.get(brand, "example.com")
    style = random.choice(["full_url", "bare_domain", "brand_text"])
    if style == "full_url":
        return f"https://www.{legit_domain}/secure/verify"
    elif style == "bare_domain":
        return legit_domain
    else:
        return f"Verify your {brand} account"


def generate_phishing_email(index: int):
    """Generate a phishing .eml file triggering multiple engines."""
    msg = EmailMessage()
    brand = random.choice(BRANDS)
    legit_domain = BRAND_DOMAINS.get(brand, "example.com")

    # ── Engine 1: Headers ──
    attacker_domain = fake.domain_name()
    msg["From"] = f'"{brand} Support" <security@{attacker_domain}>'
    msg["To"] = fake.email()
    msg["Date"] = fake.date_time_this_year().strftime("%a, %d %b %Y %H:%M:%S -0000")

    # Freemail Reply-To (triggers freemail penalty + compound scoring)
    if random.random() > 0.25:
        freemail = random.choice(FREEMAIL_DOMAINS)
        msg["Reply-To"] = f"{fake.user_name()}@{freemail}"

    # Suspicious X-Mailer
    if random.random() > 0.35:
        msg["X-Mailer"] = random.choice(SUSPICIOUS_XMAILERS)

    # Auth failure headers (triggers auth_failure_detected)
    if random.random() > 0.5:
        msg["Authentication-Results"] = f"mx.{fake.domain_name()}; spf=fail; dkim=fail; dmarc=fail"

    # ── Select template ──
    template_choice = random.choices(
        ["href_mismatch", "credential_form", "image_wrapped",
         "mfa_bypass", "oauth", "bec"],
        weights=[25, 20, 15, 15, 15, 10],
        k=1
    )[0]

    phish_url = _random_phish_url(brand)

    if template_choice == "href_mismatch":
        msg["Subject"] = random.choice(URGENCY_SUBJECTS).format(
            id=fake.random_int(1000, 9999), brand=brand)
        hidden_div = ""
        if random.random() > 0.4:
            hidden_div = f'<div style="display:none; font-size:0;">{fake.paragraph()}</div>'
        html = TEMPLATE_HREF_MISMATCH.format(
            brand=brand, phish_url=phish_url,
            display_link=_random_display_link(brand),
            hidden_div=hidden_div)

    elif template_choice == "credential_form":
        msg["Subject"] = random.choice(URGENCY_SUBJECTS).format(
            id=fake.random_int(1000, 9999), brand=brand)
        zwc = "\u200b" * random.randint(5, 30)
        zero_width_div = f'<p style="font-size:0;">{zwc}</p>' if random.random() > 0.4 else ""
        html = TEMPLATE_CREDENTIAL_FORM.format(
            brand=brand, phish_url=phish_url,
            spoof_domain=f"{brand.lower().replace(' ', '')}-secure.com",
            zero_width_div=zero_width_div)

    elif template_choice == "image_wrapped":
        msg["Subject"] = f"{random.choice(FAKE_FILE_NAMES)} — Shared via {brand}"
        html = TEMPLATE_IMAGE_WRAPPED.format(
            brand=brand, phish_url=phish_url,
            legit_domain=legit_domain,
            sender_name=fake.name(),
            file_name=random.choice(FAKE_FILE_NAMES))

    elif template_choice == "mfa_bypass":
        msg["Subject"] = random.choice(MFA_SUBJECTS)
        html = TEMPLATE_MFA_BYPASS.format(brand=brand, phish_url=phish_url)

    elif template_choice == "oauth":
        msg["Subject"] = random.choice(OAUTH_SUBJECTS).format(brand=brand)
        html = TEMPLATE_OAUTH.format(
            brand=brand, phish_url=phish_url,
            app_name=random.choice(FAKE_APP_NAMES))

    else:  # BEC
        msg["Subject"] = random.choice([
            "Urgent payment needed", "Wire transfer request",
            "Confidential — time sensitive", "Quick favor needed"])
        html = TEMPLATE_BEC.format(
            recipient_name=fake.first_name(),
            amount=f"{random.randint(5, 95) * 1000:,}",
            account_name=fake.company(),
            routing=fake.random_int(100000000, 999999999),
            account=fake.random_int(1000000000, 9999999999),
            ceo_name=fake.name())

    plain_text = f"Action required for your {brand} account. Visit: {phish_url}"
    msg.set_content(plain_text)
    msg.add_alternative(html, subtype="html")

    # ── Macro attachment (30% of phishing emails) ──
    if random.random() > 0.70:
        macro_ext = random.choice([".docm", ".xlsm", ".doc", ".xls", ".dotm"])
        macro_filename = random.choice([
            f"Invoice_{fake.random_int(1000,9999)}{macro_ext}",
            f"Payment_Details{macro_ext}",
            f"Urgent_Review{macro_ext}",
            f"HR_Update{macro_ext}",
            f"Contract_Amendment{macro_ext}",
            f"Salary_Adjustment{macro_ext}",
        ])
        # Create fake Office file with VBA macro signatures
        if macro_ext in (".doc", ".xls"):
            # OLE2 magic bytes + VBA signature
            macro_data = (
                b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'  # OLE2 magic
                + b'\x00' * 100
                + b'_VBA_PROJECT'
                + b'\x00' * 50
                + b'Attribute VB_Name'
                + b'\x00' * 50
                + b'Auto_Open'
                + b'\x00' * 200
            )
        else:
            # OOXML: create a minimal ZIP with vbaProject.bin
            import zipfile
            from io import BytesIO
            buf = BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types></Types>')
                zf.writestr("word/vbaProject.bin", b'\x00' * 100)
            macro_data = buf.getvalue()

        msg.add_attachment(
            macro_data,
            maintype="application",
            subtype="octet-stream",
            filename=macro_filename,
        )

    filepath = os.path.join(OUTPUT_DIR_PHISH, f"phish_{index:04d}.eml")
    with open(filepath, "wb") as f:
        f.write(bytes(msg))


def generate_clean_email(index: int):
    """Generate a legitimate .eml file with clean signals."""
    msg = EmailMessage()

    sender_name = fake.name()
    sender_domain = fake.domain_name()
    sender_email = f"{sender_name.replace(' ', '.').lower()}@{sender_domain}"

    msg["Subject"] = fake.sentence(nb_words=6)
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = fake.email()
    msg["Date"] = fake.date_time_this_year().strftime("%a, %d %b %Y %H:%M:%S -0000")

    # Matching or no Reply-To
    if random.random() > 0.6:
        msg["Reply-To"] = sender_email

    # Clean X-Mailer
    msg["X-Mailer"] = random.choice(LEGIT_XMAILERS)

    # Clean auth results
    if random.random() > 0.3:
        msg["Authentication-Results"] = f"mx.{sender_domain}; spf=pass; dkim=pass; dmarc=pass"

    plain_text = fake.paragraph(nb_sentences=random.randint(3, 8))

    # Occasional normal links (consistent domain)
    html = f"<html><body><p>{plain_text}</p>"
    if random.random() > 0.4:
        url = f"https://{sender_domain}/{random.choice(['about', 'contact', 'profile', 'docs'])}"
        html += f'<p>More info: <a href="{url}">{url}</a></p>'
    html += "</body></html>"

    msg.set_content(plain_text)
    msg.add_alternative(html, subtype="html")

    filepath = os.path.join(OUTPUT_DIR_CLEAN, f"clean_{index:04d}.eml")
    with open(filepath, "wb") as f:
        f.write(bytes(msg))


def main():
    print("=" * 60)
    print("  Generating Synthetic Email Dataset v2")
    print("  Covers all 5 evasion techniques + BEC")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR_PHISH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_CLEAN, exist_ok=True)

    print(f"\nGenerating {NUM_PHISH} phishing emails...")
    for i in range(NUM_PHISH):
        generate_phishing_email(i)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{NUM_PHISH} phishing generated")

    print(f"\nGenerating {NUM_CLEAN} clean emails...")
    for i in range(NUM_CLEAN):
        generate_clean_email(i)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{NUM_CLEAN} clean generated")

    print(f"\n✅ Generation complete!")
    print(f"  Phishing: {OUTPUT_DIR_PHISH} ({NUM_PHISH} files)")
    print(f"  Clean:    {OUTPUT_DIR_CLEAN} ({NUM_CLEAN} files)")


if __name__ == "__main__":
    main()

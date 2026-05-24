"""Test a single .eml file through the full PhishGuard pipeline."""
import sys
from pipeline.analyzer import PhishingAnalyzer

def main():
    eml_path = sys.argv[1] if len(sys.argv) > 1 else "advanced_phishing_test.eml"
    
    with open(eml_path, "r", encoding="utf-8") as f:
        raw_email = f.read()

    print("=" * 70)
    print(f"  PhishGuard — Analyzing: {eml_path}")
    print("=" * 70)

    analyzer = PhishingAnalyzer("config/settings.yaml")
    verdict = analyzer.analyze(raw_email)
    print(PhishingAnalyzer.generate_report(verdict))

    # Print raw engine details for deep inspection
    print("\n" + "=" * 70)
    print("  RAW ENGINE SIGNALS (Deep Inspection)")
    print("=" * 70)

    h = verdict.header_result
    print(f"\n  [Engine 1 - Header]")
    print(f"    Display Name Spoofing:  {h.display_name_spoofing_detected}")
    print(f"    Claimed Name:           {h.display_name_claimed}")
    print(f"    Actual Domain:          {h.display_name_actual_domain}")
    print(f"    Reply-To Mismatch:      {h.reply_to_mismatch_detected}")
    print(f"    Reply-To Address:       {h.reply_to_address}")
    print(f"    Suspicious X-Mailer:    {h.xmailer_anomaly_detected}")
    print(f"    Auth Failure:           {h.auth_failure_detected}")
    print(f"    Score:                  {h.score}/25")

    s = verdict.structural_result
    print(f"\n  [Engine 2 - Structure]")
    print(f"    Hidden Text:            {s.hidden_text_detected}")
    print(f"    Hidden Text Methods:    {s.hidden_text_methods}")
    print(f"    Zero-Width Chars:       {s.zero_width_chars_detected} ({s.zero_width_char_count} found)")
    print(f"    Brand Impersonation:    {s.brand_impersonation_detected}")
    print(f"    Impersonated Brand:     {s.impersonated_brand}")
    print(f"    Credential Form:        {s.credential_form_detected}")
    print(f"    Credential Details:     {s.credential_form_details}")
    print(f"    Macro Detected:         {s.macro_detected}")
    print(f"    Score:                  {s.score}/20")

    n = verdict.nlp_result
    print(f"\n  [Engine 3 - NLP]")
    print(f"    Phishing Probability:   {n.phishing_probability:.1%}")
    print(f"    Predicted Intent:       {n.predicted_intent}")
    print(f"    Urgency Score:          {n.urgency_coercion_score}")
    print(f"    Credential Harvesting:  {n.credential_harvesting_score}")
    print(f"    Financial Fraud/BEC:    {n.financial_fraud_bec_score}")
    print(f"    OAuth Consent Score:    {n.oauth_consent_score}")
    print(f"    Trigger Phrases:        {n.trigger_phrases}")
    print(f"    Score:                  {n.score}/30")

    l = verdict.link_result
    print(f"\n  [Engine 4 - Links]")
    print(f"    Total Links:            {l.total_links_found}")
    print(f"    Href Mismatch:          {l.href_mismatch_detected}")
    print(f"    Mismatch Details:       {l.mismatched_links}")
    print(f"    URL Obfuscation:        {l.url_obfuscation_detected}")
    print(f"    Suspicious TLDs:        {l.suspicious_tlds}")
    print(f"    Image-Wrapped Links:    {l.image_wrapped_link_detected}")
    print(f"    Login URL Pattern:      {l.login_url_pattern_detected}")
    print(f"    Score:                  {l.score}/25")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

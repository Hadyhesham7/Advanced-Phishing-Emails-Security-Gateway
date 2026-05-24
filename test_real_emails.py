import sys
from pipeline.analyzer import PhishingAnalyzer

def main():
    analyzer = PhishingAnalyzer('config/settings.yaml')
    files = [
        r"C:\Users\hadyh\Desktop\hiii.eml",
        r"C:\Users\hadyh\Desktop\Account Verification Required (1).eml",
        r"C:\Users\hadyh\Desktop\yarb.eml",
        r"C:\Users\hadyh\Desktop\phishing_header.eml"
    ]
    
    for filename in files:
        print(f"\n{'='*60}\nTesting {filename}\n{'='*60}")
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                verdict = analyzer.analyze(f.read())
                print(f"Verdict: {verdict.label.value} (Score: {verdict.final_score:.1f}/100)")
                print(f"Action: {verdict.action.value}")
                print(f"Flags Triggered: {verdict.feature_vector.total_flags_triggered}")
                
                print("\n--- Feature Scores ---")
                print(f"Header Score:    {verdict.header_result.score:.1f}/25")
                print(f"Structure Score: {verdict.structural_result.score:.1f}/20")
                print(f"NLP Score:       {verdict.nlp_result.score:.1f}/30")
                print(f"Links Score:     {verdict.link_result.score:.1f}/25")
                
                print("\n--- XGBoost Raw Score ---")
                print(f"Phishing Probability: {verdict.confidence:.2%}")
                
                # Check what Engine 4 found:
                print(f"\nEngine 4 found URLs: {verdict.link_result.total_links_found}")
                for link in verdict.link_result.mismatched_links:
                    print(f"  [Mismatch] {link}")
                for url in verdict.link_result.obfuscated_urls:
                    print(f"  [Obfuscated] {url}")
                for url in verdict.link_result.shortener_urls:
                    print(f"  [Shortener] {url}")
        except Exception as e:
            print(f"Error testing {filename}: {e}")

if __name__ == "__main__":
    main()

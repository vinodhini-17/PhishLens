"""
PhishLens - Flask Backend API with Hybrid Architecture & SHAP Explainability
app.py

Exposes:
- POST /predict: Accepts JSON {"url": "..."}
  Checks Trusted Domains Allowlist first; if matched, returns Safe (100% confidence) immediately.
  Otherwise extracts the 14 lexical features, runs the Random Forest model, and returns
  top 3 SHAP risk factors.
- GET /health: Healthcheck endpoint.
"""

import os
import re
import pickle
import urllib.parse
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Chrome extension and web clients

MODEL_PATH = os.path.join("model", "model.pkl")

# ---------------------------------------------------------
# Load Model & Initialize SHAP TreeExplainer
# ---------------------------------------------------------
print("[*] Loading trained model from '{}'...".format(MODEL_PATH))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("[+] Model loaded successfully!")

print("[*] Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("[+] SHAP TreeExplainer initialized and ready for real-time inference.")

# ---------------------------------------------------------
# Hybrid Architecture: Trusted Allowlist (Exactly 100 Domains)
# ---------------------------------------------------------
TRUSTED_DOMAINS = {
    # Global Tech, Cloud, & Developer Platforms (35)
    "google.com", "youtube.com", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "wikipedia.org", "linkedin.com", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "reddit.com", "netflix.com", "spotify.com",
    "yahoo.com", "bing.com", "duckduckgo.com", "cloudflare.com", "stackoverflow.com",
    "openai.com", "twitch.tv", "zoom.us", "dropbox.com", "medium.com",
    "quora.com", "pinterest.com", "wordpress.org", "adobe.com", "stripe.com",
    "paypal.com", "gitlab.com", "bitbucket.org", "kaggle.com", "huggingface.co",

    # Global Higher Ed, Standards, & Learning (15)
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "cam.ac.uk",
    "ox.ac.uk", "coursera.org", "edx.org", "khanacademy.org", "udemy.com",
    "w3schools.com", "geeksforgeeks.org", "codecademy.com", "freecodecamp.org", "developer.mozilla.org",

    # Popular Indian Education & EdTech Portals (20)
    "drishtiias.com", "todaysprint.com", "gktoday.in", "unacademy.com", "byjus.com",
    "pw.live", "physicswallah.live", "vedantu.com", "testbook.com", "adda247.com",
    "shiksha.com", "jagranjosh.com", "collegedunia.com", "embibe.com", "toppr.com",
    "javatpoint.com", "careers360.com", "studycafe.in", "allen.ac.in", "aglasem.com",

    # Indian News, Media & Portals (15)
    "ndtv.com", "indiatimes.com", "timesofindia.indiatimes.com", "thehindu.com", "indianexpress.com",
    "hindustantimes.com", "livemint.com", "moneycontrol.com", "news18.com", "zee-news.india.com",
    "amarujala.com", "jagran.com", "bhaskar.com", "firstpost.com", "thewire.in",

    # Indian Public Services, Utilities & Top Platforms (15)
    "upsc.gov.in", "nta.ac.in", "ssc.nic.in", "cbse.gov.in", "ncert.nic.in",
    "india.gov.in", "irctc.co.in", "uidai.gov.in", "flipkart.com", "swiggy.com",
    "zomato.com", "paytm.com", "phonepe.com", "hotstar.com", "bookmyshow.com"
}


def extract_base_domain(url: str) -> str:
    """
    Parses an incoming URL to extract its base/root domain name.
    Strips protocol, ports, query strings, and leading 'www.'.
    Matches exact domain or verified subdomain of trusted entries.
    """
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower().strip()
        
        # Remove port if present
        if ":" in host:
            host = host.split(":")[0]

        # Strip standard 'www.' prefix
        if host.startswith("www."):
            host = host[4:]

        # Check if direct match
        if host in TRUSTED_DOMAINS:
            return host

        # Check if host is a legitimate subdomain of a trusted domain
        # (e.g. 'subdomain.drishtiias.com' -> 'drishtiias.com')
        for trusted in TRUSTED_DOMAINS:
            if host.endswith("." + trusted):
                return trusted

        return host
    except Exception:
        return ""


# ---------------------------------------------------------
# Feature Extraction Logic (Consistent with train_model.py)
# ---------------------------------------------------------
IPV4_PATTERN = re.compile(
    r'^(?:(?:http[s]?|ftp)://)?'           # optional scheme
    r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}'       # IPv4 address
    r'(?::[0-9]{1,5})?'                    # optional port
    r'(?:[/?#]|$)',
    re.IGNORECASE
)

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'banking', 'signin', 'account',
    'confirm', 'wallet', 'authenticate', 'password', 'billing',
    'webscr', 'secure-login', 'paypal', 'recovery'
]

FEATURE_COLUMNS = [
    'url_length',
    'count_at',
    'count_hyphen',
    'count_dot',
    'count_double_slash',
    'has_ip_address',
    'is_https',
    'count_digits',
    'count_slash',
    'count_question',
    'count_equal',
    'has_suspicious_keyword',
    'hostname_length',
    'count_subdomains'
]


def extract_features(url: str) -> dict:
    """Extracts the 14 lexical and structural features from a URL."""
    if not isinstance(url, str):
        url = str(url)

    url_str = url.strip()
    is_https = 1 if url_str.lower().startswith('https://') else 0

    # Normalize url by stripping scheme for consistent structural metrics
    clean_url = re.sub(r'^[a-zA-Z]+://', '', url_str)

    url_length = len(clean_url)
    count_at = clean_url.count('@')
    count_hyphen = clean_url.count('-')
    count_dot = clean_url.count('.')
    count_double_slash = clean_url.count('//')

    # Check for direct IP address in the URL
    has_ip_address = 1 if IPV4_PATTERN.search(url_str) else 0

    # Additional lexical and structural indicators
    count_digits = sum(c.isdigit() for c in clean_url)
    count_slash = clean_url.count('/')
    count_question = clean_url.count('?')
    count_equal = clean_url.count('=')

    # Suspicious keywords presence
    clean_url_lower = clean_url.lower()
    has_suspicious_keyword = 1 if any(kw in clean_url_lower for kw in SUSPICIOUS_KEYWORDS) else 0

    # Hostname parsing and subdomain analysis
    try:
        parsed = urllib.parse.urlparse(url_str if '://' in url_str else 'http://' + url_str)
        hostname = parsed.hostname or clean_url.split('/')[0]
        hostname_length = len(hostname)
        count_subdomains = max(0, hostname.count('.') - 1)
    except Exception:
        hostname = clean_url.split('/')[0] if '/' in clean_url else clean_url
        hostname_length = len(hostname)
        count_subdomains = max(0, hostname.count('.') - 1)

    return {
        'url_length': url_length,
        'count_at': count_at,
        'count_hyphen': count_hyphen,
        'count_dot': count_dot,
        'count_double_slash': count_double_slash,
        'has_ip_address': has_ip_address,
        'is_https': is_https,
        'count_digits': count_digits,
        'count_slash': count_slash,
        'count_question': count_question,
        'count_equal': count_equal,
        'has_suspicious_keyword': has_suspicious_keyword,
        'hostname_length': hostname_length,
        'count_subdomains': count_subdomains
    }


def format_risk_factor(feature_name: str, actual_val: float, shap_val: float) -> str:
    """Formats a single feature and its SHAP contribution into an intuitive explanation."""
    descriptions = {
        'count_subdomains': f"High number of subdomains ({int(actual_val)})",
        'has_ip_address': "Presence of direct IP address instead of domain",
        'has_suspicious_keyword': "Presence of deceptive keywords (e.g., login, verify, banking)",
        'hostname_length': f"Unusually long hostname ({int(actual_val)} characters)",
        'count_dot': f"Excessive dot delimiters in URL ({int(actual_val)})",
        'url_length': f"Unusually long URL length ({int(actual_val)} characters)",
        'count_hyphen': f"Deceptive hyphen count in domain/path ({int(actual_val)})",
        'count_at': "Presence of '@' symbol for destination obfuscation",
        'count_double_slash': f"Suspicious '//' redirection pattern ({int(actual_val)})",
        'is_https': "Absence of secure HTTPS protocol" if actual_val == 0 else "HTTPS protocol verified",
        'count_slash': f"Deep URL directory depth ({int(actual_val)} slashes)",
        'count_digits': f"High concentration of digits in URL ({int(actual_val)})",
        'count_question': f"Presence of query parameter anomalies ({int(actual_val)})",
        'count_equal': f"Multiple query key-value parameters ({int(actual_val)})"
    }

    desc = descriptions.get(feature_name, f"Feature '{feature_name}' (value: {actual_val})")
    return f"{desc} (SHAP: {shap_val:+.2f})"


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "PhishLens AI Phishing Detection API",
        "model_loaded": True,
        "allowlist_size": len(TRUSTED_DOMAINS),
        "features_tracked": len(FEATURE_COLUMNS)
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight CORS request
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({
            "error": "Missing 'url' field in JSON request payload."
        }), 400

    target_url = str(data["url"]).strip()
    if len(target_url) < 3:
        return jsonify({
            "error": "Provided URL is too short or invalid."
        }), 400

    try:
        # ---------------------------------------------------------
        # Hybrid Filter: Trusted Domains Allowlist Check
        # ---------------------------------------------------------
        base_domain = extract_base_domain(target_url)
        if base_domain in TRUSTED_DOMAINS:
            # Immediately bypass ML model & SHAP evaluation for verified allowlisted domains
            return jsonify({
                "url": target_url,
                "prediction": "Safe",
                "confidence_score": 100.0,
                "risk_factors": ["Verified trusted domain (Allowlist)"]
            }), 200

        # ---------------------------------------------------------
        # Standard ML & SHAP Evaluation for Non-Allowlisted Domains
        # ---------------------------------------------------------
        # 1. Extract 14 lexical & structural features
        features_dict = extract_features(target_url)
        df_feat = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        # 2. Run model prediction and calculate probabilities
        raw_pred = int(model.predict(df_feat)[0])
        probabilities = model.predict_proba(df_feat)[0]

        # Model classes mapping: 0 = Phishing, 1 = Legitimate (Safe)
        is_safe = (raw_pred == 1)
        prediction_label = "Safe" if is_safe else "Phishing"
        confidence_score = round(float(probabilities[raw_pred]) * 100, 2)

        # 3. Calculate SHAP values via TreeExplainer
        shap_vals = explainer.shap_values(df_feat)
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            # Class 0 represents Phishing contribution
            phishing_shap_scores = shap_vals[0, :, 0]
        elif isinstance(shap_vals, list):
            phishing_shap_scores = shap_vals[0][0]
        else:
            phishing_shap_scores = shap_vals[0]

        # 4. Extract Top 3 risk factors contributing most heavily to phishing
        sorted_indices = np.argsort(phishing_shap_scores)[::-1]

        top_3_risk_factors = []
        for idx in sorted_indices[:3]:
            fname = FEATURE_COLUMNS[idx]
            actual_val = df_feat[fname].values[0]
            shap_score = float(phishing_shap_scores[idx])
            top_3_risk_factors.append(format_risk_factor(fname, actual_val, shap_score))

        # 5. Build and return structured response
        response_payload = {
            "url": target_url,
            "prediction": prediction_label,
            "confidence_score": confidence_score,
            "risk_factors": top_3_risk_factors
        }
        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to process prediction.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    print(f"[*] Starting PhishLens Flask Server (Trusted Allowlist: {len(TRUSTED_DOMAINS)} domains)...")
    print("    Running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

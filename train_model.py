"""
PhishLens - Machine Learning URL Phishing Detection
train_model.py

1. Loads the real dataset of URLs from 'dataset/real_urls.csv' (with columns 'url' and 'label',
   where 1 is legitimate and 0 is phishing).
2. Cleans the data by dropping missing values, invalid URLs, and duplicate entries.
3. Extracts normalized lexical and structural features from each URL (URL length, count of '@',
   '-', '.', '//', presence of IP address, HTTPS check, suspicious keywords, digit count,
   hostname length, subdomain count, etc.).
4. Trains a Random Forest classifier.
5. Evaluates the model with Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
6. Saves the trained model using pickle into 'model/model.pkl'.
"""

import os
import re
import pickle
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Feature Extraction Functions
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


def extract_features(url: str) -> dict:
    """
    Extracts lexical and structural features from a single URL string.
    Normalizes protocol scheme so features remain consistent regardless of
    whether the URL has 'http://', 'https://', or a bare domain.
    
    Features extracted:
    - url_length: Length of normalized URL string
    - count_at: Count of '@' symbols (used to obscure destination)
    - count_hyphen: Count of '-' symbols (common in typosquatting / phishing domains)
    - count_dot: Count of '.' symbols (excessive subdomains)
    - count_double_slash: Count of '//' occurring in the path (redirect tricks)
    - has_ip_address: 1 if the hostname is an IP address, 0 otherwise
    - is_https: 1 if URL explicitly uses HTTPS, 0 otherwise
    - count_digits: Count of numeric digits in the URL
    - count_slash: Count of single '/' characters (path depth)
    - count_question: Count of '?' characters (query parameters)
    - count_equal: Count of '=' characters
    - has_suspicious_keyword: 1 if phishing indicators appear in the URL, 0 otherwise
    - hostname_length: Length of the parsed hostname / domain
    - count_subdomains: Number of dot-separated subdomains in the hostname
    """
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


def extract_features_df(urls: pd.Series) -> pd.DataFrame:
    """Extract features for an entire pandas Series of URLs."""
    features_list = [extract_features(u) for u in urls]
    return pd.DataFrame(features_list)


# ---------------------------------------------------------
# Data Cleaning & Loading
# ---------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the loaded dataset by:
    - Checking for 'url' and 'label' columns
    - Dropping missing / NaN values
    - Stripping trailing whitespaces from URLs
    - Filtering out empty or invalid URLs
    - Removing duplicate URLs
    """
    print("\n[*] Performing data cleaning...")
    initial_len = len(df)

    # Ensure required columns are present
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset must contain 'url' and 'label' columns. Found: {list(df.columns)}")

    # Drop rows where 'url' or 'label' is NaN/null
    df = df.dropna(subset=['url', 'label']).copy()

    # Ensure types and strip whitespace
    df['url'] = df['url'].astype(str).str.strip()

    # Filter out empty or trivially short invalid URLs
    df = df[df['url'].str.len() > 3]

    # Ensure labels are binary integers (1 or 0)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    # Drop duplicate URLs
    df = df.drop_duplicates(subset=['url']).reset_index(drop=True)

    removed = initial_len - len(df)
    print(f"[+] Data cleaning complete: dropped {removed} missing/duplicate records.")
    print(f"[+] Total clean samples for training: {len(df)}")
    return df


def load_dataset(csv_path: str = "dataset/real_urls.csv") -> pd.DataFrame:
    """
    Loads dataset from 'dataset/real_urls.csv'.
    If not found, checks for alternative CSVs or runs download_data.py.
    """
    if os.path.exists(csv_path):
        print(f"[*] Loading real dataset from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        print(f"[+] Successfully loaded {len(df)} records from '{csv_path}'.")
        return clean_dataset(df)

    # Check for other CSV files in dataset/
    for alt_path in ["dataset/urls.csv", "dataset/dataset.csv"]:
        if os.path.exists(alt_path):
            print(f"[*] '{csv_path}' not found. Loading alternative '{alt_path}'...")
            df = pd.read_csv(alt_path)
            return clean_dataset(df)

    # Automatically run download_data.py if real_urls.csv doesn't exist
    print(f"[!] '{csv_path}' not found. Running download_data.py to download real dataset...")
    try:
        from download_data import download_real_dataset
        df = download_real_dataset()
        return clean_dataset(df)
    except Exception as e:
        print(f"[-] Could not run download_data automatically: {e}.")
        raise FileNotFoundError(f"Could not find dataset at '{csv_path}'. Please run 'python download_data.py' first.")


# ---------------------------------------------------------
# Model Training and Evaluation
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("         PhishLens: URL Phishing Classifier Training          ")
    print("=" * 65)

    # 1. Load real dataset
    df = load_dataset("dataset/real_urls.csv")

    print("\n[+] Dataset Distribution:")
    print(f"    - Total samples        : {len(df)}")
    print(f"    - Legitimate (label=1) : {sum(df['label'] == 1)} ({sum(df['label'] == 1) / len(df) * 100:.1f}%)")
    print(f"    - Phishing   (label=0) : {sum(df['label'] == 0)} ({sum(df['label'] == 0) / len(df) * 100:.1f}%)")

    # 2. Extract features
    print("\n[*] Extracting lexical and structural features from URLs...")
    X = extract_features_df(df['url'])
    y = df['label']

    print(f"[+] Extracted {X.shape[1]} features across {len(X)} URLs:")
    for col in X.columns:
        print(f"    - {col}")

    # Split dataset into training and test sets (stratified 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] Train set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 3. Train Classifier
    print("\n[*] Training Random Forest Classifier on real-world URL data...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[+] Model training completed successfully!")

    # 4. Evaluate the Model
    print("\n" + "=" * 65)
    print("                        MODEL EVALUATION                      ")
    print("=" * 65)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%\n")

    print("-" * 65)
    print("Confusion Matrix:")
    print(f"                    Predicted Phishing (0)   Predicted Legit (1)")
    print(f"Actual Phishing(0)         {cm[0][0]:<20}    {cm[0][1]}")
    print(f"Actual Legit(1)            {cm[1][0]:<20}    {cm[1][1]}")
    print("-" * 65)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legitimate (1)'], digits=4))

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Top Feature Importances:")
    for idx in sorted_idx[:8]:
        print(f"  - {X.columns[idx]:<25}: {importances[idx] * 100:.2f}%")

    # 5. Save the trained model using pickle into 'model/model.pkl'
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")

    print(f"\n[*] Saving trained model to '{model_path}'...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[+] Model successfully saved to '{model_path}'!")

    # 6. Real URL sample prediction tests
    print("\n" + "=" * 65)
    print("                     SAMPLE PREDICTION TEST                   ")
    print("=" * 65)
    test_samples = [
        "https://www.google.com/search?q=cybersecurity+defense",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://github.com/torvalds/linux",
        "nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/en/cgi-bin/verification/login",
        "http://192.168.1.100/secure-banking/paypal-update/login.php",
        "http://wellsfargo.com@account-verify-billing-update.ru/auth"
    ]
    for test_url in test_samples:
        feat = pd.DataFrame([extract_features(test_url)])
        pred = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0]
        result = "LEGITIMATE (1)" if pred == 1 else "PHISHING (0)"
        confidence = prob[pred] * 100
        print(f"URL: {test_url}")
        print(f" --> Prediction: {result} ({confidence:.1f}% confidence)\n")

    print("[+] All training and evaluation steps completed successfully!")


if __name__ == "__main__":
    main()

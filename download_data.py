"""
PhishLens - Real Phishing URL Dataset Downloader
download_data.py

Downloads a real, public phishing & legitimate URL dataset (e.g. from Hugging Face
'taruntiwari28/phishing-site-urls' or public GitHub raw repositories).
Saves to 'dataset/real_urls.csv' with columns:
  - 'url': URL string
  - 'label': 1 for legitimate, 0 for phishing
"""

import os
import sys
import subprocess
import pandas as pd


OUTPUT_FILE = os.path.join("dataset", "real_urls.csv")
RAW_GITHUB_DATASET_URL = (
    "https://raw.githubusercontent.com/saksham-2410/Phishing-URL-Classifier/master/phishing_site_urls.csv"
)


def try_huggingface_download() -> pd.DataFrame:
    """Attempt downloading via Hugging Face `datasets` library if installed."""
    print("[*] Checking Hugging Face 'datasets' library (taruntiwari28/phishing-site-urls)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("taruntiwari28/phishing-site-urls", split="train")
        df = ds.to_pandas()
        print(f"[+] Successfully loaded {len(df)} rows from Hugging Face!")
        return df
    except ImportError:
        print("[-] 'datasets' library is not installed (skipping HF loader).")
        return None
    except Exception as e:
        print(f"[-] Hugging Face download failed: {e}")
        return None


def download_via_curl_chunks(bad_bytes: int = 350_000, good_bytes: int = 350_000) -> pd.DataFrame:
    """
    Downloads balanced chunks from the public raw dataset via HTTP range requests.
    The first portion of phishing_site_urls.csv contains phishing ('bad') URLs,
    and the latter portion (~20MB+) contains legitimate ('good') URLs.
    """
    print("[*] Downloading authentic URLs from GitHub raw dataset mirror...")
    
    os.makedirs("dataset", exist_ok=True)
    bad_temp = os.path.join("dataset", "temp_bad.txt")
    good_temp = os.path.join("dataset", "temp_good.txt")
    
    try:
        # Range 0 to bad_bytes for phishing URLs
        print(f"    - Fetching phishing URLs chunk (bytes 0-{bad_bytes})...")
        cmd_bad = [
            "curl.exe", "-s",
            "--connect-timeout", "30",
            "--max-time", "60",
            "--retry", "3",
            "-r", f"0-{bad_bytes}",
            RAW_GITHUB_DATASET_URL,
            "-o", bad_temp
        ]
        subprocess.run(cmd_bad, check=True)

        # Range 20,000,000 to (20,000,000 + good_bytes) for legitimate URLs
        print(f"    - Fetching legitimate URLs chunk (bytes 20M-{20_000_000 + good_bytes})...")
        cmd_good = [
            "curl.exe", "-s",
            "--connect-timeout", "30",
            "--max-time", "60",
            "--retry", "3",
            "-r", f"20000000-{20000000 + good_bytes}",
            RAW_GITHUB_DATASET_URL,
            "-o", good_temp
        ]
        subprocess.run(cmd_good, check=True)

        rows = []
        # Parse bad (phishing -> label 0)
        if os.path.exists(bad_temp):
            with open(bad_temp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.lower().startswith("url,"):
                        continue
                    parts = line.rsplit(",", 1)
                    if len(parts) == 2:
                        url_val = parts[0].strip().strip('"')
                        if len(url_val) > 4:
                            rows.append({"url": url_val, "label": 0})

        # Parse good (legitimate -> label 1)
        if os.path.exists(good_temp):
            with open(good_temp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit(",", 1)
                    if len(parts) == 2:
                        url_val = parts[0].strip().strip('"')
                        if len(url_val) > 4:
                            rows.append({"url": url_val, "label": 1})

        df = pd.DataFrame(rows)
        print(f"[+] Downloaded {len(df)} authentic URLs from GitHub mirror.")
        return df

    except Exception as e:
        print(f"[-] Range download error: {e}")
        return None
    finally:
        for temp_f in [bad_temp, good_temp]:
            if os.path.exists(temp_f):
                try:
                    os.remove(temp_f)
                except Exception:
                    pass


def standardize_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to 'url' and 'label',
    converts labels to binary (1 = legitimate, 0 = phishing),
    and removes nulls, invalid formats, and duplicates.
    """
    print("\n[*] Standardizing and cleaning dataset...")
    # Find matching url and label columns
    col_map = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ['url', 'urls', 'domain', 'website', 'link']:
            col_map[col] = 'url'
        elif c_lower in ['label', 'status', 'class', 'target', 'result']:
            col_map[col] = 'label'

    df = df.rename(columns=col_map)
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Could not locate 'url' and 'label' columns. Found: {list(df.columns)}")

    # Keep only relevant columns
    df = df[['url', 'label']].copy()

    # Drop nulls
    initial_count = len(df)
    df = df.dropna(subset=['url', 'label'])

    # Ensure url is string and strip whitespace
    df['url'] = df['url'].astype(str).str.strip()
    # Filter out empty or trivially short URL strings
    df = df[df['url'].str.len() > 3]

    # Map labels to 1 (legitimate) and 0 (phishing)
    def map_label(val):
        if isinstance(val, (int, float)):
            return 1 if int(val) == 1 else 0
        v_str = str(val).strip().lower()
        if v_str in ['1', 'legitimate', 'legit', 'good', 'benign', 'safe', 'normal']:
            return 1
        elif v_str in ['0', 'phishing', 'phish', 'bad', 'malicious', 'spam']:
            return 0
        return 0

    df['label'] = df['label'].apply(map_label).astype(int)

    # Drop duplicates
    df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
    final_count = len(df)

    print(f"[+] Dropped {initial_count - final_count} invalid / duplicate rows.")
    print(f"[+] Remaining unique records: {final_count}")
    return df


def download_real_dataset() -> pd.DataFrame:
    """Main workflow to download real phishing URL dataset."""
    os.makedirs("dataset", exist_ok=True)

    df = None

    # Step 1: Try Hugging Face datasets
    df = try_huggingface_download()

    # Step 2: Try GitHub mirror with range streams
    if df is None or len(df) == 0:
        df = download_via_curl_chunks(bad_bytes=350_000, good_bytes=350_000)

    # Step 3: Fallback full curl or direct read if previous methods failed
    if df is None or len(df) == 0:
        print("[*] Trying direct pandas read_csv from GitHub raw...")
        try:
            df = pd.read_csv(RAW_GITHUB_DATASET_URL, nrows=10000)
            print(f"[+] Successfully loaded {len(df)} rows from direct URL.")
        except Exception as e:
            print(f"[-] Direct read failed: {e}")

    if df is None or len(df) == 0:
        raise RuntimeError("Failed to download real dataset from all available sources.")

    # Clean and standardize
    df_clean = standardize_and_clean_data(df)

    # Save locally to dataset/real_urls.csv
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[+] Successfully saved real dataset to '{OUTPUT_FILE}'!")
    print(f"    - Total URLs          : {len(df_clean)}")
    print(f"    - Legitimate (label=1): {sum(df_clean['label'] == 1)}")
    print(f"    - Phishing   (label=0): {sum(df_clean['label'] == 0)}")

    return df_clean


if __name__ == "__main__":
    download_real_dataset()

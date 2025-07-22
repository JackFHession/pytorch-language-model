import requests
import re
import time
import sys

GUTENBERG_IDS = [
    1342, 11, 84, 1661, 2701, 98, 74, 2600, 2542, 1232,
    345, 514, 2641, 4300, 12345, 158, 28054, 236, 1080, 120,
    46, 113, 205, 15877, 160, 2814, 19942, 408, 829, 1952,
    74, 25344, 35, 100, 34522, 768, 996, 1727, 174, 208,
    27827, 1260, 376, 1065, 6400, 17135, 1400, 798, 36, 9999
    # Add more IDs as needed
]

TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

def fetch_gutenberg_text(ebook_id):
    for suffix in (f"{ebook_id}-0.txt", f"{ebook_id}.txt"):
        url = f"https://www.gutenberg.org/files/{ebook_id}/{suffix}"
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                text = resp.text
                text = re.split(r"\*\*\* *END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", text)[0]
                text = re.split(r"^\s*\*\*\* *START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", text, flags=re.M)[-1]
                return text.strip()
        except Exception as e:
            print(f"[!] Error fetching {url}: {e}")
    print(f"[!] Failed to fetch book ID {ebook_id}")
    return ""

def print_progress(current_bytes):
    percent = (current_bytes / TARGET_BYTES) * 100
    bar_len = 40
    filled_len = int(bar_len * percent // 100)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    mb = current_bytes / (1024 * 1024)
    sys.stdout.write(f"\rProgress: [{bar}] {percent:.2f}% ({mb:.2f} MB)")
    sys.stdout.flush()

def build_100mb(output_path="100mb_gutenberg.txt"):
    total_bytes = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for gid in GUTENBERG_IDS:
            txt = fetch_gutenberg_text(gid)
            if not txt:
                continue
            encoded = txt.encode("utf-8")
            fout.write(txt)
            fout.write("\n\n")
            total_bytes += len(encoded)
            print_progress(total_bytes)
            if total_bytes >= TARGET_BYTES:
                break
            time.sleep(1)
    print(f"\n✅ Done! Written to {output_path} ({total_bytes / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    build_100mb()

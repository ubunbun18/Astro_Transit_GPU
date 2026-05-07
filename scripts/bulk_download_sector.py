import os
import re
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

def main():
    parser = argparse.ArgumentParser(description="TESS Sector Bulk Downloader")
    parser.add_argument("--sector", type=int, default=1, help="TESS Sector number (default: 1)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--threads", type=int, default=50, help="Number of parallel threads (default: 50)")
    parser.add_argument("--script", type=str, default=None, help="Local path to MAST curl script (optional)")
    args = parser.parse_args()

    # 保存先ディレクトリ
    outdir = args.outdir if args.outdir else f"data/tess_sector{args.sector}"
    os.makedirs(outdir, exist_ok=True)

    tasks = []
    pattern = re.compile(r'curl -C - -L -o (\S+) (https://\S+)')

    if args.script:
        # ローカルスクリプトを使用
        print(f"Reading tasks from local script: {args.script}")
        sh_path = args.script
    else:
        # 公式スクリプトのURLを構築
        sh_url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{args.sector}_lc.sh"
        sh_path = os.path.join(outdir, f"tesscurl_sector_{args.sector}_lc.sh")

        print(f"Downloading MAST bulk script from {sh_url}...")
        resp = requests.get(sh_url)
        resp.raise_for_status()
        with open(sh_path, "wb") as f:
            f.write(resp.content)

    # URLとファイル名の抽出
    with open(sh_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                filename, url = match.groups()
                tasks.append((url, os.path.join(outdir, filename)))

    print(f"Found {len(tasks)} files to download.")

    def download_one(task):
        url, path = task
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        subprocess.run(["curl", "-s", "-C", "-", "-L", "-o", path, url], capture_output=True)

    print(f"Starting parallel download with {args.threads} workers...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        list(tqdm(executor.map(download_one, tasks), total=len(tasks), desc=f"Sector {args.sector} Download"))

    print(f"\nDownload process finished. Files are in {outdir}")

if __name__ == "__main__":
    main()

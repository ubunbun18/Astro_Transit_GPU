import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

def main():
    # 保存先ディレクトリ
    outdir = "data/tess_sector1"
    os.makedirs(outdir, exist_ok=True)

    # 公式スクリプトのURL (TESS Sector 1 LC)
    sh_url = "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_1_lc.sh"
    sh_path = os.path.join(outdir, "tesscurl_sector_1_lc.sh")

    print(f"Downloading MAST bulk script from {sh_url}...")
    resp = requests.get(sh_url)
    resp.raise_for_status()
    with open(sh_path, "wb") as f:
        f.write(resp.content)

    # URLとファイル名の抽出
    # 形式: curl -C - -L -o tess...fits https://mast...
    pattern = re.compile(r'curl -C - -L -o (\S+) (https://\S+)')
    tasks = []
    with open(sh_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                filename, url = match.groups()
                tasks.append((url, os.path.join(outdir, filename)))

    print(f"Found {len(tasks)} files to download.")

    def download_one(task):
        url, path = task
        # 存在チェック (サイズが0でないことも確認)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        
        # curl を使用してダウンロード
        # -s: 静か, -C -: レジューム, -L: リダイレクト追従, -o: 出力先
        subprocess.run(["curl", "-s", "-C", "-", "-L", "-o", path, url], 
                       capture_output=True)

    # 並列数を増やして高速化 (50スレッド)
    print("Starting parallel download with 50 workers...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        # map をリスト化して実行を完了させる
        list(tqdm(executor.map(download_one, tasks), total=len(tasks), desc="Bulk Downloading"))

    print(f"\nDownload process finished. Files are in {outdir}")

if __name__ == "__main__":
    main()

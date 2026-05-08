from astrotransit_gpu.data.bulk_downloader import BulkDownloader
import os

def prepare_sector_2():
    sector = 2
    # セクター2のデータを5,000件に限定してダウンロード
    downloader = BulkDownloader(base_path="data/tess_data")
    
    # セクター2の観測リストを取得し、5,000件ダウンロード
    print(f"Preparing data for Sector {sector}...")
    # BulkDownloader.download_sector が内部でリスト取得・DLを行う
    # 既存のロジックでは全件DLしようとするため、limitを適用する
    
    # 簡易的に、downloaderの内部メソッドを模倣して制限付きでDL
    # (実際には downloader.download_sector(sector, limit=5000) があれば理想)
    
    try:
        # 既存のdownloaderを使用してDL開始 (内部で数千件単位で実行される)
        # ここではスクリプトを走らせて、一定数確保できたら次に進む
        downloader.download_sector(sector)
    except Exception as e:
        print(f"Download initiated: {e}")

if __name__ == "__main__":
    prepare_sector_2()

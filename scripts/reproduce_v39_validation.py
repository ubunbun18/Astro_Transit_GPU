import os
import pandas as pd
import argparse
import logging
from astrotransit_gpu.validate.official_validator import OfficialValidator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ReproductionV39")

def main():
    parser = argparse.ArgumentParser(description="Reproduce V39 Scientific Validation Results")
    parser.add_argument("--results", type=str, default="outputs/bench_219331_v39.csv", help="Path to the screening results CSV")
    parser.add_argument("--config", type=str, default="configs/validation_v39.yaml", help="Path to the validation config YAML")
    parser.add_argument("--report", type=str, default="reports/REPRODUCTION_REPORT_V39.md", help="Path to save the reproduction report")
    args = parser.parse_args()

    # 1. 前提条件のチェック
    if not os.path.exists(args.results):
        logger.error(f"Input file missing: {args.results}")
        logger.error("V39の検証を再現するには、まず全天スクリーニングを実行して結果CSVを生成する必要があります。")
        logger.error("コマンド例: python -m astrotransit_gpu screen-sector --blackwell --out " + args.results)
        return

    # 2. バリデーターの初期化
    try:
        validator = OfficialValidator(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize validator: {e}")
        return

    # 3. 検証の実行
    logger.info(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)
    
    report_data = validator.run_validation(results_df)
    summary = report_data['summary']

    # 4. レポート表示と保存
    logger.info("--- Validation Results (V39 Official) ---")
    logger.info(f"Total Targets: {summary['total_targets']:,}")
    logger.info(f"Detectable TOIs: {summary['detectable_toi']}")
    logger.info(f"Recovered TOIs: {summary['recovered_toi']}")
    logger.info(f"Completeness: {summary['completeness']:.2%}")
    logger.info(f"New Candidates: {summary['new_candidates']}")
    logger.info(f"FPR: {summary['fpr']:.2%}")

    # 保存
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# AstroTransit-GPU V39 Scientific Reproduction Report\n\n")
        f.write(f"Validated on: {pd.Timestamp.now()}\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Results Source: {args.results}\n\n")
        
        f.write("## Metrics Summary\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| Completeness (Detectable) | **{summary['completeness']:.2%}** |\n")
        f.write(f"| Recovered TOIs | {summary['recovered_toi']} / {summary['detectable_toi']} |\n")
        f.write(f"| Significant Detections | {summary['significant_sigs']:,} |\n")
        f.write(f"| New Candidates | {summary['new_candidates']:,} |\n")
        f.write(f"| False Positive Rate | {summary['fpr']:.2%} |\n\n")
        
        f.write("## Matching Parameters\n")
        for k, v in summary['config'].items():
            f.write(f"- `{k}`: {v}\n")

    logger.info(f"Reproduction report saved to {args.report}")
    
    # 期待値との照合 (簡易チェック)
    expected_rate = 0.3875 # READMEにある38.75%
    if abs(summary['completeness'] - expected_rate) < 0.05:
        logger.info("✅ 科学的完備性が期待値(約38.8%)に近い値であることを確認しました。")
    else:
        logger.warning(f"完備性が期待値({expected_rate:.1%})から乖離しています。データまたは設定を確認してください。")

if __name__ == "__main__":
    main()

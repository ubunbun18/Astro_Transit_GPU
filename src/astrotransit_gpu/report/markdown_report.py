def generate_markdown_report(results, output_path):
    """
    Generate a simple Markdown report from the search results.
    """
    md = f"# AstroTransit-GPU Search Report\n\n"
    md += f"## Summary\n\n"
    md += f"| Parameter | Value |\n"
    md += f"| --- | --- |\n"
    md += f"| Best Period | {results['best_period']:.6f} days |\n"
    md += f"| Best T0 | {results['best_t0']:.6f} |\n"
    md += f"| Best Duration | {results['best_duration']:.6f} days |\n"
    md += f"| Best Depth | {results['best_depth']:.6f} |\n"
    md += f"| SNR | {results['snr']:.2f} |\n\n"
    
    if 'match' in results:
        m = results['match']
        md += f"## Validation\n\n"
        md += f"| Metric | Result |\n"
        md += f"| --- | --- |\n"
        md += f"| Match Status | {m['is_match']} |\n"
        md += f"| Match Type | {m['match_type']} |\n"
        md += f"| Period Diff | {m['p_diff']:.6e} |\n\n"

    with open(output_path, "w") as f:
        f.write(md)
    
    print(f"Report generated: {output_path}")

"""
Generate advisor-ready three-method comparison report.

=============================================================================
PURPOSE: RESEARCH-STYLE SUMMARY FOR ADVISOR UPDATES
=============================================================================

Reads baseline, RL, and RL-with-Kalman evaluation results and generates
a markdown report with interpretable conclusions.

Key Questions Addressed:
    1. Does RL beat the greedy baseline?
    2. Does RL-with-Kalman beat direct RL?
    3. Where does Kalman filtering help most?
    4. Does tracking error correlate with interception success?
    5. Does Kalman improve robustness at challenging conditions?

Outputs:
    results/comparison/comparison_summary_all_methods.md

Usage:
    python experiments/comparison/report_all_methods.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


# Default paths
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RL_DIR = PROJECT_ROOT / "results" / "rl"
RL_KALMAN_DIR = PROJECT_ROOT / "results" / "rl_kalman"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


def load_all_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all available result files from all three methods."""
    data = {}
    
    # Overall evaluation results
    paths = {
        "baseline_eval": [
            BASELINE_DIR / "1k_episodes" / "baseline_results.csv",
            BASELINE_DIR / "baseline_results.csv",
        ],
        "rl_eval": [RL_DIR / "rl_results.csv"],
        "rl_kalman_eval": [RL_KALMAN_DIR / "rl_kalman_results.csv"],
    }
    
    for key, path_list in paths.items():
        data[key] = None
        for path in path_list:
            if path.exists():
                data[key] = pd.read_csv(path)
                break
    
    # Sweep results
    sweeps = ["defender_speed", "enemy_speed", "detection_radius", "speed_grid"]
    method_dirs = {
        "baseline": BASELINE_DIR,
        "rl": RL_DIR,
        "rl_kalman": RL_KALMAN_DIR,
    }
    
    for sweep in sweeps:
        for method, dir_path in method_dirs.items():
            key = f"{method}_{sweep}"
            path = dir_path / f"sweep_{sweep}.csv"
            data[key] = pd.read_csv(path) if path.exists() else None
    
    return data


def compute_eval_stats(df: pd.DataFrame) -> Optional[dict]:
    """Compute summary statistics from episode results."""
    if df is None or len(df) == 0:
        return None
    
    n = len(df)
    n_success = int(df["success"].sum())
    
    success_rate = n_success / n
    success_se = np.sqrt(success_rate * (1 - success_rate) / n)
    
    # Wilson score CI (more accurate for proportions)
    z = 1.96
    denominator = 1 + z**2 / n
    center = (success_rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((success_rate * (1 - success_rate) + z**2 / (4 * n)) / n) / denominator
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)
    
    # Tracking error (RL-Kalman only)
    mean_tracking_error = None
    if "mean_tracking_error" in df.columns:
        valid = df["mean_tracking_error"].dropna()
        if len(valid) > 0:
            mean_tracking_error = valid.mean()
    
    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": success_rate,
        "success_se": success_se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "fail_rate": len(df[df["outcome"].isin(["soldier_caught", "unsafe_intercept"])]) / n,
        "timeout_rate": len(df[df["outcome"] == "timeout"]) / n,
        "mean_ep_len": df["episode_length"].mean(),
        "mean_tracking_error": mean_tracking_error,
    }


def two_proportion_z_test(
    n1: int, p1: float, n2: int, p2: float
) -> Tuple[float, float, str]:
    """
    Perform two-proportion z-test.
    
    Returns:
        Tuple of (z_statistic, p_value, significance_str).
    """
    # Pooled proportion
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    if se_pooled == 0:
        return 0.0, 1.0, "n.s."
    
    z = (p2 - p1) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    
    return z, p_value, sig


def analyze_tracking_error_correlation(df: pd.DataFrame) -> Optional[dict]:
    """
    Analyze correlation between tracking error and success.
    
    Returns:
        Dictionary with correlation analysis results.
    """
    if df is None or "mean_tracking_error" not in df.columns:
        return None
    
    # Filter to episodes with valid tracking error
    valid = df.dropna(subset=["mean_tracking_error"])
    if len(valid) < 10:
        return None
    
    # Point-biserial correlation (binary success vs continuous tracking error)
    success = valid["success"].values
    tracking_error = valid["mean_tracking_error"].values
    
    # Compute correlation
    r, p_value = stats.pointbiserialr(success, tracking_error)
    
    # Compare tracking error between success and failure
    success_errors = tracking_error[success == 1]
    failure_errors = tracking_error[success == 0]
    
    if len(success_errors) > 5 and len(failure_errors) > 5:
        t_stat, t_pval = stats.ttest_ind(success_errors, failure_errors)
        mean_success = np.mean(success_errors)
        mean_failure = np.mean(failure_errors)
    else:
        t_stat, t_pval = np.nan, np.nan
        mean_success = np.mean(success_errors) if len(success_errors) > 0 else np.nan
        mean_failure = np.mean(failure_errors) if len(failure_errors) > 0 else np.nan
    
    return {
        "correlation_r": r,
        "correlation_p": p_value,
        "mean_error_success": mean_success,
        "mean_error_failure": mean_failure,
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "n_success": len(success_errors),
        "n_failure": len(failure_errors),
    }


def analyze_robustness(
    baseline_sweep: pd.DataFrame,
    rl_sweep: pd.DataFrame,
    rl_kalman_sweep: pd.DataFrame,
    x_col: str,
    challenging_condition: str,  # "low" or "high"
) -> Optional[dict]:
    """
    Analyze whether Kalman improves robustness at challenging conditions.
    
    Args:
        baseline_sweep, rl_sweep, rl_kalman_sweep: Sweep DataFrames.
        x_col: Column name for the swept parameter.
        challenging_condition: "low" for low values being challenging (e.g., detection_radius),
                              "high" for high values being challenging (e.g., enemy_speed).
    
    Returns:
        Dictionary with robustness analysis.
    """
    if any(df is None for df in [baseline_sweep, rl_sweep, rl_kalman_sweep]):
        return None
    
    # Merge all three
    merged = baseline_sweep[[x_col, "success_rate"]].copy()
    merged = merged.rename(columns={"success_rate": "baseline"})
    merged = merged.merge(
        rl_sweep[[x_col, "success_rate"]].rename(columns={"success_rate": "rl"}),
        on=x_col
    )
    merged = merged.merge(
        rl_kalman_sweep[[x_col, "success_rate"]].rename(columns={"success_rate": "rl_kalman"}),
        on=x_col
    )
    
    if len(merged) == 0:
        return None
    
    # Split into easy and challenging conditions
    median_val = merged[x_col].median()
    
    if challenging_condition == "low":
        challenging = merged[merged[x_col] <= median_val]
        easy = merged[merged[x_col] > median_val]
    else:  # "high"
        challenging = merged[merged[x_col] >= median_val]
        easy = merged[merged[x_col] < median_val]
    
    # Compute RL-Kalman advantage over RL
    kalman_advantage_challenging = (challenging["rl_kalman"] - challenging["rl"]).mean()
    kalman_advantage_easy = (easy["rl_kalman"] - easy["rl"]).mean()
    
    return {
        "parameter": x_col,
        "challenging_condition": challenging_condition,
        "kalman_advantage_challenging": kalman_advantage_challenging,
        "kalman_advantage_easy": kalman_advantage_easy,
        "improvement_ratio": kalman_advantage_challenging / kalman_advantage_easy if kalman_advantage_easy != 0 else np.nan,
        "n_challenging": len(challenging),
        "n_easy": len(easy),
    }


def format_pct(val: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{val * 100:.{decimals}f}%"


def format_ci(stats: dict) -> str:
    """Format confidence interval string."""
    return f"[{format_pct(stats['ci_95_low'])}, {format_pct(stats['ci_95_high'])}]"


def generate_report(data: Dict[str, Optional[pd.DataFrame]]) -> str:
    """Generate comprehensive markdown report."""
    lines = []
    
    # Compute statistics
    baseline_stats = compute_eval_stats(data.get("baseline_eval"))
    rl_stats = compute_eval_stats(data.get("rl_eval"))
    rl_kalman_stats = compute_eval_stats(data.get("rl_kalman_eval"))
    
    # === HEADER ===
    lines.append("# Three-Method Comparison Report")
    lines.append("")
    lines.append("## UAV Defender: Baseline vs RL vs RL-with-Kalman")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # === EXECUTIVE SUMMARY ===
    lines.append("## Executive Summary")
    lines.append("")
    
    if all([baseline_stats, rl_stats, rl_kalman_stats]):
        # RL vs Baseline
        rl_vs_baseline = rl_stats["success_rate"] - baseline_stats["success_rate"]
        z1, p1, sig1 = two_proportion_z_test(
            baseline_stats["n_episodes"], baseline_stats["success_rate"],
            rl_stats["n_episodes"], rl_stats["success_rate"]
        )
        
        # RL-Kalman vs RL
        kalman_vs_rl = rl_kalman_stats["success_rate"] - rl_stats["success_rate"]
        z2, p2, sig2 = two_proportion_z_test(
            rl_stats["n_episodes"], rl_stats["success_rate"],
            rl_kalman_stats["n_episodes"], rl_kalman_stats["success_rate"]
        )
        
        # RL-Kalman vs Baseline
        kalman_vs_baseline = rl_kalman_stats["success_rate"] - baseline_stats["success_rate"]
        z3, p3, sig3 = two_proportion_z_test(
            baseline_stats["n_episodes"], baseline_stats["success_rate"],
            rl_kalman_stats["n_episodes"], rl_kalman_stats["success_rate"]
        )
        
        lines.append("### Key Findings")
        lines.append("")
        
        # Finding 1: RL vs Baseline
        if rl_vs_baseline > 0:
            direction = "outperforms"
            magnitude = "significantly" if sig1 != "n.s." else "marginally"
        else:
            direction = "underperforms"
            magnitude = "significantly" if sig1 != "n.s." else "marginally"
        
        lines.append(f"1. **RL {direction} baseline** ({magnitude}): "
                    f"+{format_pct(rl_vs_baseline)} absolute improvement "
                    f"({format_pct(baseline_stats['success_rate'])} → {format_pct(rl_stats['success_rate'])}, "
                    f"p={p1:.4f}{sig1})")
        lines.append("")
        
        # Finding 2: RL-Kalman vs RL
        if kalman_vs_rl > 0.005:
            kalman_helps = True
            lines.append(f"2. **Kalman filtering improves RL**: "
                        f"+{format_pct(kalman_vs_rl)} additional improvement "
                        f"({format_pct(rl_stats['success_rate'])} → {format_pct(rl_kalman_stats['success_rate'])}, "
                        f"p={p2:.4f}{sig2})")
        elif kalman_vs_rl < -0.005:
            kalman_helps = False
            lines.append(f"2. **Kalman filtering degrades RL**: "
                        f"{format_pct(kalman_vs_rl)} reduction "
                        f"({format_pct(rl_stats['success_rate'])} → {format_pct(rl_kalman_stats['success_rate'])}, "
                        f"p={p2:.4f}{sig2})")
        else:
            kalman_helps = None  # Equivalent
            lines.append(f"2. **Kalman filtering shows negligible effect on RL**: "
                        f"{format_pct(kalman_vs_rl)} difference "
                        f"({format_pct(rl_stats['success_rate'])} → {format_pct(rl_kalman_stats['success_rate'])}, "
                        f"p={p2:.4f}{sig2})")
        lines.append("")
        
        # Finding 3: Overall improvement
        lines.append(f"3. **Total improvement over baseline**: "
                    f"+{format_pct(kalman_vs_baseline)} "
                    f"({format_pct(baseline_stats['success_rate'])} → {format_pct(rl_kalman_stats['success_rate'])})")
        lines.append("")
    else:
        lines.append("*Insufficient data for all three methods. See detailed sections below.*")
        lines.append("")
    
    # === RESULTS TABLE ===
    lines.append("---")
    lines.append("")
    lines.append("## Overall Performance Comparison")
    lines.append("")
    lines.append("| Method | Episodes | Success Rate | 95% CI | Failure Rate | Timeout Rate |")
    lines.append("|--------|----------|--------------|--------|--------------|--------------|")
    
    for name, stats in [("Greedy Baseline", baseline_stats), 
                        ("PPO (Direct RL)", rl_stats), 
                        ("PPO (RL + Kalman)", rl_kalman_stats)]:
        if stats:
            lines.append(f"| {name} | {stats['n_episodes']} | "
                        f"{format_pct(stats['success_rate'])} ± {format_pct(stats['success_se'])} | "
                        f"{format_ci(stats)} | "
                        f"{format_pct(stats['fail_rate'])} | "
                        f"{format_pct(stats['timeout_rate'])} |")
        else:
            lines.append(f"| {name} | — | — | — | — | — |")
    
    lines.append("")
    
    # Tracking error for RL-Kalman
    if rl_kalman_stats and rl_kalman_stats.get("mean_tracking_error"):
        lines.append(f"**RL-Kalman Mean Tracking Error:** {rl_kalman_stats['mean_tracking_error']:.4f}")
        lines.append("")
    
    # === TRACKING ERROR CORRELATION ===
    lines.append("---")
    lines.append("")
    lines.append("## Does Tracking Error Predict Failure?")
    lines.append("")
    
    correlation = analyze_tracking_error_correlation(data.get("rl_kalman_eval"))
    
    if correlation:
        r = correlation["correlation_r"]
        p = correlation["correlation_p"]
        
        lines.append(f"**Correlation Analysis** (n={correlation['n_success'] + correlation['n_failure']} episodes)")
        lines.append("")
        
        if p < 0.05:
            if r < 0:
                lines.append(f"- **Significant negative correlation**: r = {r:.3f} (p = {p:.4f})")
                lines.append("- Episodes with **lower tracking error** are **more likely to succeed**.")
            else:
                lines.append(f"- **Significant positive correlation**: r = {r:.3f} (p = {p:.4f})")
                lines.append("- Unexpectedly, higher tracking error correlates with success.")
        else:
            lines.append(f"- **No significant correlation**: r = {r:.3f} (p = {p:.4f})")
            lines.append("- Tracking error does not significantly predict success/failure.")
        
        lines.append("")
        lines.append(f"| Outcome | Mean Tracking Error | n |")
        lines.append(f"|---------|---------------------|---|")
        lines.append(f"| Success | {correlation['mean_error_success']:.4f} | {correlation['n_success']} |")
        lines.append(f"| Failure | {correlation['mean_error_failure']:.4f} | {correlation['n_failure']} |")
        
        if not np.isnan(correlation["t_pvalue"]):
            sig = "significant" if correlation["t_pvalue"] < 0.05 else "not significant"
            lines.append(f"\n*t-test: t = {correlation['t_statistic']:.2f}, p = {correlation['t_pvalue']:.4f} ({sig})*")
        lines.append("")
    else:
        lines.append("*No tracking error data available for correlation analysis.*")
        lines.append("")
    
    # === ROBUSTNESS ANALYSIS ===
    lines.append("---")
    lines.append("")
    lines.append("## Robustness Analysis: Where Does Kalman Help Most?")
    lines.append("")
    
    has_robustness_data = False
    
    # Detection radius (low = challenging)
    robustness_det = analyze_robustness(
        data.get("baseline_detection_radius"),
        data.get("rl_detection_radius"),
        data.get("rl_kalman_detection_radius"),
        x_col="detection_radius",
        challenging_condition="low"
    )
    
    if robustness_det:
        has_robustness_data = True
        lines.append("### Small Detection Radius (Delayed Detection)")
        lines.append("")
        adv_chal = robustness_det["kalman_advantage_challenging"] * 100
        adv_easy = robustness_det["kalman_advantage_easy"] * 100
        
        if adv_chal > adv_easy + 1:
            lines.append(f"- **Kalman helps more at small detection radii** (challenging conditions).")
            lines.append(f"  - Small radius: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Large radius: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        elif adv_chal < adv_easy - 1:
            lines.append(f"- **Kalman helps more at large detection radii** (contrary to hypothesis).")
            lines.append(f"  - Small radius: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Large radius: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        else:
            lines.append(f"- **Kalman advantage is similar across detection radii.**")
            lines.append(f"  - Small radius: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Large radius: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        lines.append("")
    
    # Enemy speed (high = challenging)
    robustness_enemy = analyze_robustness(
        data.get("baseline_enemy_speed"),
        data.get("rl_enemy_speed"),
        data.get("rl_kalman_enemy_speed"),
        x_col="enemy_speed",
        challenging_condition="high"
    )
    
    if robustness_enemy:
        has_robustness_data = True
        lines.append("### High Enemy Speed (Fast-Moving Target)")
        lines.append("")
        adv_chal = robustness_enemy["kalman_advantage_challenging"] * 100
        adv_easy = robustness_enemy["kalman_advantage_easy"] * 100
        
        if adv_chal > adv_easy + 1:
            lines.append(f"- **Kalman helps more at high enemy speeds** (challenging conditions).")
            lines.append(f"  - Fast enemy: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Slow enemy: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        elif adv_chal < adv_easy - 1:
            lines.append(f"- **Kalman helps more at low enemy speeds** (contrary to hypothesis).")
            lines.append(f"  - Fast enemy: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Slow enemy: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        else:
            lines.append(f"- **Kalman advantage is similar across enemy speeds.**")
            lines.append(f"  - Fast enemy: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Slow enemy: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        lines.append("")
    
    # Defender speed (low = challenging)
    robustness_def = analyze_robustness(
        data.get("baseline_defender_speed"),
        data.get("rl_defender_speed"),
        data.get("rl_kalman_defender_speed"),
        x_col="defender_speed",
        challenging_condition="low"
    )
    
    if robustness_def:
        has_robustness_data = True
        lines.append("### Low Defender Speed (Reduced Speed Advantage)")
        lines.append("")
        adv_chal = robustness_def["kalman_advantage_challenging"] * 100
        adv_easy = robustness_def["kalman_advantage_easy"] * 100
        
        if adv_chal > adv_easy + 1:
            lines.append(f"- **Kalman helps more at low defender speeds** (challenging conditions).")
            lines.append(f"  - Slow defender: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Fast defender: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        elif adv_chal < adv_easy - 1:
            lines.append(f"- **Kalman helps more at high defender speeds**.")
            lines.append(f"  - Slow defender: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Fast defender: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        else:
            lines.append(f"- **Kalman advantage is similar across defender speeds.**")
            lines.append(f"  - Slow defender: RL-Kalman +{adv_chal:.1f}pp over direct RL")
            lines.append(f"  - Fast defender: RL-Kalman +{adv_easy:.1f}pp over direct RL")
        lines.append("")
    
    if not has_robustness_data:
        lines.append("*No sweep data available for robustness analysis.*")
        lines.append("")
    
    # === INTERPRETATION ===
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation and Conclusions")
    lines.append("")
    
    if all([baseline_stats, rl_stats, rl_kalman_stats]):
        # Summarize key findings
        lines.append("### Summary")
        lines.append("")
        
        rl_beats_baseline = rl_stats["success_rate"] > baseline_stats["success_rate"] + 0.01
        kalman_beats_rl = rl_kalman_stats["success_rate"] > rl_stats["success_rate"] + 0.01
        kalman_equals_rl = abs(rl_kalman_stats["success_rate"] - rl_stats["success_rate"]) <= 0.01
        
        if rl_beats_baseline and kalman_beats_rl:
            lines.append("The experiments support a **progressive improvement** hypothesis:")
            lines.append("")
            lines.append("1. Reinforcement learning learns a policy that exceeds hand-crafted baseline performance.")
            lines.append("2. Incorporating Kalman filtering for state estimation provides additional gains.")
            lines.append("3. The RL agent benefits from filtered observations that smooth sensor noise.")
        elif rl_beats_baseline and kalman_equals_rl:
            lines.append("The experiments show **RL provides significant gains**, but Kalman filtering does not add measurable benefit:")
            lines.append("")
            lines.append("1. Direct RL successfully learns an effective pursuit policy.")
            lines.append("2. The environment's true-state observations may already be sufficient.")
            lines.append("3. Kalman filtering neither helps nor hurts in this observation regime.")
        elif rl_beats_baseline and not kalman_beats_rl:
            lines.append("**Unexpected finding**: Kalman filtering reduces RL performance:")
            lines.append("")
            lines.append("1. Direct RL outperforms both baseline and RL-Kalman.")
            lines.append("2. The Kalman filter may introduce lag or bias in state estimates.")
            lines.append("3. Consider tuning process/measurement noise parameters.")
        else:
            lines.append("Results are mixed or inconclusive. Consider:")
            lines.append("")
            lines.append("1. Increasing training timesteps for RL policies.")
            lines.append("2. Hyperparameter tuning for both RL and Kalman filter.")
            lines.append("3. Analyzing failure modes in more detail.")
        
        lines.append("")
        
        # Recommendations
        lines.append("### Recommendations for Next Steps")
        lines.append("")
        
        if kalman_beats_rl:
            lines.append("1. **Deploy RL-Kalman** as the primary policy for this environment configuration.")
            lines.append("2. **Tune Kalman parameters** (process_var, measurement_var) to further improve tracking.")
            lines.append("3. **Test generalization** with varying sensor noise levels.")
        elif kalman_equals_rl:
            lines.append("1. **Deploy direct RL** since it achieves similar performance with simpler architecture.")
            lines.append("2. **Consider Kalman** if deploying to environments with actual sensor noise.")
            lines.append("3. **Test with measurement noise** to see if Kalman becomes beneficial.")
        else:
            lines.append("1. **Deploy direct RL** which shows best performance.")
            lines.append("2. **Debug Kalman integration** - check filter tuning and observation pipeline.")
            lines.append("3. **Analyze RL-Kalman failure modes** to understand performance gap.")
        
        lines.append("")
    
    # === FOOTER ===
    lines.append("---")
    lines.append("")
    lines.append("*Report generated automatically by `experiments/comparison/report_all_methods.py`*")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate advisor-ready three-method comparison report"
    )
    parser.add_argument(
        "--output", type=str, 
        default=str(OUTPUT_DIR / "comparison_summary_all_methods.md"),
        help="Output markdown file path"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING THREE-METHOD COMPARISON REPORT")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    
    # Count available datasets
    n_available = sum(1 for v in data.values() if v is not None)
    print(f"Loaded {n_available} datasets")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(data)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    print("=" * 70)
    
    # Print preview
    print("\n--- REPORT PREVIEW (first 50 lines) ---\n")
    preview_lines = report.split("\n")[:50]
    print("\n".join(preview_lines))
    if len(report.split("\n")) > 50:
        print("\n... (truncated)")


if __name__ == "__main__":
    main()

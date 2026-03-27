"""
Three-Way Comparison: Baseline vs RL vs RL-with-Kalman.

=============================================================================
PURPOSE: COMPREHENSIVE COMPARISON OF ALL THREE METHODS
=============================================================================

Loads evaluation results from all three methods and generates side-by-side
comparison plots and summary tables.

Methods:
    1. Greedy Baseline - Direct pursuit policy
    2. PPO RL - Trained without Kalman filtering  
    3. PPO RL-Kalman - Trained with Kalman state estimation

Comparisons:
    1. Overall success rate (bar chart with confidence intervals)
    2. Success probability vs defender speed
    3. Success probability vs enemy speed
    4. Success probability vs detection radius
    5. Speed grid heatmap comparison

Outputs:
    - results/comparison/baseline_vs_rl_vs_rl_kalman_summary.csv
    - results/comparison/compare_all_overall.png
    - results/comparison/compare_all_defender_speed.png
    - results/comparison/compare_all_enemy_speed.png
    - results/comparison/compare_all_detection_radius.png
    - results/comparison/compare_all_speed_grid.png

Usage:
    python experiments/comparison/compare_all_methods.py
    python experiments/comparison/compare_all_methods.py --show-plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


# Default paths
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RL_DIR = PROJECT_ROOT / "results" / "rl"
RL_KALMAN_DIR = PROJECT_ROOT / "results" / "rl_kalman"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"

# Method styling
METHOD_STYLES = {
    "baseline": {
        "name": "Greedy Baseline",
        "color": "#2E86AB",
        "marker": "o",
        "linestyle": "-",
    },
    "rl": {
        "name": "PPO (Direct RL)",
        "color": "#9B59B6",
        "marker": "s",
        "linestyle": "--",
    },
    "rl_kalman": {
        "name": "PPO (RL + Kalman)",
        "color": "#E74C3C",
        "marker": "^",
        "linestyle": "-.",
    },
}


def compute_proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    Compute confidence interval for proportion using Wilson score.
    
    Returns:
        Tuple of (proportion, std_error, ci_lower, ci_upper).
    """
    if total == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    std_error = np.sqrt(p * (1 - p) / total) if total > 0 else 0.0
    
    return (p, std_error, ci_lower, ci_upper)


def load_evaluation_results() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load overall evaluation results for all three methods.
    
    Returns:
        Dictionary mapping method name to DataFrame (or None if not found).
    """
    results = {}
    
    # Baseline
    baseline_paths = [
        BASELINE_DIR / "1k_episodes" / "baseline_results.csv",
        BASELINE_DIR / "baseline_results.csv",
    ]
    for path in baseline_paths:
        if path.exists():
            results["baseline"] = pd.read_csv(path)
            print(f"Loaded baseline: {len(results['baseline'])} episodes from {path}")
            break
    else:
        print(f"Warning: Baseline results not found")
        results["baseline"] = None
    
    # RL
    rl_path = RL_DIR / "rl_results.csv"
    if rl_path.exists():
        results["rl"] = pd.read_csv(rl_path)
        print(f"Loaded RL: {len(results['rl'])} episodes")
    else:
        print(f"Warning: RL results not found at {rl_path}")
        results["rl"] = None
    
    # RL-Kalman
    rl_kalman_path = RL_KALMAN_DIR / "rl_kalman_results.csv"
    if rl_kalman_path.exists():
        results["rl_kalman"] = pd.read_csv(rl_kalman_path)
        print(f"Loaded RL-Kalman: {len(results['rl_kalman'])} episodes")
    else:
        print(f"Warning: RL-Kalman results not found at {rl_kalman_path}")
        results["rl_kalman"] = None
    
    return results


def compute_summary_stats(df: pd.DataFrame, method_name: str) -> Optional[dict]:
    """Compute summary statistics from episode results."""
    if df is None or len(df) == 0:
        return None
    
    n = len(df)
    n_success = df["success"].sum()
    n_fail = len(df[df["outcome"].isin(["soldier_caught", "unsafe_intercept"])])
    n_timeout = len(df[df["outcome"] == "timeout"])
    
    success_rate = n_success / n
    fail_rate = n_fail / n
    timeout_rate = n_timeout / n
    
    # Confidence intervals
    _, success_se, ci_low, ci_high = compute_proportion_ci(int(n_success), n)
    
    # Episode length
    mean_ep_len = df["episode_length"].mean()
    std_ep_len = df["episode_length"].std()
    
    # Detection time
    detected = df[df["detected"] == 1]
    valid_det = detected[detected["detection_time"] > 0]
    mean_det_time = valid_det["detection_time"].mean() if len(valid_det) > 0 else np.nan
    
    # Intercept time
    success_df = df[df["success"] == 1]
    valid_int = success_df[success_df["intercept_time"] > 0]
    mean_int_time = valid_int["intercept_time"].mean() if len(valid_int) > 0 else np.nan
    
    # Tracking error (RL-Kalman only)
    mean_tracking_error = np.nan
    if "mean_tracking_error" in df.columns:
        valid_track = df["mean_tracking_error"].dropna()
        if len(valid_track) > 0:
            mean_tracking_error = valid_track.mean()
    
    return {
        "method": method_name,
        "n_episodes": n,
        "success_rate": success_rate,
        "success_se": success_se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "fail_rate": fail_rate,
        "timeout_rate": timeout_rate,
        "mean_episode_length": mean_ep_len,
        "std_episode_length": std_ep_len,
        "mean_detection_time": mean_det_time,
        "mean_intercept_time": mean_int_time,
        "mean_tracking_error": mean_tracking_error,
    }


def create_summary_table(
    results: Dict[str, Optional[pd.DataFrame]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Create summary comparison table for all methods.
    
    Returns:
        Summary DataFrame.
    """
    rows = []
    
    for method, df in results.items():
        stats = compute_summary_stats(df, METHOD_STYLES[method]["name"])
        if stats:
            rows.append(stats)
    
    if not rows:
        print("No data available for summary table")
        return None
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = [
        "method",
        "n_episodes",
        "success_rate",
        "success_se",
        "ci_lower",
        "ci_upper",
        "fail_rate",
        "timeout_rate",
        "mean_episode_length",
        "std_episode_length",
        "mean_detection_time",
        "mean_intercept_time",
        "mean_tracking_error",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSummary table saved to: {output_path}")
    
    # Print formatted table
    print("\n" + "=" * 100)
    print("THREE-WAY COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Method':<25} {'Success':>15} {'95% CI':>18} {'Failure':>12} {'Ep Len':>10}")
    print("-" * 100)
    for _, row in df.iterrows():
        ci_str = f"[{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]"
        print(f"{row['method']:<25} "
              f"{row['success_rate']*100:>7.1f}% ±{row['success_se']*100:.1f}% "
              f"{ci_str:>18} "
              f"{row['fail_rate']*100:>11.1f}% "
              f"{row['mean_episode_length']:>10.1f}")
    print("=" * 100)
    
    return df


def plot_overall_comparison(
    results: Dict[str, Optional[pd.DataFrame]],
    output_path: Path,
    show: bool = False,
) -> None:
    """
    Create bar chart comparing overall success rates with confidence intervals.
    """
    methods = []
    success_rates = []
    ci_errors = []
    colors = []
    
    for method_key in ["baseline", "rl", "rl_kalman"]:
        df = results.get(method_key)
        if df is None or len(df) == 0:
            continue
        
        n = len(df)
        n_success = df["success"].sum()
        p, se, ci_low, ci_high = compute_proportion_ci(int(n_success), n)
        
        methods.append(METHOD_STYLES[method_key]["name"])
        success_rates.append(p * 100)
        ci_errors.append([(p - ci_low) * 100, (ci_high - p) * 100])
        colors.append(METHOD_STYLES[method_key]["color"])
    
    if not methods:
        print("No data for overall comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, success_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ci_errors = np.array(ci_errors).T
    ax.errorbar(x, success_rates, yerr=ci_errors, fmt='none', 
                capsize=8, capthick=2, ecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Overall Success Rate Comparison\n(with 95% Confidence Intervals)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def load_sweep_results(sweep_name: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load sweep results for all methods.
    
    Args:
        sweep_name: "defender_speed", "enemy_speed", "detection_radius", or "speed_grid"
    
    Returns:
        Dictionary mapping method name to DataFrame.
    """
    results = {}
    
    paths = {
        "baseline": BASELINE_DIR / f"sweep_{sweep_name}.csv",
        "rl": RL_DIR / f"sweep_{sweep_name}.csv",
        "rl_kalman": RL_KALMAN_DIR / f"sweep_{sweep_name}.csv",
    }
    
    for method, path in paths.items():
        if path.exists():
            results[method] = pd.read_csv(path)
            print(f"Loaded {method} sweep_{sweep_name}: {len(results[method])} rows")
        else:
            results[method] = None
    
    return results


def plot_1d_comparison(
    results: Dict[str, Optional[pd.DataFrame]],
    x_col: str,
    x_label: str,
    title: str,
    output_path: Path,
    show: bool = False,
    add_reference_line: Optional[float] = None,
    reference_label: str = "",
) -> None:
    """
    Create comparison plot for 1D sweep with all three methods.
    """
    # Check if any data available
    has_data = any(df is not None for df in results.values())
    if not has_data:
        print(f"No data for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for method_key in ["baseline", "rl", "rl_kalman"]:
        df = results.get(method_key)
        if df is None or x_col not in df.columns:
            continue
        
        style = METHOD_STYLES[method_key]
        
        # Try to find SE column
        se_col = None
        for col in ["success_se", "standard_error"]:
            if col in df.columns:
                se_col = col
                break
        
        if se_col and se_col in df.columns:
            ax.errorbar(
                df[x_col],
                df["success_rate"] * 100,
                yerr=df[se_col] * 100,
                fmt=f'{style["marker"]}{style["linestyle"]}',
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8,
                color=style["color"],
                ecolor=style["color"],
                alpha=0.9,
                label=style["name"]
            )
        else:
            ax.plot(
                df[x_col],
                df["success_rate"] * 100,
                f'{style["marker"]}{style["linestyle"]}',
                linewidth=2,
                markersize=8,
                color=style["color"],
                label=style["name"]
            )
    
    # Reference line
    if add_reference_line is not None:
        ax.axvline(x=add_reference_line, color='gray', linestyle=':', alpha=0.7,
                   linewidth=2, label=reference_label)
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_speed_grid_comparison(
    results: Dict[str, Optional[pd.DataFrame]],
    output_path: Path,
    show: bool = False,
) -> None:
    """
    Create 3-panel heatmap comparison for 2D speed grid sweep.
    """
    # Filter to available data
    available = {k: v for k, v in results.items() if v is not None}
    if not available:
        print("No speed grid data available")
        return
    
    # Create figure
    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    
    # Custom colormap
    colors = ['#E94F37', '#F2C14E', '#44AF69']
    cmap = LinearSegmentedColormap.from_list('success', colors)
    
    im = None
    for idx, (method_key, df) in enumerate(available.items()):
        ax = axes[idx]
        style = METHOD_STYLES[method_key]
        
        # Pivot data
        pivot = df.pivot(
            index="defender_speed",
            columns="enemy_speed",
            values="success_rate"
        ).sort_index(ascending=False)
        
        # Plot heatmap
        im = ax.imshow(
            pivot.values * 100,
            aspect='auto',
            cmap=cmap,
            vmin=0,
            vmax=100,
        )
        
        # Set tick labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.0f}' for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.0f}' for v in pivot.index])
        
        # Add value annotations
        for i, v_d in enumerate(pivot.index):
            for j, v_e in enumerate(pivot.columns):
                value = pivot.loc[v_d, v_e] * 100
                text_color = 'white' if value < 40 or value > 70 else 'black'
                ax.text(j, i, f'{value:.0f}%',
                       ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color=text_color)
        
        ax.set_xlabel('Enemy Speed (v_e)', fontsize=11)
        ax.set_ylabel('Defender Speed (v_d)', fontsize=11)
        ax.set_title(style["name"], fontsize=12, color=style["color"], fontweight='bold')
    
    # Add colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, label='Success Rate (%)', shrink=0.8, pad=0.02)
    
    plt.suptitle('Success Rate Heat Maps: Defender Speed vs Enemy Speed', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_improvement_bars(
    results: Dict[str, Optional[pd.DataFrame]],
    output_path: Path,
    show: bool = False,
) -> None:
    """
    Create bar chart showing improvement of RL methods over baseline.
    """
    baseline_df = results.get("baseline")
    if baseline_df is None:
        print("No baseline data for improvement comparison")
        return
    
    n_baseline = len(baseline_df)
    baseline_success = baseline_df["success"].sum() / n_baseline * 100
    
    improvements = []
    colors = []
    methods = []
    
    for method_key in ["rl", "rl_kalman"]:
        df = results.get(method_key)
        if df is None:
            continue
        
        method_success = df["success"].sum() / len(df) * 100
        improvement = method_success - baseline_success
        
        improvements.append(improvement)
        colors.append(METHOD_STYLES[method_key]["color"])
        methods.append(METHOD_STYLES[method_key]["name"])
    
    if not improvements:
        print("No RL data for improvement comparison")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_offset = 0.5 if height >= 0 else -0.5
        sign = '+' if imp > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{sign}{imp:.1f}%', ha='center', va=va, fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Improvement Over Baseline (percentage points)', fontsize=11)
    ax.set_title(f'Success Rate Improvement vs Greedy Baseline\n(Baseline: {baseline_success:.1f}%)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Three-way comparison: Baseline vs RL vs RL-Kalman"
    )
    parser.add_argument(
        "--show-plots", action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("THREE-WAY COMPARISON: BASELINE vs RL vs RL-KALMAN")
    print("=" * 80)
    
    # 1. Load overall evaluation results
    print("\n--- Loading Overall Evaluation Results ---")
    eval_results = load_evaluation_results()
    
    # 2. Create summary table
    print("\n--- Creating Summary Table ---")
    summary_path = output_dir / "baseline_vs_rl_vs_rl_kalman_summary.csv"
    summary_df = create_summary_table(eval_results, summary_path)
    
    # 3. Plot overall comparison
    print("\n--- Plotting Overall Comparison ---")
    plot_overall_comparison(
        eval_results,
        output_dir / "compare_all_overall.png",
        show=args.show_plots
    )
    
    # 4. Plot improvement bars
    print("\n--- Plotting Improvement Over Baseline ---")
    plot_improvement_bars(
        eval_results,
        output_dir / "compare_all_improvement.png",
        show=args.show_plots
    )
    
    # 5. Defender speed sweep
    print("\n--- Loading Defender Speed Sweep ---")
    defender_results = load_sweep_results("defender_speed")
    plot_1d_comparison(
        defender_results,
        x_col="defender_speed",
        x_label="Defender Speed (v_d)",
        title="Success Rate vs Defender Speed\n(All Methods)",
        output_path=output_dir / "compare_all_defender_speed.png",
        show=args.show_plots,
        add_reference_line=12.0,
        reference_label="Enemy speed (v_e = 12)"
    )
    
    # 6. Enemy speed sweep
    print("\n--- Loading Enemy Speed Sweep ---")
    enemy_results = load_sweep_results("enemy_speed")
    plot_1d_comparison(
        enemy_results,
        x_col="enemy_speed",
        x_label="Enemy Speed (v_e)",
        title="Success Rate vs Enemy Speed\n(All Methods)",
        output_path=output_dir / "compare_all_enemy_speed.png",
        show=args.show_plots
    )
    
    # 7. Detection radius sweep
    print("\n--- Loading Detection Radius Sweep ---")
    detection_results = load_sweep_results("detection_radius")
    plot_1d_comparison(
        detection_results,
        x_col="detection_radius",
        x_label="Detection Radius",
        title="Success Rate vs Detection Radius\n(All Methods)",
        output_path=output_dir / "compare_all_detection_radius.png",
        show=args.show_plots
    )
    
    # 8. Speed grid comparison
    print("\n--- Loading Speed Grid Sweep ---")
    grid_results = load_sweep_results("speed_grid")
    plot_speed_grid_comparison(
        grid_results,
        output_dir / "compare_all_speed_grid.png",
        show=args.show_plots
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("compare_all_*")):
        print(f"  - {f.name}")
    if summary_path.exists():
        print(f"  - {summary_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
METHOD_ALIASES = {
    "undersampled": "zero_filled",
    "zero_filled": "zero_filled",
    "baseline": "resunet",
    "resunet": "resunet",
    "resunet_dc": "resunet_data_consistency",
    "resunet_data_consistency": "resunet_data_consistency",
}
METHOD_ORDER = ("zero_filled", "resunet", "resunet_data_consistency")
PLANE_ORDER = ("sagittal", "coronal", "axial")
DEFAULT_FIGURE_FORMATS = ("png", "pdf")
WIDE_SUFFIXES = {
    "zero_filled": "undersampled",
    "resunet": "resunet",
    "resunet_data_consistency": "resunet_dc",
}
PAIR_KEY_CANDIDATES = (
    "sample_id",
    "volume_id",
    "subject_id",
    "plane",
    "slice_index",
    "sampling_ratio",
    "mask_id",
)
REQUIRED_LONG_COLUMNS = {"sample_id", "plane", "method", "psnr", "ssim"}
def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
def _normalize_plane(value: object) -> str:
    return str(value).strip().lower()
def _normalize_sampling_ratio(value: object) -> float:
    ratio = float(value)
    if ratio > 1.5:
        ratio /= 100.0
    if not (0 < ratio <= 1):
        raise ValueError(f"Invalid sampling ratio: {value!r}")
    return float(round(ratio, 6))
def _normalize_method(value: object) -> str:
    method = str(value).strip().lower()
    return METHOD_ALIASES.get(method, method)
def _finite_series(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all():
        raise ValueError(f"Column {name!r} contains NaN or infinite values.")
    return numeric
def _is_long_format(df: pd.DataFrame) -> bool:
    return {"method", "psnr", "ssim"}.issubset(df.columns)
def _derive_volume_id(row: pd.Series) -> str:
    for col in ("volume_id", "subject_id", "resolved_volume_path", "sample_id"):
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return Path(value).stem if col.endswith("path") else value
    raise ValueError("Unable to derive volume_id from available columns.")
def _canonicalize_long(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_LONG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    ratio_col = "sampling_ratio" if "sampling_ratio" in out.columns else "retain_ratio"
    if ratio_col not in out.columns:
        raise ValueError("Expected either 'sampling_ratio' or 'retain_ratio' in the input CSV.")
    out["plane"] = out["plane"].map(_normalize_plane)
    out["sampling_ratio"] = out[ratio_col].map(_normalize_sampling_ratio)
    out["retain_ratio"] = out["sampling_ratio"]
    out["method"] = out["method"].map(_normalize_method)
    out["psnr"] = _finite_series(out["psnr"], "psnr")
    out["ssim"] = _finite_series(out["ssim"], "ssim")
    for metric in ("nrmse", "mae", "hfen"):
        if metric in out.columns:
            out[metric] = _finite_series(out[metric], metric)
    if "volume_id" not in out.columns:
        out["volume_id"] = out.apply(_derive_volume_id, axis=1)
    if "mask_id" in out.columns:
        out["mask_id"] = out["mask_id"].astype(str)
    if "slice_index" in out.columns:
        out["slice_index"] = pd.to_numeric(out["slice_index"], errors="coerce")
    return out
def _canonicalize_wide(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "plane", "retain_ratio"}
    for method, suffix in WIDE_SUFFIXES.items():
        required.add(f"psnr_{suffix}")
        required.add(f"ssim_{suffix}")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required wide-format columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    optional_cols = [
        "subject_id",
        "volume_id",
        "resolved_volume_path",
        "mask_id",
        "slice_index",
        "checkpoint_name",
        "epoch",
        "split",
        "normalization_method",
        "data_range",
        "metric_policy",
        "inference_time_ms",
    ]
    for _, raw_row in df.iterrows():
        base = raw_row.to_dict()
        plane = _normalize_plane(base["plane"])
        ratio = _normalize_sampling_ratio(base["retain_ratio"])
        for method, suffix in WIDE_SUFFIXES.items():
            psnr_key = f"psnr_{suffix}"
            ssim_key = f"ssim_{suffix}"
            if pd.isna(base.get(psnr_key)) or pd.isna(base.get(ssim_key)):
                raise ValueError(
                    f"Missing metric values for sample_id={base['sample_id']!r}, method={method!r}."
                )
            row = {
                "sample_id": base["sample_id"],
                "plane": plane,
                "sampling_ratio": ratio,
                "retain_ratio": ratio,
                "method": method,
                "psnr": float(base[psnr_key]),
                "ssim": float(base[ssim_key]),
                "volume_id": base.get("volume_id") or base.get("subject_id") or base["sample_id"],
            }
            for col in optional_cols:
                if col in base and pd.notna(base[col]):
                    row[col] = base[col]
            for metric in ("nrmse", "mae", "hfen"):
                metric_key = f"{metric}_{suffix}"
                if metric_key in base and pd.notna(base[metric_key]):
                    row[metric] = float(base[metric_key])
            rows.append(row)
    return _canonicalize_long(pd.DataFrame(rows))
def load_sample_metrics(sample_metrics_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sample_metrics_csv, pd.DataFrame):
        df = sample_metrics_csv.copy()
    else:
        df = pd.read_csv(Path(sample_metrics_csv))
    return _canonicalize_long(df) if _is_long_format(df) else _canonicalize_wide(df)
def aggregate_metrics(
    sample_metrics: pd.DataFrame,
    group_cols: Sequence[str],
    metric_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    metrics = list(metric_names) if metric_names is not None else ["psnr", "ssim"]
    metrics = [metric for metric in metrics if metric in sample_metrics.columns]
    if not metrics:
        raise ValueError("No metric columns found to aggregate.")
    agg_spec: dict[str, tuple[str, object]] = {"count": ("sample_id", "count")}
    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, lambda s: s.std(ddof=1))
    out = sample_metrics.groupby(list(group_cols), dropna=False, sort=False).agg(**agg_spec).reset_index()
    if "method" in out.columns:
        out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    if "plane" in out.columns:
        out["plane"] = pd.Categorical(out["plane"], categories=PLANE_ORDER, ordered=True)
    if "sampling_ratio" in out.columns:
        out = out.sort_values([c for c in ("method", "plane", "sampling_ratio") if c in out.columns], kind="mergesort")
    return out.reset_index(drop=True)
def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "undefined"
    if pd.isna(std):
        return f"{float(mean):.4f} ± undefined"
    return f"{float(mean):.4f} ± {float(std):.4f}"
def build_report_table(pooled_aggregate: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "sampling_ratio", "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"}
    missing = required - set(pooled_aggregate.columns)
    if missing:
        raise ValueError(f"Missing pooled aggregate columns: {sorted(missing)}")
    table = pooled_aggregate.copy()
    table["Method"] = table["method"].astype(str)
    table["Sampling ratio"] = (table["sampling_ratio"] * 100).round().astype(int).astype(str) + "%"
    table["PSNR"] = [_format_mean_std(m, s) for m, s in zip(table["psnr_mean"], table["psnr_std"], strict=False)]
    table["SSIM"] = [_format_mean_std(m, s) for m, s in zip(table["ssim_mean"], table["ssim_std"], strict=False)]
    cols = [
        "Method",
        "Sampling ratio",
        "count",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "PSNR",
        "SSIM",
    ]
    cols = [c for c in cols if c in table.columns]
    return table[cols].rename(columns={"count": "Number of samples"})
def _save_figure(fig: plt.Figure, output_base: Path, formats: Sequence[str]) -> list[Path]:
    saved: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved
def _label_for_method(method: str, baseline_method: str, proposed_method: str) -> str:
    if method == baseline_method:
        return f"Baseline ({method})"
    if method == proposed_method:
        return "ResUNet + DC"
    return method
def _ratio_values(df: pd.DataFrame) -> list[float]:
    ratios = sorted(float(x) for x in df["sampling_ratio"].dropna().unique())
    if not ratios:
        raise ValueError("No sampling ratios available for plotting.")
    return ratios
def plot_metric_vs_ratio(
    aggregated: pd.DataFrame,
    metric: str,
    output_base: Path,
    baseline_method: str,
    proposed_method: str,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    title_suffix: str = "all planes",
) -> list[Path]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in aggregated.columns:
        raise ValueError(f"Missing required metric aggregate column: {mean_col}")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ratios = _ratio_values(aggregated)
    x = np.arange(len(ratios))
    for method in (baseline_method, proposed_method):
        subset = aggregated[aggregated["method"].astype(str) == method].copy()
        if subset.empty:
            raise ValueError(f"Missing method {method!r} for {metric} vs ratio plot ({title_suffix}).")
        y: list[float] = []
        yerr: list[float] = []
        for ratio in ratios:
            match = subset[np.isclose(subset["sampling_ratio"].astype(float), ratio)]
            if match.empty:
                y.append(np.nan)
                yerr.append(np.nan)
                continue
            row = match.iloc[0]
            y.append(float(row[mean_col]))
            yerr.append(float(row[std_col]) if std_col in row.index and pd.notna(row[std_col]) else 0.0)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=_label_for_method(method, baseline_method, proposed_method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(r * 100))}%" for r in ratios])
    ax.set_xlabel("Sampling ratio")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs sampling ratio ({title_suffix})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)
def _pair_for_scatter(
    sample_metrics: pd.DataFrame,
    baseline_method: str,
    proposed_method: str,
    metric: str,
) -> pd.DataFrame:
    subset = sample_metrics[sample_metrics["method"].astype(str).isin([baseline_method, proposed_method])].copy()
    if subset.empty:
        raise ValueError(f"No rows found for methods {baseline_method!r} and {proposed_method!r}.")
    key_cols = [c for c in PAIR_KEY_CANDIDATES if c in subset.columns]
    if not key_cols:
        raise ValueError("Could not derive a sample pairing key from the available columns.")
    duplicates = subset.duplicated(subset=key_cols + ["method"], keep=False)
    if duplicates.any():
        examples = subset.loc[duplicates, key_cols + ["method"]].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate sample keys detected; cannot pair scatter rows reliably. Examples: {examples}")
    pivot = subset.pivot_table(index=key_cols, columns="method", values=metric, aggfunc="first")
    if baseline_method not in pivot.columns or proposed_method not in pivot.columns:
        raise ValueError(f"Missing paired columns after pivoting for {metric!r} scatter.")
    paired = pivot.reset_index()
    missing = paired[paired[baseline_method].isna() | paired[proposed_method].isna()]
    if not missing.empty:
        warnings.warn(
            f"{len(missing)} unmatched pairs dropped before plotting {metric} scatter.",
            RuntimeWarning,
            stacklevel=2,
        )
        paired = paired.dropna(subset=[baseline_method, proposed_method])
    if len(paired) < 2:
        warnings.warn(
            f"Fewer than two paired samples available for {metric} scatter; Pearson r is undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
    return paired
def compute_pearson_r(x: Iterable[float], y: Iterable[float]) -> tuple[float | None, str, int]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    n_pairs = int(x_arr.size)
    if n_pairs < 2:
        warnings.warn("Pearson r is undefined for fewer than two valid pairs.", RuntimeWarning, stacklevel=2)
        return None, "r = undefined", n_pairs
    if np.ptp(x_arr) == 0 or np.ptp(y_arr) == 0:
        warnings.warn("Pearson r is undefined for constant arrays.", RuntimeWarning, stacklevel=2)
        return None, "r = undefined", n_pairs
    r, _ = pearsonr(x_arr, y_arr)
    if not np.isfinite(r):
        warnings.warn("Pearson r calculation returned a non-finite value.", RuntimeWarning, stacklevel=2)
        return None, "r = undefined", n_pairs
    return float(r), f"r = {float(r):.3f}", n_pairs
def plot_baseline_vs_proposed_scatter(
    sample_metrics: pd.DataFrame,
    metric: str,
    output_base: Path,
    baseline_method: str,
    proposed_method: str,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    title_suffix: str = "all planes",
) -> tuple[list[Path], dict[str, object]]:
    paired = _pair_for_scatter(sample_metrics, baseline_method, proposed_method, metric)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ratios = sorted(float(x) for x in paired["sampling_ratio"].astype(float).unique()) if "sampling_ratio" in paired.columns else [None]
    markers = ["o", "s", "^", "D", "P", "X"]
    if "sampling_ratio" in paired.columns:
        for i, ratio in enumerate(ratios):
            subset = paired[np.isclose(paired["sampling_ratio"].astype(float), ratio)]
            ax.scatter(
                subset[baseline_method],
                subset[proposed_method],
                marker=markers[i % len(markers)],
                alpha=0.85,
                label=f"{int(round(ratio * 100))}% (n={len(subset)})",
            )
    else:
        ax.scatter(paired[baseline_method], paired[proposed_method], alpha=0.85, label=f"n={len(paired)}")
    combined = paired[[baseline_method, proposed_method]].to_numpy(dtype=float, copy=False)
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    pad = 0.03 * max(1.0, hi - lo)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{_label_for_method(baseline_method, baseline_method, proposed_method)} {metric.upper()}")
    ax.set_ylabel(f"{_label_for_method(proposed_method, baseline_method, proposed_method)} {metric.upper()}")
    ax.set_title(f"{metric.upper()} baseline vs proposed ({title_suffix})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    r, text, n_pairs = compute_pearson_r(paired[baseline_method], paired[proposed_method])
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    paths = _save_figure(fig, output_base, formats)
    return paths, {"pearson_r": r, "pearson_text": text, "n_pairs": n_pairs}
def _write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
def _filter_planes(sample_metrics: pd.DataFrame, plane_filter: Sequence[str] | None) -> pd.DataFrame:
    if plane_filter is None:
        return sample_metrics
    desired = {str(plane).strip().lower() for plane in plane_filter}
    filtered = sample_metrics[sample_metrics["plane"].astype(str).isin(desired)].copy()
    if filtered.empty:
        raise ValueError(f"No samples remain after filtering by plane(s): {sorted(desired)}")
    return filtered
def _generate_scope_outputs(
    sample_metrics: pd.DataFrame,
    scope_name: str,
    output_dir: Path,
    baseline_method: str,
    proposed_method: str,
    figure_formats: Sequence[str],
    metric_names: Sequence[str] | None,
) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    aggregated = aggregate_metrics(sample_metrics, ["method", "plane", "sampling_ratio"], metric_names=metric_names)
    pooled = aggregate_metrics(sample_metrics, ["method", "sampling_ratio"], metric_names=metric_names)
    aggregate_csv = output_dir / f"aggregate_metrics_{scope_name}.csv"
    pooled_csv = output_dir / f"aggregate_metrics_all_planes_{scope_name}.csv"
    aggregated.to_csv(aggregate_csv, index=False)
    pooled.to_csv(pooled_csv, index=False)
    outputs["aggregate_metrics"] = [aggregate_csv]
    outputs["aggregate_metrics_all_planes"] = [pooled_csv]
    report_table = build_report_table(pooled)
    report_csv = output_dir / f"report_metrics_table_{scope_name}.csv"
    report_tex = output_dir / f"report_metrics_table_{scope_name}.tex"
    report_md = output_dir / f"report_metrics_table_{scope_name}.md"
    report_table.to_csv(report_csv, index=False)
    report_table.to_latex(report_tex, index=False, escape=False)
    _write_markdown_table(report_table, report_md)
    outputs["report_metrics_table_csv"] = [report_csv]
    outputs["report_metrics_table_tex"] = [report_tex]
    outputs["report_metrics_table_md"] = [report_md]
    for metric in ("psnr", "ssim"):
        base = output_dir / f"{metric}_vs_ratio_{scope_name}"
        outputs[f"{metric}_vs_ratio"] = plot_metric_vs_ratio(
            pooled,
            metric,
            base,
            baseline_method=baseline_method,
            proposed_method=proposed_method,
            formats=figure_formats,
            title_suffix=scope_name.replace("_", " "),
        )
    for metric in ("psnr", "ssim"):
        base = output_dir / f"{metric}_scatter_{scope_name}"
        paths, corr = plot_baseline_vs_proposed_scatter(
            sample_metrics,
            metric,
            base,
            baseline_method=baseline_method,
            proposed_method=proposed_method,
            formats=figure_formats,
            title_suffix=scope_name.replace("_", " "),
        )
        outputs[f"{metric}_scatter"] = paths
        (output_dir / f"{metric}_scatter_{scope_name}_pearson.json").write_text(
            json.dumps(corr, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return outputs
def generate_all_evaluation_outputs(
    sample_metrics_csv: str | Path | pd.DataFrame,
    output_dir: str | Path,
    baseline_method: str = "resunet",
    proposed_method: str = "resunet_data_consistency",
    plane_filter: Sequence[str] | None = None,
    figure_formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    metric_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    out_dir = _ensure_dir(output_dir)
    sample_metrics = load_sample_metrics(sample_metrics_csv)
    sample_metrics = _filter_planes(sample_metrics, plane_filter)
    sample_metrics = sample_metrics.sort_values(
        [c for c in ("method", "plane", "sampling_ratio", "sample_id") if c in sample_metrics.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    canonical_csv = out_dir / "test_sample_metrics.csv"
    sample_metrics.to_csv(canonical_csv, index=False)
    report_summary = {
        "n_rows": int(len(sample_metrics)),
        "planes": sorted(sample_metrics["plane"].dropna().astype(str).unique().tolist()),
        "methods": sorted(sample_metrics["method"].dropna().astype(str).unique().tolist()),
        "canonical_csv": str(canonical_csv),
    }
    (out_dir / "report_summary.json").write_text(json.dumps(report_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs: dict[str, Path] = {"test_sample_metrics_csv": canonical_csv}
    scopes = ["all_planes"] + [plane for plane in PLANE_ORDER if plane in report_summary["planes"]]
    for scope in scopes:
        scope_df = sample_metrics if scope == "all_planes" else sample_metrics[sample_metrics["plane"].astype(str) == scope].copy()
        if scope_df.empty:
            warnings.warn(f"Skipping empty plane scope {scope!r}.", RuntimeWarning, stacklevel=2)
            continue
        scope_outputs = _generate_scope_outputs(
            scope_df,
            scope,
            out_dir,
            baseline_method=baseline_method,
            proposed_method=proposed_method,
            figure_formats=figure_formats,
            metric_names=metric_names,
        )
        for key, paths in scope_outputs.items():
            # Keep the first path in the returned summary; the file naming on disk is explicit.
            outputs[f"{key}_{scope}"] = paths[0]
    print("Generated report outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return outputs
def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report plots and tables from MRI evaluation CSV files.")
    parser.add_argument("--input-csv", type=str, required=True, help="Sample-level evaluation CSV to analyze.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where report outputs will be written.")
    parser.add_argument("--baseline-method", type=str, default="resunet", help="Baseline method label.")
    parser.add_argument("--proposed-method", type=str, default="resunet_data_consistency", help="Proposed method label.")
    parser.add_argument("--plane-filter", action="append", default=None, help="Restrict to one plane; can be repeated.")
    parser.add_argument("--figure-formats", type=str, default=",".join(DEFAULT_FIGURE_FORMATS), help="Comma-separated list such as png,pdf.")
    parser.add_argument("--metrics", type=str, default=None, help="Optional comma-separated metric subset to aggregate.")
    return parser
def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    figure_formats = _parse_csv_list(args.figure_formats) or list(DEFAULT_FIGURE_FORMATS)
    metric_names = _parse_csv_list(args.metrics)
    try:
        generate_all_evaluation_outputs(
            sample_metrics_csv=args.input_csv,
            output_dir=args.output_dir,
            baseline_method=args.baseline_method,
            proposed_method=args.proposed_method,
            plane_filter=args.plane_filter,
            figure_formats=figure_formats,
            metric_names=metric_names,
        )
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}")
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

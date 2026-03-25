import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


@dataclass
class Config:
    excel_path: str = "figure/Resting feature（智能插帧）.xls"
    sheet_name: str = "智能插帧"
    out_dir: str = "figure"


def load_subject_level_df(cfg: Config) -> pd.DataFrame:
    raw = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_name, header=None)
    columns = raw.iloc[1].tolist()
    body = raw.iloc[2:].copy()
    body.columns = columns
    body = body.rename(columns={body.columns[0]: "subject_id"})

    # Keep only subject rows (the sheet also contains summary rows)
    body = body[pd.to_numeric(body["subject_id"], errors="coerce").notna()].copy()
    body["subject_id"] = body["subject_id"].astype(int)

    # Placeholder time inference:
    # first appearance of each subject -> before, second appearance -> after.
    body["visit_idx"] = body.groupby("subject_id").cumcount()
    body = body[body["visit_idx"] < 2].copy()
    body["time"] = np.where(body["visit_idx"] == 0, "Before", "After")

    # Placeholder group inference:
    # random half split with fixed seed for reproducibility.
    sorted_ids = sorted(body["subject_id"].unique().tolist())
    rng = np.random.default_rng(20260324)
    pick_n = len(sorted_ids) // 2
    memc_ids = set(rng.choice(sorted_ids, size=pick_n, replace=False).tolist())
    body["group"] = np.where(body["subject_id"].isin(memc_ids), "MEMC", "Control")

    return body


def summarize_by_group_time(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tmp = df.groupby(["group", "time"], as_index=False)[value_col].mean()
    return tmp


def make_plot_a(df: pd.DataFrame, out_path: str) -> None:
    # A: mean O1/O2 alpha relative power
    dfa = df.copy()
    dfa["O1_O2_alpha_mean"] = (
        dfa["O1-alpha波相对功率"].astype(float) + dfa["O2-alpha波相对功率"].astype(float)
    ) / 2.0
    agg = summarize_by_group_time(dfa, "O1_O2_alpha_mean")

    order = [("MEMC", "Before"), ("MEMC", "After"), ("Control", "Before"), ("Control", "After")]
    agg["x"] = agg.apply(lambda r: f"{r['group']}-{r['time']}", axis=1)
    agg["x"] = pd.Categorical(agg["x"], categories=[f"{g}-{t}" for g, t in order], ordered=True)
    agg = agg.sort_values("x")

    plt.figure(figsize=(8, 5))
    palette = ["#2f6db3", "#78a9e3", "#2f8f7b", "#8ad1bf"]
    sns.barplot(
        data=agg,
        x="x",
        y="O1_O2_alpha_mean",
        hue="x",
        palette=palette,
        legend=False,
    )
    plt.title("Resting-state O1/O2 Alpha Relative Power (Placeholder Grouping)")
    plt.xlabel("Group-Time")
    plt.ylabel("Mean O1/O2 Alpha Relative Power")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_long_band_power(df: pd.DataFrame) -> pd.DataFrame:
    bands = ["theta", "alpha", "beta", "gamma"]
    channels = ["O1", "O2", "平均电极"]
    rows: List[Dict] = []

    for _, r in df.iterrows():
        for band in bands:
            for ch in channels:
                col = f"{ch}-{band}波相对功率"
                if col in df.columns:
                    rows.append(
                        {
                            "group": r["group"],
                            "time": r["time"],
                            "band": band.capitalize(),
                            "channel": "O1-O2 Mean" if ch == "平均电极" else ch,
                            "value": float(r[col]),
                        }
                    )
    return pd.DataFrame(rows)


def make_plot_b(df: pd.DataFrame, out_path: str) -> None:
    # B: O1, O2, O1-O2 mean theta/alpha/beta/gamma relative power
    long_df = build_long_band_power(df)
    agg = (
        long_df.groupby(["group", "time", "channel", "band"], as_index=False)["value"]
        .mean()
        .copy()
    )
    agg["group_time"] = agg["group"] + "-" + agg["time"]

    g = sns.catplot(
        data=agg,
        kind="bar",
        x="band",
        y="value",
        hue="group_time",
        col="channel",
        order=["Theta", "Alpha", "Beta", "Gamma"],
        hue_order=["MEMC-Before", "MEMC-After", "Control-Before", "Control-After"],
        height=4.2,
        aspect=1.0,
        palette=["#2f6db3", "#78a9e3", "#2f8f7b", "#8ad1bf"],
        sharey=True,
    )
    g.set_axis_labels("Band", "Relative Power")
    g.set_titles("{col_name}")
    g.fig.suptitle("Band Relative Power by Channel (Placeholder Grouping)", y=1.03)
    g.tight_layout()
    g.savefig(out_path, dpi=300)
    plt.close("all")


def make_plot_c(df: pd.DataFrame, out_path: str) -> None:
    # C: O1-O2 connectivity (theta/alpha/beta/gamma)
    conn_map = {
        "Theta": "O1-O2电极theta功能连接",
        "Alpha": "O1-O2电极alpha功能连接",
        "Beta": "O1-O2电极beta功能连接",
        "Gamma": "O1-O2电极gamma功能连接",
    }
    rows: List[Dict] = []
    for _, r in df.iterrows():
        for band, col in conn_map.items():
            if col in df.columns:
                rows.append(
                    {
                        "group": r["group"],
                        "time": r["time"],
                        "band": band,
                        "value": float(r[col]),
                    }
                )
    long_df = pd.DataFrame(rows)
    agg = long_df.groupby(["group", "time", "band"], as_index=False)["value"].mean()
    agg["group_time"] = agg["group"] + "-" + agg["time"]

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=agg,
        x="band",
        y="value",
        hue="group_time",
        order=["Theta", "Alpha", "Beta", "Gamma"],
        hue_order=["MEMC-Before", "MEMC-After", "Control-Before", "Control-After"],
        palette=["#2f6db3", "#78a9e3", "#2f8f7b", "#8ad1bf"],
    )
    plt.title("O1-O2 Connectivity by Band (Placeholder Grouping)")
    plt.xlabel("Band")
    plt.ylabel("Connectivity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def make_plot_figure9_style(df: pd.DataFrame, out_path: str) -> None:
    def find_col(prefix: str) -> str:
        for c in df.columns:
            if isinstance(c, str) and c.startswith(prefix):
                return c
        raise KeyError(f"Missing column with prefix: {prefix}")

    band_map = {
        "Delta": find_col("平均电极-delta波相对功率"),
        "Theta": find_col("平均电极-theta波相对功率"),
        "Alpha": find_col("平均电极-alpha波相对功率"),
        "Beta": find_col("平均电极-beta波相对功率"),
        "Gamma": find_col("平均电极-gamma波相对功率"),
    }

    before = df[df["time"] == "Before"][["subject_id", "group"] + list(band_map.values())].copy()
    after = df[df["time"] == "After"][["subject_id", "group"] + list(band_map.values())].copy()
    merged = before.merge(after, on=["subject_id", "group"], suffixes=("_before", "_after"))

    rows: List[Dict] = []
    for _, r in merged.iterrows():
        for band, col in band_map.items():
            delta = float(r[f"{col}_after"] - r[f"{col}_before"])
            rows.append({"subject_id": r["subject_id"], "group": r["group"], "band": band, "delta": delta})
    long_df = pd.DataFrame(rows)

    summary = long_df.groupby(["group", "band"], as_index=False).agg(
        mean_delta=("delta", "mean"),
        sem=("delta", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )

    band_order = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    x = np.arange(len(band_order))
    width = 0.35

    memc = summary[summary["group"] == "MEMC"].set_index("band").reindex(band_order)
    ctrl = summary[summary["group"] == "Control"].set_index("band").reindex(band_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        memc["mean_delta"].values,
        width,
        yerr=memc["sem"].values,
        capsize=4,
        color="#E5E53A",
        edgecolor="#444444",
        label="MEMC group",
    )
    ax.bar(
        x + width / 2,
        ctrl["mean_delta"].values,
        width,
        yerr=ctrl["sem"].values,
        capsize=4,
        color="#58B85A",
        edgecolor="#444444",
        label="Control group",
    )

    # Group comparison stars per band (placeholder inference on random grouping)
    y_top = max((summary["mean_delta"] + summary["sem"]).max(), 0.05)
    y_range = max(y_top - min((summary["mean_delta"] - summary["sem"]).min(), -0.05), 0.2)
    step = y_range * 0.10

    for i, band in enumerate(band_order):
        g1 = long_df[(long_df["group"] == "MEMC") & (long_df["band"] == band)]["delta"].values
        g2 = long_df[(long_df["group"] == "Control") & (long_df["band"] == band)]["delta"].values
        p = ttest_ind(g1, g2, equal_var=False).pvalue if len(g1) > 1 and len(g2) > 1 else 1.0
        stars = p_to_stars(float(p))
        if stars != "ns":
            ymax = max(memc.loc[band, "mean_delta"] + memc.loc[band, "sem"], ctrl.loc[band, "mean_delta"] + ctrl.loc[band, "sem"])
            y = ymax + step
            ax.plot([i - width / 2, i - width / 2, i + width / 2, i + width / 2], [y - 0.01, y, y, y - 0.01], color="#333333", lw=1.2)
            ax.text(i, y + 0.005, stars, ha="center", va="bottom", fontsize=11, color="#222222")

    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(band_order)
    ax.set_ylabel("Relative power change (After - Before)")
    ax.set_xlabel("EEG frequency bands")
    ax.set_title("Figure 9-style EEG Frequency Band Changes (Placeholder Mapping)")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_subject_level_df(cfg)

    make_plot_a(df, os.path.join(cfg.out_dir, "FigureA_O1O2_alpha_placeholder.png"))
    make_plot_b(df, os.path.join(cfg.out_dir, "FigureB_band_power_placeholder.png"))
    make_plot_c(df, os.path.join(cfg.out_dir, "FigureC_connectivity_placeholder.png"))
    make_plot_figure9_style(df, os.path.join(cfg.out_dir, "Figure9_style_placeholder_random.png"))

    note_path = os.path.join(cfg.out_dir, "placeholder_assumptions.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Placeholder assumptions used for plotting:\n")
        f.write("1) For each subject, first row is Before and second row is After.\n")
        f.write("2) Random split (seed=20260324) assigns half IDs to MEMC and half to Control.\n")
        f.write("3) Replace with real group/time labels for publication figures.\n")
        f.write("4) placeholder_random.png uses average-electrode band relative power change.\n")

    id_map = (
        df[["subject_id", "group"]]
        .drop_duplicates()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    id_map.to_csv(os.path.join(cfg.out_dir, "random_group_mapping.csv"), index=False, encoding="utf-8-sig")
    print("Generated placeholder figures in figure/ .")


if __name__ == "__main__":
    main()

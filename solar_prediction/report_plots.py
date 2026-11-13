"""Report figures for sweep results. Functions return figures; callers save/show.

Inputs are the tables written by the sweep notebooks:

* test summaries from :func:`solar_prediction.sweep.summarize_test`
  (``horizon``, ``scope``, ``model``, ``test_rmse_mean``, ``test_rmse_std``),
* test forecasts from :func:`solar_prediction.sweep.runs_test_metrics`
  (``timestamp``, ``actual``, ``daylight``, baselines and one column per network).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINES = ("persistence", "seasonal_naive")
LABELS = {
    "actual": "Actual",
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal naive",
    "gru": "GRU",
    "lstm": "LSTM",
}
COLORS = {
    "actual": "black",
    "persistence": "#9e9e9e",
    "seasonal_naive": "#e6a23c",
    "gru": "#1f77b4",
    "lstm": "#2ca02c",
}
SCOPE_TITLES = {"all": "All hours", "daylight": "Daylight only"}


def _label(name):
    return LABELS.get(name, name)


def _models(frame, networks):
    present = list(frame["model"].unique()) if "model" in frame else list(frame.columns)
    return [name for name in (*BASELINES, *networks) if name in present]


def plot_rmse_by_horizon(summary: pd.DataFrame, horizons, networks=("lstm", "gru")):
    """Grouped bars: test RMSE (mean +/- seed std) per horizon, all hours vs daylight."""
    scopes = [s for s in SCOPE_TITLES if s in set(summary.scope)]
    fig, axes = plt.subplots(1, len(scopes), figsize=(7 * len(scopes), 4.5), squeeze=False)
    models = _models(summary, networks)
    width = 0.8 / len(models)
    for ax, scope in zip(axes[0], scopes):
        table = summary[summary.scope == scope].set_index(["horizon", "model"])
        x = np.arange(len(horizons))
        for i, name in enumerate(models):
            means = [table.test_rmse_mean.get((h, name), np.nan) for h in horizons]
            stds = [table.test_rmse_std.get((h, name), 0) for h in horizons]
            ax.bar(
                x + (i - (len(models) - 1) / 2) * width,
                means,
                width,
                yerr=np.nan_to_num(stds),
                capsize=2,
                color=COLORS.get(name),
                label=_label(name),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_title(f"Test RMSE by horizon: {SCOPE_TITLES[scope].lower()}")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("RMSE (GHI units)")
        ax.grid(axis="y", alpha=0.2)
    axes[0][0].legend()
    fig.tight_layout()
    return fig


def skill_table(summary: pd.DataFrame, networks=("lstm", "gru")) -> pd.DataFrame:
    """Skill = 1 - network RMSE / best baseline RMSE, per horizon and scope."""
    rows = []
    for (horizon, scope), group in summary.groupby(["horizon", "scope"], sort=False):
        by_model = group.set_index("model")
        baseline = by_model.loc[[b for b in BASELINES if b in by_model.index], "test_rmse_mean"]
        best = baseline.idxmin()
        for name in networks:
            if name not in by_model.index:
                continue
            rmse, std = by_model.loc[name, ["test_rmse_mean", "test_rmse_std"]]
            rows.append(
                {
                    "horizon": horizon,
                    "scope": scope,
                    "model": name,
                    "best_baseline": best,
                    "skill": 1 - rmse / baseline[best],
                    "skill_std": np.nan_to_num(std) / baseline[best],
                }
            )
    return pd.DataFrame(rows)


def plot_skill_by_horizon(summary: pd.DataFrame, horizons, networks=("lstm", "gru")):
    """Skill against the best naive baseline vs horizon, all hours and daylight."""
    skill = skill_table(summary, networks)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(horizons))
    # All-hours and daylight skill are often nearly equal (night errors are ~0
    # for every model), so offset and mark them differently to keep both visible.
    styles = {"all": ("-", "o", -0.04), "daylight": ("--", "s", 0.04)}
    for name in networks:
        for scope, (style, marker, offset) in styles.items():
            rows = skill[(skill.model == name) & (skill.scope == scope)].set_index("horizon")
            if rows.empty:
                continue
            values = [rows.skill.get(h, np.nan) * 100 for h in horizons]
            errors = [rows.skill_std.get(h, 0) * 100 for h in horizons]
            ax.errorbar(
                x + offset,
                values,
                yerr=errors,
                linestyle=style,
                marker=marker,
                markerfacecolor="white" if scope == "daylight" else None,
                capsize=3,
                color=COLORS.get(name),
                label=f"{_label(name)}, {SCOPE_TITLES[scope].lower()}",
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE reduction vs best baseline (%)")
    ax.set_title("Skill over the best naive baseline")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def rmse_by_hour(forecasts: pd.DataFrame, columns) -> pd.DataFrame:
    """Daylight RMSE per target hour of day (timestamp clock) for each column."""
    day = forecasts[forecasts.daylight.astype(bool)]
    hour = pd.to_datetime(day.timestamp).dt.hour
    return pd.DataFrame(
        {
            name: ((day[name] - day.actual) ** 2).groupby(hour).mean().pow(0.5)
            for name in columns
            if name in day
        }
    )


def plot_rmse_by_hour(forecasts: dict, horizons, networks=("lstm", "gru")):
    """Daylight RMSE by target hour of day for each horizon."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        frame = forecasts[horizon]
        table = rmse_by_hour(frame, _models(frame, networks))
        for name in table:
            ax.plot(
                table.index, table[name], marker=".", color=COLORS.get(name), label=_label(name)
            )
        ax.set_title(f"Daylight test RMSE by hour, {horizon} ahead")
        ax.set_xlabel("Target hour (dataset clock, UTC)")
        ax.set_ylabel("RMSE (GHI units)")
        ax.grid(alpha=0.2)
    axes[0][0].legend()
    fig.tight_layout()
    return fig


def pick_example_weeks(forecasts: pd.DataFrame, days: int = 7):
    """Start dates of a clear and a cloudy week in the test period.

    Clearness of a week = its mean daylight GHI divided by the mean of a
    clear-sky proxy (95th percentile of test GHI for the same month and time
    of day). Only weeks in the brighter half of the year are considered, so
    "cloudy" means clouds rather than winter.
    """
    frame = forecasts[["timestamp", "actual", "daylight"]].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame = frame[frame.daylight.astype(bool)]
    stamp = frame.timestamp
    key = [stamp.dt.month, stamp.dt.hour * 60 + stamp.dt.minute]
    frame["reference"] = frame.groupby(key).actual.transform(lambda v: v.quantile(0.95))
    start = stamp.dt.normalize().min()
    frame["week"] = (stamp.dt.normalize() - start).dt.days // days
    weeks = frame.groupby("week").agg(
        actual=("actual", "mean"),
        reference=("reference", "mean"),
        n_days=("timestamp", lambda t: t.dt.normalize().nunique()),
    )
    weeks = weeks[weeks.n_days == days]
    bright = weeks[weeks.reference >= weeks.reference.median()]
    if len(bright) >= 2:
        weeks = bright
    if weeks.empty:
        raise ValueError("No complete weeks in the test forecasts")
    clearness = weeks.actual / weeks.reference
    to_date = lambda week: start + pd.Timedelta(days=int(week) * days)  # noqa: E731
    return to_date(clearness.idxmax()), to_date(clearness.idxmin())


def plot_example_forecasts(
    forecasts: dict, horizons, weeks: dict, networks=("lstm", "gru"), days: int = 7
):
    """Actual vs every model over named weeks (rows) and horizons (columns).

    ``weeks`` maps a row title (e.g. "Clear week") to its start timestamp.
    """
    fig, axes = plt.subplots(
        len(weeks), len(horizons), figsize=(8 * len(horizons), 3.4 * len(weeks)), squeeze=False
    )
    for row, (title, begin) in zip(axes, weeks.items()):
        begin = pd.Timestamp(begin)
        end = begin + pd.Timedelta(days=days)
        for ax, horizon in zip(row, horizons):
            frame = forecasts[horizon].copy()
            frame["timestamp"] = pd.to_datetime(frame.timestamp)
            window = frame[(frame.timestamp >= begin) & (frame.timestamp < end)]
            for name in ("actual", *_models(frame, networks)):
                ax.plot(
                    window.timestamp,
                    window[name],
                    color=COLORS.get(name),
                    linewidth=1.6 if name == "actual" else 1.0,
                    alpha=1.0 if name in ("actual", *networks) else 0.8,
                    label=_label(name),
                )
            ax.set_title(f"{title} from {begin:%Y-%m-%d}, {horizon} ahead")
            ax.set_ylabel("GHI")
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", labelrotation=30)
    axes[0][0].legend(ncol=5, fontsize="small", loc="upper left")
    fig.tight_layout()
    return fig


def plot_stage2(summary: pd.DataFrame, stage1_winners: dict):
    """Validation RMSE (mean +/- seed std) of stage-2 configs per horizon.

    ``stage1_winners`` maps horizon -> stage-1 winning run name, drawn as a line.
    """
    horizons = list(stage1_winners)
    fig, axes = plt.subplots(1, len(horizons), figsize=(7 * len(horizons), 4.5), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        rows = summary[summary.horizon == horizon].sort_values("val_rmse_mean")
        labels = [
            f"h{r.hidden_dim} l{r.num_layers} lr{r.learning_rate:g}" for r in rows.itertuples()
        ]
        colors = [
            "#1f77b4" if name == stage1_winners[horizon] else "#9ecae1" for name in rows.run_name
        ]
        ax.bar(
            labels,
            rows.val_rmse_mean,
            yerr=rows.val_rmse_std.fillna(0),
            capsize=3,
            color=colors,
        )
        base = summary[summary.run_name == stage1_winners[horizon]]
        base = base[base.horizon == horizon].val_rmse_mean
        if not base.empty:
            ax.axhline(base.iloc[0], color="black", linestyle="--", label="stage-1 winner")
            ax.legend()
        spread = rows.val_rmse_std.fillna(0).max()
        ax.set_ylim(
            rows.val_rmse_mean.min() - 3 * spread - 0.1, rows.val_rmse_mean.max() + 3 * spread + 0.1
        )
        ax.set_title(f"Stage 2 validation RMSE, {horizon} (mean +/- std over seeds)")
        ax.set_ylabel("RMSE (GHI units)")
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig

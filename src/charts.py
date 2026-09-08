"""
The two exhibits that don't work as tables.

1. Airport supply against demand by hour. The whole Part B argument is the
   crossover: demand peaks when supply bottoms out. A table makes the reader
   do that work; a chart hands it to them.
2. Rejection rate by handset and document. The point is the contrast between
   three documents and the other three, which a chart shows instantly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as CFG

INK = "#1a1a1a"
ACCENT = "#c0392b"
COOL = "#2874a6"
MUTED = "#95a5a6"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def airport_hour_profile(hourly: pd.DataFrame, path):
    """Requests, online captains and fill rate at the terminals, by hour."""
    apt = hourly[hourly.zone_type == "airport_terminal"].copy()
    apt["hour"] = apt.hour_ts.dt.hour
    g = apt.groupby("hour").agg(requests=("requests", "mean"),
                                online=("online_captains", "mean"),
                                fulfilled=("fulfilled_requests", "mean"))
    g["fill"] = g.fulfilled / g.requests * 100

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.4), height_ratios=[2.2, 1],
                                  sharex=True)
    # shade the shortage window
    for lo, hi in [(-0.5, 3.5), (19.5, 23.5)]:
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.07, zorder=0)
        ax2.axvspan(lo, hi, color=ACCENT, alpha=0.07, zorder=0)

    ax.plot(g.index, g.requests, color=ACCENT, lw=2.4, label="Ride requests per hour")
    ax.plot(g.index, g.online, color=COOL, lw=2.4, label="Captains online")
    ax.fill_between(g.index, g.online, g.requests, where=(g.requests > g.online),
                    color=ACCENT, alpha=0.16, zorder=1)
    ax.set_ylabel("Per hour, per terminal")
    ax.set_ylim(0, 132)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)
    # No in-chart title: the deck slide supplies the headline, and repeating it
    # inside the exhibit just wastes vertical space.
    ax.annotate("8pm–4am:\n86% of all unmet demand", xy=(21.6, 88),
                xytext=(13.4, 74), fontsize=9, color=ACCENT, ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("Midday: 63 captains\nwaiting on 23 requests", xy=(12.2, 63),
                xytext=(7.4, 88), fontsize=9, color=COOL, ha="center",
                arrowprops=dict(arrowstyle="->", color=COOL, lw=1.1))
    _style(ax)

    ax2.plot(g.index, g.fill, color=INK, lw=2.2)
    ax2.axhline(96, color=MUTED, ls="--", lw=1.1)
    ax2.text(0.15, 98.5, "rest of city: 96–97%", fontsize=8.5, color=MUTED)
    ax2.set_ylabel("Requests filled")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylim(0, 112)
    ax2.set_yticks([0, 50, 100])
    ax2.set_yticklabels(["0%", "50%", "100%"])
    ax2.set_xticks(range(0, 24, 2))
    _style(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def device_document_gap(pairs: pd.DataFrame, coh: pd.DataFrame, path):
    """Unreadable-photo rejection rate by handset tier, per document."""
    p = pairs.merge(coh[["captain_id", "device_tier"]], on="captain_id", how="inner")
    p = p[p.verdict_ts.notna()]
    order = CFG.A4_PAPER_DOCS + CFG.CARD_DOCS
    label = {"RC": "Registration\nCertificate", "FITNESS": "Fitness\nCertificate",
             "INSURANCE": "Insurance", "DL": "Driving\nLicence",
             "AADHAAR": "Aadhaar", "PERMIT": "Permit"}
    t = (p.pivot_table(index="doc_type", columns="device_tier",
                       values="image_quality_fail", aggfunc="mean") * 100
         ).reindex(order)

    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.bar(x - w, t["low"], w, label="Basic phone", color=ACCENT)
    ax.bar(x, t["mid"], w, label="Mid phone", color="#e59866")
    ax.bar(x + w, t["high"], w, label="Premium phone", color=COOL)

    ax.axvline(2.5, color="#bbbbbb", ls="--", lw=1.2)
    ax.text(1.0, 37.0, "FULL-PAGE PAPER", fontsize=9.5, color=ACCENT,
            ha="center", weight="bold")
    ax.text(4.0, 37.0, "LAMINATED CARD", fontsize=9.5, color=MUTED,
            ha="center", weight="bold")
    ax.text(1.0, 34.0, "basic phones fail about 4x more often", fontsize=8.5,
            color=ACCENT, ha="center", style="italic")
    ax.text(4.0, 34.0, "no gap at all", fontsize=8.5, color=MUTED,
            ha="center", style="italic")

    for xi, d in zip(x, order):
        ax.text(xi - w, t.loc[d, "low"] + 0.8, f"{t.loc[d,'low']:.0f}%",
                ha="center", fontsize=8.5, color=ACCENT, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label[d] for d in order], fontsize=9)
    ax.set_ylabel("Submissions rejected for an\nunreadable photo")
    ax.set_ylim(0, 41)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 0.80))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def campaign_ladder(ladder: pd.DataFrame, path):
    """How the claimed campaign effect collapses under fair comparison."""
    labels = ["Everyone who got it\nvs everyone who didn't\n(THE CLAIM)",
              "vs captains who had also\ncleared two documents",
              "...and were still active\nwhen it was sent",
              "...allowing for city, vehicle,\nchannel and handset"]
    vals = ladder.lift_pp.tolist()
    colors = [MUTED, "#e59866", "#d98880", ACCENT]

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    bars = ax.barh(range(len(vals))[::-1], vals, color=colors, height=0.6)
    for i, (v, b) in enumerate(zip(vals, bars)):
        ax.text(v + 0.35, b.get_y() + b.get_height() / 2, f"+{v:.1f} pts",
                va="center", fontsize=10.5, weight="bold",
                color=ACCENT if i == 3 else INK)
    ax.set_yticks(range(len(vals))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 21)
    ax.set_xlabel("Apparent gain in approval rate (percentage points)")
    ax.annotate("", xy=(3.8, 0.15), xytext=(17.9, 2.9),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6,
                                connectionstyle="arc3,rad=-0.25"))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_all(hourly, pairs, coh, ladder):
    CFG.OUT.mkdir(exist_ok=True)
    charts = CFG.OUT / "charts"
    charts.mkdir(exist_ok=True)
    return [
        airport_hour_profile(hourly, charts / "airport_hour_profile.png"),
        device_document_gap(pairs, coh, charts / "device_document_gap.png"),
        campaign_ladder(ladder, charts / "campaign_ladder.png"),
    ]

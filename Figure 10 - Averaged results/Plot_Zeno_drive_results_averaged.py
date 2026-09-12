#!/usr/bin/env python3
"""
Plot extracted work and dissipated heat from Otto_* .txt files.

Encoding:
- n (measurements) -> color
- Gamma -> linestyle

Expected filenames like:
    Otto_lubricated_Fluctuating_Gamma_20.0_measurements_200_trajectories_50_tau_from_5_to_10.txt

Usage:
    python plot_otto_n_color_gamma_style.py --dir /path/to/files --out otto_overlay.pdf
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_GAMMAS = [20]
DEFAULT_MEASUREMENTS = [200, 400, 800, 1600, 3200]


def parse_gamma_from_filename(fname):
    base = os.path.basename(fname)
    m = re.search(r"gamma[_\- ]*([0-9]+(?:\.[0-9]+)?)", base, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_measurements_from_filename(fname):
    base = os.path.basename(fname)
    m = re.search(r"measurements[_\- ]*(\d+)", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def read_three_column_file(path):
    header = None
    data_rows = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if header is None:
                    header = s
                continue
            data_rows.append(s)

    if not data_rows:
        raise ValueError(f"No numeric data found in {path}")

    data = np.loadtxt(data_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}, got {data.shape[1]}")

    tau = data[:, 0]
    w_out = data[:, 1]
    q_meas = data[:, 2]
    return header, tau, w_out, q_meas


def collect_series(directory, gammas=None, measurements=None):
    """
    results[gamma][n] = list of (path, tau, w_out, q_meas)
    """
    if gammas is not None:
        gammas = set(gammas)
    if measurements is not None:
        measurements = set(measurements)

    files = glob.glob(os.path.join(directory, "*.txt"))
    if not files:
        files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)

    results = defaultdict(lambda: defaultdict(list))

    for path in files:
        try:
            _, tau, w_out, q_meas = read_three_column_file(path)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue

        gamma = parse_gamma_from_filename(path)
        nmeas = parse_measurements_from_filename(path)

        if gamma is None or nmeas is None:
            continue

        if gammas is not None and gamma not in gammas:
            continue
        if measurements is not None and nmeas not in measurements:
            continue

        results[gamma][nmeas].append((path, tau, w_out, q_meas))

    return results


def gamma_linestyle(gamma):
    styles = {
        20.0: "-",
        40.0: "--",
        60.0: ":",
        80.0: "-.",
    }
    return styles.get(gamma, "-")


def plot_overlay_pdf(results, gammas_order, measurements_order, out_pdf):
    """
    Single-page PDF with two panels:
      - W_out
      - Q_meas

    Encoding:
      - color = n
      - linestyle = Gamma
    """
    n_cmap = cm.get_cmap("viridis")
    if measurements_order:
        n_norm = Normalize(vmin=min(measurements_order), vmax=max(measurements_order))
    else:
        n_norm = Normalize(vmin=0, vmax=1)

    work_cmap = cm.get_cmap("Blues")
    heat_cmap = cm.get_cmap("Reds")

    def n_color(n):
        return n_cmap(n_norm(n))

    fig, (ax_w, ax_q) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    plotted_any = False

    for gamma in gammas_order:
        if gamma not in results:
            continue

        ls = gamma_linestyle(gamma)

        for nmeas in measurements_order:
            if nmeas not in results[gamma]:
                continue

            # Keep n as the color parameter. Use blue tones for work, red for heat.
            color_w = work_cmap(n_norm(nmeas))
            color_q = heat_cmap(n_norm(nmeas))

            for _, tau, w_out, q_meas in results[gamma][nmeas]:
                order = np.argsort(tau)
                tau = tau[order]
                w_out = w_out[order]
                q_meas = q_meas[order]

                ax_w.plot(
                    tau, w_out,
                    color=color_w,
                    linestyle=ls,
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                )
                ax_q.plot(
                    tau, q_meas,
                    color=color_q,
                    linestyle=ls,
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                )
                plotted_any = True

    if not plotted_any:
        raise RuntimeError("No curves were plotted. Check filenames and filters.")

    #ax_w.set_title("Average Zeno Extracted Work", fontsize=22)
    #ax_q.set_title("Average Dissipated heat", fontsize=22)

    ax_w.set_xlabel(r"$\tau_{\rm comp}+\tau_{\rm exp}$", fontsize=27)
    ax_q.set_xlabel(r"$\tau_{\rm comp}+\tau_{\rm exp}$", fontsize=27)
    ax_w.set_ylabel(r"$-\Delta W_{\rm mean}^{\rm (Zeno)}$", fontsize=27)
    ax_q.set_ylabel(r"$\Delta Q_{\rm mean}^{\rm (meas)}$", fontsize=27)

    ax_w.grid(True, alpha=0.25)
    ax_q.grid(True, alpha=0.25)
    ax_w.tick_params(axis="both", labelsize=24)
    ax_q.tick_params(axis="both", labelsize=24)

    # Legends: n as color, Gamma as linestyle
    n_handles_work = [
        Line2D([0], [0], color=work_cmap(n_norm(n)), lw=3, label=fr"$n={n}$")
        for n in measurements_order
    ]
    n_handles_heat = [
        Line2D([0], [0], color=heat_cmap(n_norm(n)), lw=3, label=fr"$n={n}$")
        for n in measurements_order
    ]

    gamma_handles = [
        Line2D([0], [0], color="black", lw=3, linestyle=gamma_linestyle(g), label=fr"$\Gamma={g:g}$")
        for g in gammas_order
    ]

    leg1 = ax_w.legend(
        handles=n_handles_work,
        #title=r"Measurements",
        fontsize=20,
        title_fontsize=21,
        loc="best",
        frameon=True,
    )
    ax_w.add_artist(leg1)
    # ax_w.legend(
    #     handles=gamma_handles,
    #     title=r"Gamma",
    #     fontsize=10,
    #     title_fontsize=11,
    #     loc="lower left",
    #     frameon=True,
    # )

    leg3 = ax_q.legend(
        handles=n_handles_heat,
        #title=r"Measurements",
        fontsize=20,
        title_fontsize=21,
        loc="best",
        frameon=True,
    )
    ax_q.add_artist(leg3)
    # ax_q.legend(
    #     handles=gamma_handles,
    #     title=r"Gamma",
    #     fontsize=10,
    #     title_fontsize=11,
    #     loc="lower left",
    #     frameon=True,
    # )

    #fig.suptitle("Otto results: n encoded by color, Gamma encoded by linestyle", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)

    plt.close(fig)
    print(f"Saved figure to {out_pdf}")


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Plot W_out and Q_meas from Otto_* .txt files with n as color and Gamma as linestyle."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing the .txt files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="otto_n_color_gamma_style.pdf",
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GAMMAS),
        help="Comma-separated Gamma values to include, e.g. 20,40,60,80",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        default=",".join(str(m) for m in DEFAULT_MEASUREMENTS),
        help="Comma-separated measurements values to include, e.g. 50,100,200,400",
    )
    args = parser.parse_args()

    try:
        gammas = parse_float_list(args.gammas)
    except Exception:
        print("Could not parse --gammas; using defaults.", file=sys.stderr)
        gammas = [float(g) for g in DEFAULT_GAMMAS]

    try:
        measurements = parse_int_list(args.measurements)
    except Exception:
        print("Could not parse --measurements; using defaults.", file=sys.stderr)
        measurements = DEFAULT_MEASUREMENTS

    results = collect_series(args.dir, gammas=gammas, measurements=measurements)
    plot_overlay_pdf(
        results,
        gammas_order=gammas,
        measurements_order=measurements,
        out_pdf=args.out,
    )


if __name__ == "__main__":
    main()
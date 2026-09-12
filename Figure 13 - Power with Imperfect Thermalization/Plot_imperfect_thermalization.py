#!/usr/bin/env python3
"""
Plot power and efficiency vs cycle time for lubricated / non-lubricated data.

Features:
- Separate PDF for power and efficiency
- Lubricated curves: orange colormap, varying with Gamma
- Non-lubricated curves: fixed blue dash-dot style
- Supports multiple files with different Gamma values
- Can use an explicit FILES list, or command-line files, or --dir/--pattern
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_FILES = [
    "Imperfect_otto_thermalization_sweep_lubricated_Gamma_60.000.txt",
    "Imperfect_otto_thermalization_sweep_non_lubricated_Gamma_60.000.txt",
    # Add more here if you want a fixed list
]


def truncate_colormap(cmap, minval=0.18, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def find_gamma_from_filename(fname):
    base = os.path.basename(fname).lower()
    m = re.search(r"gamma[_\s]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", base)
    if m:
        return float(m.group(1))
    return None


def find_lubrication_from_filename(fname):
    base = os.path.basename(fname).lower()
    if "non_lubricated" in base or "non-lubricated" in base or "nonlubricated" in base:
        return "non_lubricated"
    if "lubricated" in base:
        return "lubricated"
    return None


def read_data_file(path):
    """
    Expected format:
    # total_time tau_hot tau_cold eta power W_out Q_hot Q_cold W_comp W_exp W_net
    """
    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 5:
        raise ValueError(f"File {path} has only {data.shape[1]} columns; need at least 5.")

    total_time = data[:, 0]
    eta = data[:, 3]
    power = data[:, 4]

    order = np.argsort(total_time)
    return total_time[order], eta[order], power[order]


def collect_files(args_files, directory=None, pattern=None):
    files = []

    if args_files:
        files.extend(args_files)
    else:
        files.extend(DEFAULT_FILES)

    if directory is not None:
        if pattern is None:
            pattern = "*"
        files.extend(glob.glob(os.path.join(directory, pattern)))

    # keep only existing, unique files
    seen = set()
    cleaned = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            cleaned.append(f)
            seen.add(f)

    return cleaned


def plot_observable(series, observable_key, ylabel, outpath):
    """
    series = list of dicts with keys:
      - gamma
      - lubrication
      - t
      - eta
      - power
    observable_key = 'power' or 'eta'
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    gammas = sorted({s["gamma"] for s in series if s["gamma"] is not None})
    nonzero_gammas = [g for g in gammas if g != 0.0]

    cmap_full = cm.get_cmap("Oranges")
    cmap = truncate_colormap(cmap_full, minval=0.38, maxval=1.0) if nonzero_gammas else cmap_full
    norm = None
    if nonzero_gammas:
        norm = Normalize(vmin=min(nonzero_gammas), vmax=max(nonzero_gammas))

    # legend handles
    handles = []
    labels = []
    used_labels = set()

    for s in series:
        gamma = s["gamma"]
        lub = s["lubrication"]
        t = s["t"]
        y = s[observable_key]

        if lub == "lubricated":
            color = cmap(norm(gamma)) if (norm is not None and gamma is not None) else cmap(0.9)
            linestyle = "-"
            linewidth = 2.4
            marker = "^"
            marker_size = 0
        else:
            color = "tab:blue"
            linestyle = "-"
            linewidth = 1
            marker = "o"
            marker_size = 0

        line, = ax.plot(
            t,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=marker_size,
            markevery=max(1, len(t) // 10),
        )

        gamma_str = f"{gamma:g}" if gamma is not None else "unknown"
        label = f"lubricated  Γ={gamma_str},n=400" if lub == "lubricated" else "non-lubricated"
        if label not in used_labels:
            handles.append(line)
            labels.append(label)
            used_labels.add(label)

    ax.set_xlabel(r"$\tau$", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, linestyle=":", alpha=0.7)

    if handles:
        ax.legend(handles, labels, fontsize=18, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot power and efficiency vs cycle time for lubricated/non-lubricated data."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Explicit input files. If omitted, uses DEFAULT_FILES.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Optional directory to scan for files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*Gamma_*.txt",
        help="Glob pattern used with --dir.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="cycle_time",
        help="Prefix for output PDFs.",
    )
    args = parser.parse_args()

    files = collect_files(args.files, directory=args.dir, pattern=args.pattern)

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    series = []
    for fpath in files:
        lub = find_lubrication_from_filename(fpath)
        gamma = find_gamma_from_filename(fpath)

        if lub is None:
            print(f"Skipping {fpath}: cannot determine lubrication type from filename.", file=sys.stderr)
            continue
        if gamma is None:
            print(f"Skipping {fpath}: cannot parse Gamma from filename.", file=sys.stderr)
            continue

        try:
            t, eta, power = read_data_file(fpath)
        except Exception as exc:
            print(f"Skipping {fpath}: {exc}", file=sys.stderr)
            continue

        series.append(
            {
                "file": os.path.basename(fpath),
                "lubrication": lub,
                "gamma": gamma,
                "t": t,
                "eta": eta,
                "power": power,
            }
        )

    if not series:
        print("No readable data found.", file=sys.stderr)
        sys.exit(1)

    plot_observable(
        series,
        observable_key="power",
        ylabel="Power",
        outpath=f"{args.out_prefix}_power.pdf",
    )
    plot_observable(
        series,
        observable_key="eta",
        ylabel="Efficiency",
        outpath=f"{args.out_prefix}_efficiency.pdf",
    )


if __name__ == "__main__":
    main()
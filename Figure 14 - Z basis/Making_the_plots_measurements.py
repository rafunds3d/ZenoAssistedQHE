#!/usr/bin/env python3
"""
Plotting script for lubricated / non-lubricated result files grouped by Measurement count,
for a fixed (hardcoded) Gamma. Also optionally overlays Gamma=0 series in blue.

Usage:
    python plot_by_measurements_fixed_gamma.py --dir /path/to/txt/files --out plot.png

Set the fixed Gamma by editing DEFAULT_GAMMA below.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
import re
from matplotlib.colors import PowerNorm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------- CONFIG ----------
DEFAULT_MEASUREMENTS = [0, 250, 2500]#, 1750, 2000, 3000, 4000]
DEFAULT_GAMMA = 50   # <<-- set fixed Gamma here
OVERLAY_GAMMA0 = True  # set False if you don't want to overlay Gamma=0
# ----------------------------


def find_gamma_from_filename(fname):
    base = os.path.basename(fname).lower()
    idx = base.find('gamma')
    if idx == -1:
        return None
    tail = base[idx + len('gamma'):]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None


def find_measurement_from_filename(fname):
    base = os.path.basename(fname).lower()
    # prefer 'measurements' token, fall back to 'measurement'
    if 'measurements' in base:
        idx = base.find('measurements')
        token_len = len('measurements')
    elif 'measurement' in base:
        idx = base.find('measurement')
        token_len = len('measurement')
    else:
        return None
    tail = base[idx + token_len:]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None


def read_two_column_file(path):
    with open(path, 'r') as f:
        lines = [ln for ln in (l.rstrip('\n') for l in f) if ln.strip() != '']
    if not lines:
        raise ValueError(f"Empty file: {path}")
    header = lines[0]
    data_lines = lines[1:]
    if re.match(r"^[\s\d.+\-eE]+$", header):
        data_lines = lines
    try:
        data = np.loadtxt(data_lines)
    except Exception as e:
        raise ValueError(f"Could not parse numeric data in {path}: {e}")
    if data.ndim == 1:
        if len(data) < 2:
            raise ValueError(f"File {path} does not look like two columns")
        x = np.array([data[0]])
        y = np.array([data[1]])
    else:
        x = data[:, 0]
        y = data[:, 1]
    return header, x, y


def collect_series_by_measurement(directory, gamma_fixed, measurements=None, debug=False):
    """
    Collect files matching a specific gamma_fixed (int). Returns results[meas][state] list.
    """
    pattern = os.path.join(directory, '*.txt')
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    results = defaultdict(lambda: defaultdict(list))

    for fpath in files:
        try:
            header, x, y = read_two_column_file(fpath)
        except Exception as e:
            print(f"Skipping {fpath} (couldn't read): {e}", file=sys.stderr)
            continue

        low_header = header.lower()
        low_fname = os.path.basename(fpath).lower()

        # detect non-lubricated explicitly first to avoid misclassification
        if re.search(r'\b(?:non|no|un)[-_ ]?lubricat', low_header) or re.search(r'\b(?:non|no|un)[-_ ]?lubricat', low_fname):
            is_lub = False
        elif 'lubricat' in low_header or 'lubricat' in low_fname:
            is_lub = True
        else:
            # default assume non-lubricated if nothing explicit
            is_lub = False
        state = 'lubricated' if is_lub else 'non_lubricated'

        # find gamma and require it equals gamma_fixed
        gamma = find_gamma_from_filename(fpath)
        if gamma is None:
            m = re.search(r"gamma\s*[=:_-]?\s*(\d+)", header, flags=re.IGNORECASE)
            if m:
                gamma = int(m.group(1))
        if gamma is None or gamma != gamma_fixed:
            if debug:
                print(f"SKIP (gamma mismatch): {fpath}  parsed_gamma={gamma}", file=sys.stderr)
            continue

        meas = find_measurement_from_filename(fpath)
        if meas is None:
            m2 = re.search(r"measurements?\s*[=:_-]?\s*(\d+)", header, flags=re.IGNORECASE)
            if m2:
                meas = int(m2.group(1))

        if measurements is None or (meas in measurements):
            results[meas][state].append((fpath, header, x, y))
            if debug:
                print(f"FOUND: {fpath}  gamma={gamma}, meas={meas}, state={state}", file=sys.stderr)

    return results


def make_color_for_measurement(meas, cmap, norm):
    if meas is None:
        return '0.5'
    start_frac = 0.05
    frac = float(0.6*norm(meas))
    return cmap(start_frac + (1.0 - start_frac) * frac)


def plot_results_by_measurement(results, measurements_list, gamma_fixed, results_gamma0=None, savepath=None, show=True):
    """
    results: main results for gamma_fixed
    results_gamma0: optional dict of gamma==0 results (same structure results_gamma0[meas][state])
    """
    plt.figure(figsize=(10, 5))

    cmap = cm.get_cmap('Oranges_r')  # reversed so larger M -> lighter, if you want inverse
    valid_meas = [m for m in (measurements_list or []) if m is not None]
    if not valid_meas:
        valid_meas = sorted([m for m in results.keys() if isinstance(m, (int, float))])
    if not valid_meas:
        raise RuntimeError("No measurement values found to plot.")
    norm = Normalize(vmin=min(valid_meas), vmax=max(valid_meas))
    

    markers = {'non_lubricated': '^', 'lubricated': '^'}

    marker_list = ['s', '^', 'o', 'D', '*', 'v', '<', '>', 'p', 'h']   # markers to cycle

    # Ensure measurements_list is sorted and unique
    unique_meas = sorted([m for m in measurements_list if m is not None])
    if not unique_meas:
        raise RuntimeError("No measurement values found to plot.")
    
    # Build mapping meas -> (marker, linestyle)
    meas_style = {}
    for i, meas in enumerate(unique_meas):
        marker = marker_list[i % len(marker_list)]
        # Optional: first measurement gets dash-dot line, others solid
        linestyle = '-' if i == 0 else '-'
        meas_style[meas] = (marker, linestyle)


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    ax = plt.gca()
    seen_labels = set()
    plotted_any = False

    # --- Plot main (Γ fixed) series: all non‑lubricated and lubricated ---
    for state_key in ['non_lubricated', 'lubricated']:
        for meas in unique_meas:
            if meas not in results:
                continue
            group = results.get(meas, {})
            color = make_color_for_measurement(meas, cmap, norm)
            marker_for_meas, linestyle_for_meas = meas_style[meas]

            for (fname, header, x, y) in group.get(state_key, []):
                order = np.argsort(x)
                x2 = x[order]
                y2 = y[order]

                # Build label (only once per combination)
                label_base = f"{'' if state_key=='lubricated' else 'Non‑lubricated'} n={meas}"
                label = label_base if label_base not in seen_labels else None
                if label is not None:
                    seen_labels.add(label_base)

                # Control marker frequency (tune thresholds as needed)
                if meas < 10:
                    markevery = 1
                elif meas < 50:
                    markevery = 1
                elif meas < 200:
                    markevery = 1
                else:
                    markevery = 1

                ax.plot(x2, y2,
                        marker=marker_for_meas,
                        linestyle=linestyle_for_meas,
                        label=label,
                        markersize=0,          # visible marker size
                        linewidth=2.5,
                        color=color,
                        markeredgewidth=0.5,
                        markeredgecolor='black',
                        markerfacecolor=color,
                        markevery=markevery)

                # try:
                #     ax.fill_between(x2, y2, alpha=0.12, facecolor=color)
                # except Exception:
                #     pass
                plotted_any = True

    # --- Overlay Γ=0 curves (if present) in blue, with a single legend entry ---
    if results_gamma0:
        first_gamma0_legend = True
        for meas in unique_meas:
            if meas not in results_gamma0:
                continue
            group0 = results_gamma0.get(meas, {})
            for state_key in ['non_lubricated', 'lubricated']:
                for (fname, header, x, y) in group0.get(state_key, []):
                    order = np.argsort(x)
                    x2 = x[order]
                    y2 = y[order]

                    # Use same marker as for this meas, but blue colour
                    marker_for_meas, _ = meas_style[meas]   # ignore linestyle, use solid
                    label = "Non-lubricated" if first_gamma0_legend else None
                    first_gamma0_legend = False

                    ax.plot(x2, y2,
                            marker=marker_for_meas,
                            linestyle='-.',
                            label=label,
                            markersize=0,
                            linewidth=2.5,
                            color='tab:blue',
                            alpha=0.9,
                            markeredgewidth=0.5,
                            markeredgecolor='black',
                            markerfacecolor='tab:blue',
                            markevery=markevery)   # same markevery as above

                    # try:
                    #     ax.fill_between(x2, y2, alpha=0.08, facecolor='tab:blue')
                    # except Exception:
                    #     pass
                    plotted_any = True

    if not plotted_any:
        raise RuntimeError("No data were plotted – check your data structures.")

    # Styling: increase font sizes for labels, ticks, legend
    label_fontsize = 26    # axis labels
    tick_fontsize = 20     # tick labels (numbers)
    legend_fontsize = 18   # legend text
    # (tweak the numbers above to taste)

    ax.set_xlabel(r'$\tau_{\rm comp}$', fontsize=label_fontsize)
    ax.set_ylabel(r'$C_{\ell_1}$', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True)
    #ax.set_title(f"Gamma = {gamma_fixed}", fontsize=label_fontsize)
    #ax.legend(fontsize=legend_fontsize, loc='best')
    # Re-order legend so the Gamma=0 "Non-lubricated" entry appears first.
    handles, labels = ax.get_legend_handles_labels()

    # Find the gamma0 label: one that contains 'non' but is NOT an 'n=' entry.
    gamma0_idx = None
    for i, lab in enumerate(labels):
        ll = lab.lower()
        if 'non' in ll and 'n=' not in ll:
            gamma0_idx = i
            break

    if gamma0_idx is not None and gamma0_idx != 0:
        new_order = [gamma0_idx] + [i for i in range(len(labels)) if i != gamma0_idx]
        handles = [handles[i] for i in new_order]
        labels = [labels[i] for i in new_order]

    ax.legend(handles, labels, fontsize=legend_fontsize, loc='best')
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Saved figure to {savepath}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot lubricated/non-lubricated results grouped by Measurements for a fixed Gamma')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing .txt files')
    parser.add_argument('--out', type=str, default=None, help='If set, save figure to this file (e.g. plot.png)')
    parser.add_argument('--measurements', type=str, default=','.join(str(m) for m in DEFAULT_MEASUREMENTS),
                        help='Comma-separated list of measurement values to include (e.g. 250,500,1000). Use blank to include any parsed measurements.')
    parser.add_argument('--no-show', action='store_true', help="Don't call plt.show() (useful for headless runs)")
    parser.add_argument('--debug', action='store_true', help="Print parsed (gamma, measurement, state) for each accepted file (diagnostic)")
    args = parser.parse_args()

    # parse measurements argument: allow empty string to mean "include all"
    if args.measurements.strip() == '':
        measurements = None
    else:
        try:
            measurements = [int(x) for x in args.measurements.split(',') if x.strip() != '']
        except Exception:
            print('Could not parse --measurements; using defaults', file=sys.stderr)
            measurements = DEFAULT_MEASUREMENTS

    gamma_fixed = DEFAULT_GAMMA

    # collect main results for the desired (hardcoded) Gamma
    results = collect_series_by_measurement(args.dir, gamma_fixed=gamma_fixed, measurements=measurements, debug=args.debug)

    # optionally also collect Gamma=0 series to overlay (independent of gamma_fixed)
    results_gamma0 = None
    if OVERLAY_GAMMA0:
        results_gamma0 = collect_series_by_measurement(args.dir, gamma_fixed=0, measurements=measurements, debug=args.debug)

    if measurements is None:
        measurements_list = sorted([m for m in (results.keys() | (results_gamma0.keys() if results_gamma0 else set())) if isinstance(m, (int, float))])
    else:
        measurements_list = measurements

    if not measurements_list:
        raise RuntimeError("No measurements selected / found to plot.")

    plot_results_by_measurement(results, measurements_list, gamma_fixed=gamma_fixed, results_gamma0=results_gamma0, savepath=args.out, show=not args.no_show)


if __name__ == '__main__':
    main()
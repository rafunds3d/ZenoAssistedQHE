#!/usr/bin/env python3
"""
Plot W_out and W_out_joint vs tau from data files containing lubricated/non-lubricated results,
grouped by the Gamma value extracted from the filename.

Behavior:
- Joint series (W_out_joint) are plotted for all requested --gammas (unless missing files).
- System series (W_out) are plotted for ALL gammas unless --sys-gamma is provided;
  when --sys-gamma is given only system curves for that Gamma are shown.
- Legend contains marker-type entries (Lubricated/Non-lubricated joint/sys),
  colored swatches for each Gamma actually plotted (with "(joint)" / "(sys)"),
  and the dashed Otto reference line.
- No colorbar is shown.
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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D

DEFAULT_GAMMAS = [20,40,60,80,100]


def truncate_colormap(cmap, minval=0.18, maxval=1.0, n=256):
    """Return a truncated copy of cmap that uses the colormap range [minval, maxval]."""
    return LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )


def find_gamma_from_filename(fname):
    base = os.path.basename(fname).lower()
    m = re.search(r"gamma[_\s]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", base)
    if m:
        return float(m.group(1))
    return None


def find_measurement_from_filename(fname):
    base = os.path.basename(fname).lower()
    m = re.search(r"measurements?[_\s]*([0-9]+)", base)
    if m:
        return int(m.group(1))
    return None


def read_data_file(path):
    """Expect header line then numeric rows with >=7 columns:
    tau eta eta_joint W_out W_out_joint power power_joint
    """
    with open(path, 'r') as f:
        header = f.readline().strip()
        lines = [ln for ln in (l.strip() for l in f) if ln]
    if not lines:
        raise ValueError(f"Empty file: {path}")

    data = np.loadtxt(lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"File {path} does not have >=7 columns (got {data.shape[1]})")

    tau = data[:, 0]
    W_out = data[:, 3]
    W_out_joint = data[:, 4]

    return header, tau, W_out, W_out_joint


def collect_series(directory, gammas=None):
    """Scan directory for .txt files (recursive) and group by gamma and lubrication."""
    pattern = os.path.join(directory, '*.txt')
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    results = defaultdict(lambda: defaultdict(list))

    # convert gammas to a set of floats for exact membership checks
    gammas_set = None
    if gammas is not None:
        gammas_set = set(float(g) for g in gammas)

    for fpath in files:
        low_fname = os.path.basename(fpath).lower()
        if 'lubricated' in low_fname:
            lubrication = 'lubricated'
        elif 'non-lubricated' in low_fname or 'nonlubricated' in low_fname:
            lubrication = 'non_lubricated'
        else:
            print(f"Skipping {fpath}: cannot determine lubrication state from filename", file=sys.stderr)
            continue

        gamma = find_gamma_from_filename(fpath)
        if gamma is None:
            print(f"Skipping {fpath}: cannot parse Gamma from filename", file=sys.stderr)
            continue

        if gammas_set is not None and float(gamma) not in gammas_set:
            continue

        measurement = find_measurement_from_filename(fpath)
        try:
            header, tau, W_out, W_out_joint = read_data_file(fpath)
        except Exception as e:
            print(f"Skipping {fpath}: {e}", file=sys.stderr)
            continue

        results[float(gamma)][lubrication].append({
            'measurement': measurement,
            'tau': tau,
            'W_out': W_out,
            'W_out_joint': W_out_joint,
            'fname': fpath
        })

    # sort entries by measurement if present
    for gamma in list(results.keys()):
        for lub in results[gamma]:
            results[gamma][lub].sort(key=lambda x: x['measurement'] if x['measurement'] is not None else 0)

    return results


def plot_results(results, gammas, savepath=None, show=True, sys_gamma=None):
    plt.figure(figsize=(10, 6))

    # prepare gamma lists
    gammas = [float(g) for g in gammas]
    non_zero_gammas = sorted([g for g in gammas if g != 0.0])
    zero_gamma_exists = (0.0 in gammas) and (0.0 in results)

    # truncated Oranges for joint
    cmap_full = cm.get_cmap('Oranges')
    cmap = truncate_colormap(cmap_full, minval=0.18, maxval=1.0) if non_zero_gammas else cmap_full
    norm = None
    if non_zero_gammas:
        norm = Normalize(vmin=min(non_zero_gammas), vmax=max(non_zero_gammas))

    # marker mapping (marker only; we keep markersize=0 for line-only appearance, but legend uses handles)
    marker_map = {
        ('lubricated', True): '^',   # lubricated joint
        ('lubricated', False): 'x',  # lubricated system
        ('non_lubricated', False): 'o', # non-lubricated system
        ('non_lubricated', True): 's',  # non-lubricated joint
    }

    plotted_any = False

    # marker-type handles + labels for legend
    marker_handles = {}
    marker_labels = {
        ('lubricated', True):   'Lubricated (joint)',
        ('lubricated', False):  'Lubricated (sys)',
        ('non_lubricated', False): 'Non-lubricated (sys)',
        ('non_lubricated', True):  'Non-lubricated (joint)',
    }

    # track which gammas had joint/sys actually plotted
    gammas_plotted_joint = set()
    gammas_plotted_sys = set()

    # Plot non-zero gammas (joint always; system depending on sys_gamma)
    for gamma in gammas:
        if gamma == 0.0:
            continue
        if gamma not in results:
            continue

        color_joint = cmap(norm(gamma)) if norm is not None else cmap(0.9)
        # choose system color (fixed)
        color_sys = 'black'

        for lubrication in ['lubricated', 'non_lubricated']:
            if lubrication not in results[gamma]:
                continue
            for entry in results[gamma][lubrication]:
                tau = entry['tau']
                order = np.argsort(tau)
                tau_sorted = tau[order]
                W_out_sorted = entry['W_out'][order]
                W_out_joint_sorted = entry['W_out_joint'][order]

                # joint (always plotted for available gammas)
                marker_joint = marker_map.get((lubrication, True))
                if marker_joint:
                    line_joint, = plt.plot(tau_sorted, W_out_joint_sorted,
                                           marker=marker_joint, linestyle='-',
                                           color=color_joint, markersize=0,
                                           markevery=max(1, int(len(tau_sorted) * 0.1)),
                                           linewidth=1.0, label=None)
                    key_joint = (lubrication, True)
                    if key_joint not in marker_handles:
                        marker_handles[key_joint] = line_joint
                    plotted_any = True
                    gammas_plotted_joint.add(gamma)

                # system (only if sys_gamma is None or equals this gamma)
                marker_sys = marker_map.get((lubrication, False))
                if marker_sys:
                    if (sys_gamma is None) or (abs(float(sys_gamma) - float(gamma)) < 1e-12):
                        line_sys, = plt.plot(tau_sorted, W_out_sorted,
                                             marker=marker_sys, linestyle='-',
                                             color=color_sys, markersize=0,
                                             markevery=max(1, int(len(tau_sorted) * 0.1)),
                                             linewidth=2.4, label=None)
                        key_sys = (lubrication, False)
                        if key_sys not in marker_handles:
                            marker_handles[key_sys] = line_sys
                        plotted_any = True
                        gammas_plotted_sys.add(gamma)
                    # otherwise skip plotting this system series for this gamma

    # Plot Gamma = 0 if present (joint always; sys only if sys_gamma is None or 0)
    if zero_gamma_exists:
        color_joint_0 = 'tab:blue'
        color_sys_0 = 'purple'
        gamma0 = 0.0
        for lubrication in ['lubricated', 'non_lubricated']:
            if lubrication not in results[gamma0]:
                continue
            for entry in results[gamma0][lubrication]:
                tau = entry['tau']
                order = np.argsort(tau)
                tau_sorted = tau[order]
                W_out_sorted = entry['W_out'][order]
                W_out_joint_sorted = entry['W_out_joint'][order]

                # joint at gamma=0
                marker_joint = marker_map.get((lubrication, True))
                if marker_joint:
                    line_joint, = plt.plot(tau_sorted, W_out_joint_sorted,
                                           marker=marker_joint, linestyle='-',
                                           color=color_joint_0, markersize=0,
                                           markevery=max(1, int(len(tau_sorted) * 0.1)),
                                           linewidth=1.0, label=None)
                    key_joint = (lubrication, True)
                    if key_joint not in marker_handles:
                        marker_handles[key_joint] = line_joint
                    plotted_any = True
                    gammas_plotted_joint.add(gamma0)

                # system at gamma=0: only if requested
                marker_sys = marker_map.get((lubrication, False))
                if marker_sys:
                    if (sys_gamma is None) or (abs(float(sys_gamma) - 0.0) < 1e-12):
                        line_sys, = plt.plot(tau_sorted, W_out_sorted,
                                             marker=marker_sys, linestyle='-.',
                                             color=color_sys_0, markersize=0,
                                             markevery=max(1, int(len(tau_sorted) * 0.1)),
                                             linewidth=1.0, label=None)
                        key_sys = (lubrication, False)
                        if key_sys not in marker_handles:
                            marker_handles[key_sys] = line_sys
                        plotted_any = True
                        gammas_plotted_sys.add(gamma0)

    if not plotted_any:
        raise RuntimeError('No series plotted – check your directory and file naming')

    # horizontal reference: ideal Otto value (and include in legend)
    ideal_W = 0.29935
    # hline = plt.axhline(ideal_W, color='k', linestyle='--', linewidth=1.5)

    # --- Build legend entries ---
    ax = plt.gca()
    handles = []
    labels = []

    # 1) Add marker-type legend entries (in a sensible order), if available
    marker_order = [('lubricated', True), ('lubricated', False),
                    ('non_lubricated', False), ('non_lubricated', True)]
    # for key in marker_order:
    #     if key in marker_handles:
    #         handles.append(marker_handles[key])
    #         labels.append(marker_labels[key])

    # 2) Add Gamma swatches for joint and sys separately (only for gammas that were actually plotted)
    # collect present non-zero gammas that are both requested and present in results
    present_nonzero_gammas = [g for g in non_zero_gammas if g in results]

    # For deterministic ordering, sort them
    present_nonzero_gammas = sorted(present_nonzero_gammas)

    for gamma in present_nonzero_gammas:
        # joint swatch
        if gamma in gammas_plotted_joint:
            color_joint = cmap(norm(gamma)) if norm is not None else cmap(0.9)
            swatch_joint = Line2D([0], [0], color=color_joint, lw=2)
            handles.append(swatch_joint)
            labels.append(f"$\\Gamma={gamma:g}$ (sys+lub)")
        # sys swatch (only if sys for this gamma was plotted)
        if gamma in gammas_plotted_sys:
            # match color used for system lines for non-zero gammas (we used 'blue')
            color_sys_legend = 'black'
            swatch_sys = Line2D([0], [0], color=color_sys_legend, lw=2)
            handles.append(swatch_sys)
            labels.append(f"$\\Gamma={gamma:g}$ (sys)")

    # include gamma = 0 swatches if present
    if zero_gamma_exists:
        if 0.0 in gammas_plotted_joint:
            sw = Line2D([0], [0], color='tab:blue', lw=2)
            handles.append(sw)
            labels.append(f"$\\Gamma=0$ (sys+lub)")
        if 0.0 in gammas_plotted_sys:
            sw = Line2D([0], [0], color='purple', lw=2)
            handles.append(sw)
            labels.append(f"$\\Gamma=0$ (sys)")

    # append Otto reference line handle + label
    # handles.append(hline)
    # labels.append(f'Otto ref: {ideal_W:.4f}')

    # show legend (combined)
    if handles:
        ax.legend(handles, labels, fontsize=18, loc='best')

    # Axis styling
    ax.set_xlabel(r'$\tau_{\mathrm{comp}} + \tau_{\mathrm{exp}}$', fontsize=26)
    ax.set_ylabel(r'Extracted Work', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Figure saved to {savepath}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot W_out and W_out_joint vs tau from lubricated/non-lubricated data files, grouped by Gamma.'
    )
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing .txt files (searched recursively)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output file name for the plot (e.g., plot.png)')
    parser.add_argument('--gammas', type=str,
                        default=','.join(str(g) for g in DEFAULT_GAMMAS),
                        help='Comma-separated list of Gamma values to include (floats)')
    parser.add_argument('--sys-gamma', type=float, default=None,
                        help='If set, only plot system (W_out) series for this Gamma (float). If omitted, plot all system series.')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (useful for headless runs)')
    args = parser.parse_args()

    try:
        gammas = [float(x.strip()) for x in args.gammas.split(',') if x.strip()]
    except Exception:
        print('Error parsing --gammas. Using defaults.', file=sys.stderr)
        gammas = DEFAULT_GAMMAS

    results = collect_series(args.dir, gammas=gammas)
    plot_results(results, gammas=gammas, savepath=args.out, show=not args.no_show, sys_gamma=80)


if __name__ == '__main__':
    main()
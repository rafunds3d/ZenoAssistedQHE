"""
Plotting script for lubricated / non-lubricated result files grouped by tau.

Usage:
    python plot_lubricated_nonlubricated_tau.py --dir /path/to/txt/files --out plot.png

Behavior / assumptions:
- Looks for files with "tau" in their filename (case-insensitive) and ending with .txt.
- Reads the first line of each file as a header. If the header contains the substring
  'lubricated' (case-insensitive) the file is considered a "lubricated" series; otherwise
  it's treated as non-lubricated.
- Numeric data is expected from the second line on: two columns separated by whitespace
  (x and y). Empty lines are ignored.
- Tau considered by default: [0,5, 1, 2, 5, 10, 20]. You can override with --taus.
- Color mapping: tau==0 -> blue; tau>0 -> colormap 'Oranges' from light (small tau)
  to dark (tau=120). Each series is plotted with markers and a light shaded fill under
  the curve.

Outputs:
- Displays the plot and optionally saves to --out filename.

"""

import argparse
import glob
import os
import sys
from collections import defaultdict
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

DEFAULT_TAUS = [1, 2, 5, 10, 20]
# DEFAULT_MEASUREMENTS = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000]


def find_tau_from_filename(fname, re=None):
    """Try to extract the integer tau from the filename.
    Looks for patterns like '_tau_15' or 'tau-15' (case-insensitive).
    Returns int or None.
    """
    base = os.path.basename(fname)
    lowered = base.lower()
    # naive parse: find 'tau' then parse following digits
    idx = lowered.find('tau')
    if idx == -1:
        return None
    # take substring after 'tau'
    tail = lowered[idx + len('tau'):]
    # remove non-digit prefix characters like '_' or '-'
    import re
    m = re.search(r"(\d+)", tail)
    if m:
        return int(m.group(1))
    return None


def find_measurement_from_filename(fname):
    """Try to extract the integer indicating the number of measurements from the filename.
    Looks for patterns like '_Measurement_15' or 'Measurement-15' (case-insensitive).
    Returns int or None.
    """
    base = os.path.basename(fname)
    lowered = base.lower()
    # naive parse: find 'measurement' then parse following digits
    idx = lowered.find('measurement')
    if idx == -1:
        return None
    # take substring after 'measurement'
    tail = lowered[idx + len('measurement'):]
    # remove non-digit prefix characters like '_' or '-'
    import re
    m = re.search(r"(\d+)", tail)
    if m:
        return int(m.group(1))
    return None


def read_two_column_file(path):
    """Reads a two-column whitespace-separated file while skipping a single header line.
    Returns x, y as 1D numpy arrays.
    """
    with open(path, 'r') as f:
        lines = [ln for ln in (l.rstrip('\n') for l in f) if ln.strip() != '']
    if not lines:
        raise ValueError(f"Empty file: {path}")
    header = lines[0]
    data_lines = lines[1:]
    # If there isn't an explicit header (i.e. first line is numeric), handle gracefully
    import re
    if re.match(r"^[\s\d.+\-eE]+$", header):
        # first line looks numeric; include it as data
        data_lines = lines
    # now load numeric data
    data = np.loadtxt(data_lines, dtype=complex)
    if data.ndim == 1:
        # single row
        if len(data) < 2:
            raise ValueError(f"File {path} does not look like two columns")
        x = np.array([data[0]])
        y = np.array([data[1]])
    else:
        x = data[:, 0]
        y = data[:, 1]
    return header, x, y


def collect_series(directory, taus=DEFAULT_TAUS):
    """Search directory for .txt files, classify them by tau and lubrication state.
    Returns a dict: results[tau][state] = list of (filename, header, x, y)
    where state is 'lubricated' or 'non_lubricated'.
    """
    pattern = os.path.join(directory, '*.txt')
    files = glob.glob(pattern)
    if not files:
        # try recursive search
        files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)
    results = defaultdict(lambda: defaultdict(list))
    for fpath in files:
        try:
            header, x, y = read_two_column_file(fpath)
        except Exception as e:
            print(f"Skipping {fpath} (couldn't read): {e}", file=sys.stderr)
            continue
        # determine lubricated vs non-lubricated by header OR filename
        low_header = header.lower()
        low_fname = os.path.basename(fpath).lower()
        is_lub = ('lubricat' in low_header) or ('lubricat' in low_fname)
        state = 'lubricated' if is_lub else 'non_lubricated'
        tau = find_tau_from_filename(fpath)
        # if not found, try parse header tokens for tau
        if tau is None:
            # try to find a token like 'tau=' in header
            import re
            m = re.search(r"tau\s*[=:_-]?\s*(\d+)", header, flags=re.IGNORECASE)
            if m:
                tau = int(m.group(1))
        # only keep expected taus (if provided)
        if taus is None or tau in taus:
            results[tau][state].append((fpath, header, x, y))
    return results


def make_color_for_tau(tau, cmap, norm):
    if tau is None:
        # fallback grey
        return '0.5'
    if tau == 0:
        return 'tab:blue'
    else:
        start_frac = 0.35  # never use the first 15% of the colormap
        color = cmap(start_frac + (1.0 - start_frac) * float(norm(tau)))
    return color  # cmap(norm(tau))


def plot_results(results, taus=DEFAULT_TAUS, savepath=False, show=True):
    plt.figure(figsize=(10, 5))

    # colormap for nonzero taus
    cmap = cm.get_cmap('Oranges')
    norm = Normalize(vmin=min([g for g in taus if g != 0]), vmax=max(taus))
    print(norm)
    # markers per state
    markers = {'non_lubricated': 'o', 'lubricated': '^'}

    plotted_any = False
    for tau in taus:
        series = results.get(tau, {})
        if not series:
            continue
        # choose color
        color = make_color_for_tau(tau, cmap, norm)
        for state_key in ['non_lubricated', 'lubricated']:
            entries = series.get(state_key, [])
            for (fname, header, x, y) in entries:
                # sort by x in case
                order = np.argsort(x)
                x2 = x[order]
                # x2 = x2[:len(x2)//2]
                y2 = y[order]
                # y2 = y2[:len(y2)//2]
                # label = f"{'Lubricated' if state_key=='lubricated' else 'Non-lubricated'}, $\\tau={tau}$"
                if tau == 0:
                    label = "Non-lubricated"
                else:
                    label = f"{'' if state_key == 'lubricated' else 'Non-lubricated'} $\\tau_{{comp}}={tau}$"
                # To avoid duplicate legend entries, only label the first time this (tau,state) appears
                show_label = True
                # check if we've already plotted same label (matplotlib handles duplicates, but we'll try)
                plt.plot(x2, y2, marker=0, linestyle='-', label=label if show_label else None,
                         markersize=0, linewidth=2.4, color=color)
                # shaded area under curve
                # try:
                #     plt.fill_between(x2, y2, alpha=0.15, facecolor=color)
                # except Exception:
                #     pass
                plotted_any = True

    if not plotted_any:
        raise RuntimeError('No series plotted -- check your directory and file naming')

    # Styling: increase font sizes for labels, ticks, legend
    label_fontsize = 28  # axis labels
    tick_fontsize = 20  # tick labels (numbers)
    legend_fontsize = 22  # legend text
    # (tweak the numbers above to taste)

    ax = plt.gca()
    ax.set_xlabel(r'$\Gamma$', fontsize=label_fontsize)
    # ax.set_ylabel(r'$E_N(\rho_{\rm SL})$', fontsize=label_fontsize)
    ax.set_ylabel(r'$W^{dec}$', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True)

    # Legend with increased font size
    legend = ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Saved figure to {savepath}")
    if show:
        # plt.savefig("TRUE_Log_Neg_+L_GammaX_thick.pdf")
        plt.savefig("TRUE_Dec_Cost_+L_GammaX_thick.pdf")
        # plt.savefig("TRUE_M100_Dec_COST_+L_TauX.pdf")
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot lubricated/non-lubricated results grouped by Tau')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing .txt files')
    parser.add_argument('--out', type=str, default=None, help='If set, save figure to this file (e.g. plot.png)')
    parser.add_argument('--taus', type=str, default=','.join(str(g) for g in DEFAULT_TAUS),
                        help='Comma-separated list of tau values to include (e.g. 0,5,15,30)')
    parser.add_argument('--no-show', action='store_true', help="Don't call plt.show() (useful for headless runs)")
    args = parser.parse_args()

    try:
        taus = [int(x) for x in args.taus.split(',') if x.strip() != '']
    except Exception:
        print('Could not parse --taus; using defaults', file=sys.stderr)
        taus = DEFAULT_TAUS
    # plt.style.use('science')
    results = collect_series(args.dir, taus=taus)
    plot_results(results, taus=taus, savepath=args.out, show=not args.no_show)


if __name__ == '__main__':
    main()
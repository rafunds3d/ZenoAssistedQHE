"""
Plotting script for lubricated / non-lubricated result files grouped by Gamma.

Usage:
    python plot_lubricated_nonlubricated_gamma.py --dir /path/to/txt/files --out plot.png

Behavior / assumptions:
- Looks for files with "Gamma" in their filename (case-insensitive) and ending with .txt.
- Reads the first line of each file as a header. If the header contains the substring
  'lubricated' (case-insensitive) the file is considered a "lubricated" series; otherwise
  it's treated as non-lubricated.
- Numeric data is expected from the second line on: two columns separated by whitespace
  (x and y). Empty lines are ignored.
- Gammas considered by default: [0,5,15,30,45,60,75,90,120]. You can override with --gammas.
- Color mapping: Gamma==0 -> blue; Gamma>0 -> colormap 'Oranges' from light (small Gamma)
  to dark (Gamma=120). Each series is plotted with markers and a light shaded fill under
  the curve.

Outputs:
- Displays the plot and optionally saves to --out filename.

"""

import argparse
import glob
import os
import sys
from collections import defaultdict
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


DEFAULT_GAMMAS = [0,2,5,10, 20, 40, 80, 160, 320] # These were the ones used in our manuscript v1
# DEFAULT_GAMMAS = [0,2]
DEFAULT_MEASUREMENTS = [100]


def find_gamma_from_filename(fname):
    """Try to extract the integer Gamma from the filename.
    Looks for patterns like '_Gamma_15' or 'Gamma-15' (case-insensitive).
    Returns int or None.
    """
    base = os.path.basename(fname)
    lowered = base.lower()
    # naive parse: find 'gamma' then parse following digits
    idx = lowered.find('gamma')
    if idx == -1:
        return None
    # take substring after 'gamma'
    tail = lowered[idx + len('gamma'):]
    # remove non-digit prefix characters like '_' or '-'
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
    if re.match(r"^[\s\d.+\-eE]+$", header):
        # first line looks numeric; include it as data
        data_lines = lines
    # now load numeric data
    data = np.loadtxt(data_lines)
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


def collect_series(directory, gammas=DEFAULT_GAMMAS):
    """Search directory for .txt files, classify them by Gamma and lubrication state.
    Returns a dict: results[gamma][state] = list of (filename, header, x, y)
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

        # detect non-lubricated explicitly first to avoid misclassification
        if re.search(r'\b(?:non|no|un)[-_ ]?lubricat', low_header) or re.search(r'\b(?:non|no|un)[-_ ]?lubricat', low_fname):
            is_lub = False
        elif 'lubricat' in low_header or 'lubricat' in low_fname:
            is_lub = True
        else:
            # default assume non-lubricated if nothing explicit
            is_lub = False
        state = 'lubricated' if is_lub else 'non_lubricated'

        gamma = find_gamma_from_filename(fpath)
        # if not found, try parse header tokens for Gamma
        if gamma is None:
            # try to find a token like 'Gamma=' in header
            m = re.search(r"gamma\s*[=:_-]?\s*(\d+)", header, flags=re.IGNORECASE)
            if m:
                gamma = int(m.group(1))
        # only keep expected gammas (if provided)
        if gammas is None or gamma in gammas:
            results[gamma][state].append((fpath, header, x, y))
    return results


def make_color_for_gamma(gamma, cmap, norm):
    if gamma is None:
        # fallback grey
        return '0.5'
    if gamma == 0:
        return 'tab:blue'
    else:
        start_frac = 0.2   # never use the first 15% of the colormap
        color = cmap(start_frac + (1.0 - start_frac) * 15 *float(norm(gamma)))
    return color #cmap(norm(gamma))


def plot_results(results, gammas=DEFAULT_GAMMAS, savepath=None, show=True):
    plt.figure(figsize=(10, 5))

    # colormap for nonzero gammas
    cmap = cm.get_cmap('Oranges')
    norm = Normalize(vmin=min([g for g in gammas if g != 0]), vmax=max(gammas))
    
    # markers per state
    markers = {'non_lubricated': 'o', 'lubricated': '^'}

    plotted_any = False
    for gamma in gammas:
        series = results.get(gamma, {})
        if not series:
            continue
        # choose color
        color = make_color_for_gamma(gamma, cmap, norm)
        for state_key in ['non_lubricated', 'lubricated']:
            entries = series.get(state_key, [])
            for (fname, header, x, y) in entries:
                # sort by x in case
                order = np.argsort(x)
                x2 = x[order]
                y2 = y[order]
                #label = f"{'Lubricated' if state_key=='lubricated' else 'Non-lubricated'}, $\\Gamma={gamma}$"
                if gamma == 0:
                    label = "Non-lubricated"
                else:
                    label = f"{'' if state_key=='lubricated' else 'Non-lubricated'} $\\Gamma={gamma}$"
                # To avoid duplicate legend entries, only label the first time this (gamma,state) appears
                show_label = True
                # check if we've already plotted same label (matplotlib handles duplicates but we'll try)
                if gamma==0:
                    plt.plot(x2, y2, marker=markers[state_key], linestyle='-.', label=label if show_label else None, markersize=0, linewidth=2.5, color=color, zorder=10)
                else:
                    plt.plot(x2, y2, marker=markers[state_key], linestyle='-', label=label if show_label else None, markersize=0, linewidth=2.5, color=color)
                # shaded area under curve
                # try:
                #    plt.fill_between(x2, y2, alpha=0.15, facecolor=color)
                # except Exception:
                #    pass
                plotted_any = True

    if not plotted_any:
        raise RuntimeError('No series plotted -- check your directory and file naming')

    # Styling: increase font sizes for labels, ticks, legend
    label_fontsize = 26    # axis labels
    tick_fontsize = 20     # tick labels (numbers)
    legend_fontsize = 16   # legend text
    # (tweak the numbers above to taste)

    ax = plt.gca()
    ax.set_xlabel(r'$\tau_{\rm comp}$', fontsize=label_fontsize)
    ax.set_ylabel(r'$C_{\ell_1}$', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True)

    # Legend with increased font size
    legend = ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Saved figure to {savepath}")
    if show:
        plt.show()
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Plot lubricated/non-lubricated results grouped by Gamma')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing .txt files')
    parser.add_argument('--out', type=str, default=None, help='If set, save figure to this file (e.g. plot.png)')
    parser.add_argument('--gammas', type=str, default=','.join(str(g) for g in DEFAULT_GAMMAS), help='Comma-separated list of Gamma values to include (e.g. 0,5,15,30)')
    parser.add_argument('--no-show', action='store_true', help="Don't call plt.show() (useful for headless runs)")
    args = parser.parse_args()

    try:
        gammas = [int(x) for x in args.gammas.split(',') if x.strip()!='']
    except Exception:
        print('Could not parse --gammas; using defaults', file=sys.stderr)
        gammas = DEFAULT_GAMMAS

    results = collect_series(args.dir, gammas=gammas)
    plot_results(results, gammas=gammas, savepath=args.out, show=not args.no_show)


if __name__ == '__main__':
    main()

"""
plot_otto_metrics.py (extended)

Reads multiple result .txt files (e.g. otto_lubricated_results_Gamma_0_measurements_0.txt)
and produces three PDF plots:
  <out_prefix>_eta.pdf
  <out_prefix>_power.pdf
  <out_prefix>_W_out.pdf
"""

import argparse
import glob
import os
import re
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import math
import sys

# DEFAULT_GAMMAS = [0, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 2, 2.100 , 2.200, 2.300, 2.400, 2.500, 2.600, 3 , 4, 5, 6, 7, 8, 9, 10 , 50]
DEFAULT_GAMMAS = [0, 1.2, 2.2, 8, 20] # That's our choice for the paper
# DEFAULT_GAMMAS = [0, 1.4, 2.2, 20]
DEFAULT_MEASUREMENTS = [0, 40, 200]

def find_number_after_keyword(fname, keyword):
    """
    Find a numeric token (int or float, with optional exponent) after the keyword in filename.
    Returns a float (or None if not found).
    Examples matched: 1, 1.200, -0.5, 3e2, 1.2E-3
    """
    base = os.path.basename(fname).lower()
    idx = base.find(keyword.lower())
    if idx == -1:
        return None
    tail = base[idx + len(keyword):]
    # match optional sign, digits, optional decimal part, optional exponent
    m = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", tail)
    if m:
        try:
            val = float(m.group(1))
            return val
        except Exception:
            return None
    return None

def find_gamma_from_filename(fname):
    # support tokens like 'gamma', 'Gamma', etc.
    return find_number_after_keyword(fname, 'gamma')

def find_int_after_keyword(fname, keyword):
    # keep original int helper (used for measurements)
    base = os.path.basename(fname).lower()
    idx = base.find(keyword.lower())
    if idx == -1:
        return None
    tail = base[idx + len(keyword):]
    m = re.search(r"(\d+)", tail)
    if m:
        return int(m.group(1))
    return None

def find_measurement_from_filename(fname):
    # accept 'measurement' or 'measurements'
    base = os.path.basename(fname).lower()
    idx = base.find('measurement')
    if idx == -1:
        return None
    tail = base[idx + len('measurement'):]
    m = re.search(r"(\d+)", tail)
    if m:
        return int(m.group(1))
    return None

def is_lubricated_from_name_or_header(fname, header_text):
    """
    Determine whether a file is 'lubricated' by looking at filename or header.
    Rules:
      - If the name/header contains an explicit negative token preceding 'lubricat'
        (e.g. non_lubricated, non-lubricated, nonlubricated, not_lubricated, nolubricated),
        return False.
      - Otherwise if 'lubricat' appears as a token in name/header, return True.
      - Otherwise return False.
    This avoids mis-classifying 'non_lubricated' as lubricated.
    """
    lowname = os.path.basename(fname).lower()
    lowheader = (header_text or '').lower()

    # patterns that explicitly say "non" / "no" / "not" / "un" + optional separator + "lubricat"
    neg_pattern = re.compile(r'(?:^|[_\-\s])(?:non|no|not|un)[_\-]?lubricat')
    # also catch cases like "nonlubricated" without separator
    if neg_pattern.search(lowname) or neg_pattern.search(lowheader) or 'nonlubricat' in lowname or 'nonlubricat' in lowheader:
        return False

    # positive: 'lubricat' as a token (with optional separator) or at start/end
    pos_pattern = re.compile(r'(?:^|[_\-\s])lubricat')
    if pos_pattern.search(lowname) or pos_pattern.search(lowheader) or 'lubricat' in lowname or 'lubricat' in lowheader:
        return True

    return False

def compute_otto_eta(omega, Omega0):
    """Return Otto efficiency 1 - omega/Omega where Omega = sqrt(omega^2 + Omega0^2)."""
    Omega = math.sqrt(omega**2 + Omega0**2)
    if Omega == 0:
        return float('nan')
    return 1.0 - (omega / Omega)

def parse_file(path):
    """
    Attempt to parse file, returning (header_text, data_dict)
    data_dict maps column name -> numpy array
    If header line contains column names (like '# tau eta power W_out Q_hot' or a plain first line),
    use those names. Otherwise assume default order:
        ['tau', 'eta', 'power', 'W_out', 'Q_hot']
    """
    with open(path, 'r') as f:
        raw_lines = [ln.rstrip('\n') for ln in f]

    # drop empty lines
    lines = [ln for ln in raw_lines if ln.strip() != '']
    if not lines:
        raise ValueError(f"File {path} is empty or only whitespace")

    # detect header
    first = lines[0].strip()
    header_text = ""
    data_lines = None

    # If first line starts with '#', treat it as header (strip leading '#')
    if first.startswith('#'):
        header_text = first.lstrip('#').strip()
        data_lines = lines[1:]
    else:
        # if first line contains any non-numeric alphabetic chars, treat as header
        if re.search(r"[A-Za-z]", first):
            header_text = first
            data_lines = lines[1:]
        else:
            # first line numeric -> no header
            header_text = ""
            data_lines = lines

    if not data_lines:
        raise ValueError(f"No numeric data found in {path}")

    # turn data_lines into numpy array robustly
    try:
        data = np.loadtxt(data_lines)
    except Exception:
        # fallback: try genfromtxt to handle inconsistent whitespace
        data = np.genfromtxt(data_lines)
    if data.ndim == 1:
        # single row -> shape (ncols,)
        if data.size < 2:
            raise ValueError(f"File {path} doesn't have enough columns")
        data = data.reshape((1, data.size))

    # build column names
    if header_text:
        # split header by whitespace and common separators
        tokens = re.split(r"[,\t\s;]+", header_text.strip())
        tokens = [re.sub(r"[^\w]+", "_", t).strip() for t in tokens if t.strip() != ""]
        if len(tokens) != data.shape[1]:
            tokens = None
    else:
        tokens = None

    if tokens is None:
        default_names = ['tau', 'eta', 'power', 'W_out', 'Q_hot']
        if data.shape[1] > len(default_names):
            extra = [f"col_{i}" for i in range(len(default_names), data.shape[1])]
            colnames = default_names + extra
        else:
            colnames = default_names[:data.shape[1]]
    else:
        colnames = tokens

    # create dictionary
    data_dict = {}
    for i, name in enumerate(colnames):
        data_dict[name] = data[:, i]

    return header_text, data_dict

def collect_files(directory, gammas=None, measurements=None):
    """
    Search directory for .txt files (non-recursive by default, fallback to recursive).
    Returns dict: grouped[gamma][state] -> list of entries
    each entry is dict: { 'path', 'header', 'cols', 'measurements' }
    If `measurements` is provided (list or set), files whose parsed measurement integer is
    not in that set will be skipped. To include files without a measurement token, include None
    in the measurements list (or set measurements=None to disable filtering).
    """
    pattern = os.path.join(directory, '*.txt')
    files = glob.glob(pattern)
    # fallback to recursive if none found
    if not files:
        files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    grouped = defaultdict(lambda: defaultdict(list))
    for f in files:
        try:
            header, cols = parse_file(f)
        except Exception as e:
            print(f"Skipping {f}: couldn't parse ({e})", file=sys.stderr)
            continue
        gamma = find_gamma_from_filename(f)  # now returns float or None
        meas = find_measurement_from_filename(f)

        # apply measurement-level filtering if requested
        if measurements is not None:
            # if file has no meas and user requested only specific measurements and did not include None -> skip
            if meas is None:
                if (None not in measurements):
                    continue
            else:
                if meas not in measurements:
                    continue

        # apply gamma-level filtering if requested
        if gammas is not None and gamma is not None:
            # use math.isclose for robust float comparison
            matched = False
            for gg in gammas:
                if gg is None:
                    continue
                try:
                    if math.isclose(float(g), float(gamma), rel_tol=1e-9, abs_tol=1e-12):
                        matched = True
                        break
                except Exception:
                    # last-resort equality
                    if gg == gamma:
                        matched = True
                        break
            if not matched:
                continue

        state = 'lubricated' if is_lubricated_from_name_or_header(f, header) else 'non_lubricated'
        grouped[gamma][state].append({
            'path': f,
            'header': header,
            'cols': cols,
            'measurements': meas
        })
    # sort entries by measurements when available
    for gamma in list(grouped.keys()):
        for state in grouped[gamma]:
            grouped[gamma][state].sort(key=lambda e: (e['measurements'] if e['measurements'] is not None else -1))
    return grouped

def make_color_map(gamma_list):
    """
    Returns a dict gamma -> color using a sequential colormap for gamma>0,
    and a fixed blue for gamma==0 or None.
    gamma_list should be a list of unique gammas (None allowed).
    """
    gammas = [g for g in gamma_list if (g is not None and g != 0)]
    if gammas:
        vmin = min(gammas)
        vmax = max(gammas)
    else:
        vmin, vmax = 1, 1
    cmap = cm.get_cmap('Oranges')
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize(vmin=vmin, vmax=vmin+1)
    color_for = {}
    for g in gamma_list:
        if g is None or g == 0:
            color_for[g] = 'tab:blue'
        else:
            color_for[g] = cmap(0.25 + 2 * norm(g))
    return color_for

def plot_metric(grouped, metric_key, out_path, otto_eta=None,
                xlabel=r'$\tau_{\rm}$', ylabel=None, show=True, save_png=False,
                measurement_mode='latest', xscale='linear', yscale='linear'):
    """
    Plot a metric (eta/power/W_out) from grouped data.
    measurement_mode: 'latest'|'all'|'specific' (specific means grouped already filtered by collect_files)
    xscale: 'linear' or 'log' (log requires all x>0; otherwise falls back to linear with a warning)
    """
    # Build sorted list of gammas preserving numeric order with None at the end
    gammas = sorted([g for g in grouped.keys() if g is not None])
    if None in grouped:
        gammas.append(None)
    if not gammas:
        raise RuntimeError("No data series found to plot")

    color_map = make_color_map(gammas)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    markers = ['o', '^', 's', 'D', 'v', '>', '<', 'p', '*', 'h']
    marker_idx_map = {}

    # track global tau and y ranges
    global_x_min = None
    global_x_max = None
    global_y_min = None
    global_y_max = None
    
    for g in gammas:
        series_by_state = grouped.get(g, {})
        for state in ['non_lubricated', 'lubricated']:
            entries = series_by_state.get(state, [])
            if not entries:
                continue

            if measurement_mode == 'latest':
                entry_list = [entries[-1]]
            elif measurement_mode == 'all':
                entry_list = entries
            elif measurement_mode == 'specific':
                entry_list = entries if len(entries) > 1 else entries
            else:
                entry_list = entries

            for entry in entry_list:
                cols = entry['cols']
                if 'tau' in cols:
                    tau = np.asarray(cols['tau'])
                else:
                    tau = np.asarray(list(cols.values())[0])
                if metric_key not in cols:
                    print(f"File {entry['path']} missing '{metric_key}' -> skipping", file=sys.stderr)
                    continue
                y = np.asarray(cols[metric_key])

                order = np.argsort(tau)
                x2 = tau[order]
                y2 = y[order]

                # update global tau range
                if x2.size > 0:
                    xmin = float(np.min(x2)); xmax = float(np.max(x2))
                    global_x_min = xmin if (global_x_min is None) else min(global_x_min, xmin)
                    global_x_max = xmax if (global_x_max is None) else max(global_x_max, xmax)

                # update global y range
                if y2.size > 0:
                    ymin = float(np.min(y2)); ymax = float(np.max(y2))
                    global_y_min = ymin if (global_y_min is None) else min(global_y_min, ymin)
                    global_y_max = ymax if (global_y_max is None) else max(global_y_max, ymax)

                meas = entry.get('measurements', None)
                # meas_label = f", m={meas}" if meas is not None and measurement_mode in ('all','specific') else "" # this is by default set to not appear
                meas_label = f", $n$={meas}" if meas is not None else "" # always shows the measurements
                gamma_label = f"$\\Gamma$={g}" if g is not None else "Γ=?"
                if state == 'lubricated':
                    label = f"{'Lubricated'} ({gamma_label}{meas_label})"
                else:
                    label = f"{'Non-lubricated'} ({gamma_label}{meas_label})"

                color = color_map.get(g, '0.5')
                mk_key = (g, state)
                if mk_key not in marker_idx_map:
                    marker_idx_map[mk_key] = len(marker_idx_map)
                marker = markers[marker_idx_map[mk_key] % len(markers)]
                linestyle = '-' if state == 'lubricated' else '-.'

                # put the blue (gamma=0 / gamma=None) curve on top of everything else
                z = 20 if color == 'tab:blue' else 5

                ax.plot(x2, y2,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markevery=max(1, len(x2)//12),
                        linewidth=2.4,
                        markersize=0,
                        zorder=z)

    # decide xscale (log requires all x > 0)
    if xscale == 'log':
        if (global_x_min is None) or (global_x_min <= 0):
            print(f"Warning: requested log x-scale for '{metric_key}' but found non-positive x values "
                  f"(min={global_x_min}). Falling back to linear.", file=sys.stderr)
        else:
            ax.set_xscale('log')

    # decide yscale (log requires all y > 0)
    if yscale == 'log':
        if (global_y_min is None) or (global_y_min <= 0):
            print(f"Warning: requested log y-scale for '{metric_key}' but found non-positive y values "
                  f"(min={global_y_min}). Falling back to linear.", file=sys.stderr)
        else:
            ax.set_yscale('log')

    # draw otto line (after plotting series, so it appears in legend and on top)
    if otto_eta is not None and (global_x_min is not None) and (global_x_max is not None):
        ax.hlines(otto_eta, global_x_min, global_x_max,
                  colors='k', linestyles='--', linewidth=2,
                  label=f"Otto eff: $1-\\omega/\\Omega$ = {otto_eta:.3f}", zorder=10)
    if metric_key == 'eta':
        omega = 1.0
        Omega0 = 3.01105
        Omega_ref = (omega**2 + Omega0**2)**0.5
        eta_otto = 1.0 - omega/Omega_ref

        plt.axhline(eta_otto, color='k', linestyle='--',
                    linewidth=1, label=f"Otto eff: $1-\\omega/\\Omega$ = {eta_otto:.3f}", zorder=3)

    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel if ylabel is not None else metric_key, fontsize=26)
    ax.grid(True, which='both', alpha=0.25)
    ax.tick_params(axis='both', labelsize=26)
    #ax.legend(fontsize=24, loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {out_path}")
    if save_png:
        png_path = os.path.splitext(out_path)[0] + '.png'
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        print(f"Saved PNG: {png_path}")
    if show:
        plt.show()
    plt.close()

def parse_int_list_arg(s):
    """Parse a comma-separated list of ints; 'None' (case-ins) maps to Python None token in the list."""
    if s is None:
        return None
    parts = [t.strip() for t in s.split(',') if t.strip() != '']
    if not parts:
        return None
    out = []
    for p in parts:
        if p.lower() == 'none':
            out.append(None)
        else:
            try:
                out.append(int(p))
            except Exception:
                # ignore unparsable token
                pass
    return out if out else None

def parse_number_list_arg(s):
    """
    Parse a comma-separated list of numbers (floats) or 'None'.
    Returns list of floats and/or None, or None if input empty/None/unparsable.
    """
    if s is None:
        return None
    parts = [t.strip() for t in s.split(',') if t.strip() != '']
    if not parts:
        return None
    out = []
    for p in parts:
        if p.lower() == 'none':
            out.append(None)
        else:
            try:
                out.append(float(p))
            except Exception:
                # ignore unparsable token
                pass
    return out if out else None

def main():
    parser = argparse.ArgumentParser(description="Plot metrics (eta, power, W_out) from Otto result .txt files")
    parser.add_argument('--xscale', choices=['linear','log'], default='linear',
                        help="Default x-axis scale for plots ('linear' or 'log').")
    parser.add_argument('--xscale-eta', choices=['linear','log'], default=None,
                        help="x-axis scale for the eta plot (overrides --xscale if set).")
    parser.add_argument('--xscale-power', choices=['linear','log'], default=None,
                        help="x-axis scale for the power plot (overrides --xscale if set).")
    parser.add_argument('--xscale-W_out', choices=['linear','log'], default=None,
                        help="x-axis scale for the W_out plot (overrides --xscale if set).")
    parser.add_argument('--yscale', choices=['linear','log'], default='linear',
                        help="Default y-axis scale for plots ('linear' or 'log').")
    parser.add_argument('--yscale-eta', choices=['linear','log'], default='linear',
                        help="y-axis scale for the eta plot (overrides --yscale if set).")
    parser.add_argument('--yscale-power', choices=['linear','log'], default='linear',
                        help="y-axis scale for the power plot (overrides --yscale if set).")
    parser.add_argument('--yscale-W_out', choices=['linear','log'], default='linear',
                        help="y-axis scale for the W_out plot (overrides --yscale if set).")
    parser.add_argument('--dir', '-d', type=str, default='.', help='Directory containing .txt result files')
    parser.add_argument('--out-prefix', '-o', type=str, default='otto_plots', help='Output filename prefix')
    parser.add_argument('--gammas', type=str, default=','.join(str(g) for g in DEFAULT_GAMMAS),
                        help='Comma-separated list of Gamma values to include (e.g. 0,5,15). Use "None" to include files without Gamma.')
    parser.add_argument('--measurements', type=str, default=','.join(str(m) for m in DEFAULT_MEASUREMENTS),
                        help='Comma-separated list of measurement counts to include (e.g. 250,500). Use "None" to include files without measurement tag. Use empty string to disable measurement filtering.')
    parser.add_argument('--measurement-mode', type=str, choices=['latest','all','specific'], default='latest',
                        help="How to treat multiple measurement-series per (Gamma,state): 'latest' (default), 'all', or 'specific' (requires --measurements).")
    parser.add_argument('--no-show', action='store_true', help="Don't call plt.show() (useful for headless runs)")
    parser.add_argument('--save-png', action='store_true', help="Also save PNG versions of the figures")
    parser.add_argument('--otto', action='store_true', help="Plot Otto reference efficiency as a dashed horizontal line")
    parser.add_argument('--omega', type=float, default=1.0, help="omega parameter for Otto efficiency (default 1)")
    parser.add_argument('--Omega0', type=float, default=3.01105, help="Omega0 parameter for Otto efficiency (default 3.01105)")

    args = parser.parse_args()
    # compute otto eta if requested
    otto_eta_val = compute_otto_eta(args.omega, args.Omega0) if args.otto else None

    # parse gammas (now allowing floats or "None")
    gammas = None
    parsed_gammas = parse_number_list_arg(args.gammas)
    if parsed_gammas is None:
        # fallback to DEFAULT_GAMMAS but coerce to floats where appropriate
        gammas = [float(g) if not (g is None) else None for g in DEFAULT_GAMMAS]
    else:
        gammas = parsed_gammas

    # parse measurements: allow "" (empty) meaning "no measurement filtering" (i.e. measurements=None)
    measurements = None
    if args.measurements is not None and args.measurements.strip() != '':
        measurements = parse_int_list_arg(args.measurements)
        # measurements may be None if parsing failed -> fall back to DEFAULT_MEASUREMENTS
        if measurements is None:
            measurements = DEFAULT_MEASUREMENTS

    # If mode is 'specific' but no measurements were given, warn and fall back to latest
    if args.measurement_mode == 'specific' and measurements is None:
        print("Warning: --measurement-mode specific selected but no --measurements provided; falling back to 'latest' mode.", file=sys.stderr)
        measurement_mode = 'latest'
    else:
        measurement_mode = args.measurement_mode

    grouped = collect_files(args.dir, gammas=gammas, measurements=measurements)
    if not grouped:
        raise RuntimeError("No valid data files found in the specified directory (after filtering)")

    # eta
    eta_out = f"{args.out_prefix}_eta.pdf"
    eta_xscale = args.xscale_eta if args.xscale_eta is not None else args.xscale
    eta_yscale = args.yscale_eta if args.yscale_eta is not None else args.yscale
    plot_metric(grouped, 'eta', eta_out,
                otto_eta=otto_eta_val,
                xlabel=r'$\tau_{\rm}$',
                ylabel=r'Efficiency',
                show=(not args.no_show),
                save_png=args.save_png,
                measurement_mode=measurement_mode,
                xscale=eta_xscale,
                yscale=eta_yscale)

    # power
    power_out = f"{args.out_prefix}_power.pdf"
    power_xscale = args.xscale_power if args.xscale_power is not None else args.xscale
    power_yscale = args.yscale_power if args.yscale_power is not None else args.yscale
    plot_metric(grouped, 'power', power_out,
                xlabel=r'$\tau_{\rm}$',
                ylabel=r'Power',
                show=(not args.no_show),
                save_png=args.save_png,
                measurement_mode=measurement_mode,
                xscale=power_xscale,
                yscale=power_yscale)

    # W_out
    W_out_path = f"{args.out_prefix}_W_out.pdf"
    W_xscale = args.xscale_W_out if args.xscale_W_out is not None else args.xscale
    W_yscale = args.yscale_W_out if args.yscale_W_out is not None else args.yscale
    plot_metric(grouped, 'W_out', W_out_path,
                xlabel=r'$\tau_{\rm}$',
                ylabel=r'Extracted work',
                show=(not args.no_show),
                save_png=args.save_png,
                measurement_mode=measurement_mode,
                xscale=W_xscale,
                yscale=W_yscale)

if __name__ == '__main__':
    main()
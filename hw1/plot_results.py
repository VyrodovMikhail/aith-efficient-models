"""Plot all measured grid points alongside calibrated curves; report memory errors."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrate import error_metrics, load_measurements
from equations import energy, latency, memory


def make_plots(rows, theta, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = sorted({r['S'] for r in rows})
    bmin, bmax = min(r['B'] for r in rows), max(r['B'] for r in rows)
    batches = np.geomspace(bmin, bmax, 200)
    memory_metrics = {}
    for split, flag in [('train', False), ('validation', True)]:
        selected = [r for r in rows if r['status'] == 'OK'
                    and r['memory_bytes'] is not None and r['is_validation'] == flag]
        memory_metrics[split] = error_metrics(
            [r['memory_bytes'] for r in selected],
            memory([r['S'] for r in selected], [r['B'] for r in selected]))
    specs = [
        ('latency', 'latency_seconds', lambda s, b: latency(s, b, theta['latency']), 1000, 'Latency (ms)'),
        ('energy', 'energy_joules', lambda s, b: energy(s, b, theta['energy']), 1, 'Energy (J / batch forward)'),
        ('memory', 'memory_bytes', memory, 1 / 2**20, 'Peak allocated memory (MiB)'),
    ]
    for name, field, predict, scale, label in specs:
        fig, axes = plt.subplots((len(sizes) + 2)//3, 3, figsize=(15, 3.2*((len(sizes)+2)//3)),
                                 squeeze=False)
        for ax, s in zip(axes.flat, sizes):
            ax.plot(batches, predict(s, batches)*scale, color='#244f8a', label='Predicted')
            for flag, marker, color, split in [(False, 'o', '#d17b00', 'Measured: train'),
                                              (True, '^', '#12826b', 'Measured: validation')]:
                selected = [r for r in rows if r['S'] == s and r['status'] == 'OK'
                            and r[field] is not None and r['is_validation'] == flag]
                ax.scatter([r['B'] for r in selected], [r[field]*scale for r in selected],
                           s=22, marker=marker, color=color, label=split, zorder=3)
            for row in rows:
                if row['S'] == s and row['status'] == 'OOM':
                    # An OOM has no measured y-value: locate it on the axis, not the curve.
                    ax.plot(row['B'], .04, 'rx', transform=ax.get_xaxis_transform())
                    ax.text(.98, .12, f"OOM: B={row['B']}", transform=ax.transAxes,
                            ha='right', color='firebrick', fontsize=8)
            ax.set(xscale='log', yscale='log', xlabel='Batch size B', ylabel=label,
                   title=f'S = {s} px')
            ax.grid(True, which='major', alpha=.2)
        for ax in list(axes.flat)[len(sizes):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .97))
        fig.suptitle(f'{label}: measurements and analytical prediction', y=.995)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output / f'{name}.png', dpi=150)
        plt.close(fig)
    return {'memory': memory_metrics, 'oom': [
        {'S': r['S'], 'B': r['B'], 'baseline_bytes': memory(r['S'], r['B']),
         'baseline_GiB': memory(r['S'], r['B'])/2**30}
        for r in rows if r['status'] == 'OOM']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, default=Path('results/measurements.csv'))
    parser.add_argument('--theta', type=Path, default=Path('results/theta.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/figures'))
    args = parser.parse_args()
    rows = load_measurements(args.measurements)
    summary = make_plots(rows, json.loads(args.theta.read_text()), args.output_dir)
    (args.output_dir.parent / 'validation_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved three figures to {args.output_dir}')


if __name__ == '__main__':
    main()

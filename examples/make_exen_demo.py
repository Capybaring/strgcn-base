"""Generate independent synthetic splits for a runnable smoke experiment.

This is NOT a real-data performance benchmark.
"""
import argparse
from pathlib import Path
import numpy as np


def make_split(seed, count):
    rng = np.random.default_rng(seed)
    rows = []
    for series in range(count):
        offset = rng.uniform(-.2, .2)
        for time in [0., .4, 1.1, 1.8, 2.2, 2.8]:
            x = np.sin(time) + offset
            for var, value in [(10, x), (20, .5 * x + .2 * time), (30, np.cos(time))]:
                rows.append([series, var, time, value])
    return np.asarray(rows, dtype=float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data/exen_demo.npz')
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_data=make_split(1, 24), valid_data=make_split(2, 8),
             test_data=make_split(3, 8))
    print(f'Synthetic demo written to {path}')

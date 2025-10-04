#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def pair_bins_plot(d_true: np.ndarray, d_est: np.ndarray, bins: int, out_path: str):
    iu = np.triu_indices(d_true.shape[0], 1)
    x = d_true[iu]
    y = d_est[iu]
    # target is y^2 vs alpha*x+beta? Here just error |y-x| grouped by x
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    qs[0] = x.min() - 1e-12
    qs[-1] = x.max() + 1e-12
    rmse = []
    centers = []
    for k in range(bins):
        m = (x > qs[k]) & (x <= qs[k + 1])
        if m.any():
            err = (y[m] - x[m]) ** 2
            rmse.append(np.sqrt(err.mean()))
            centers.append(0.5 * (qs[k] + qs[k + 1]))
    plt.figure(figsize=(6, 4))
    plt.plot(centers, rmse, marker='o')
    plt.xlabel('d_true (bin center)')
    plt.ylabel('RMSE')
    plt.title('Pair RMSE by distance bin')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--true', required=True, help='npy path of true distance matrix (N,N)')
    ap.add_argument('--est', required=True, help='npy path of estimated distance matrix (N,N)')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    d_true = np.load(args.true)
    d_est = np.load(args.est)
    pair_bins_plot(d_true, d_est, args.bins, args.out)


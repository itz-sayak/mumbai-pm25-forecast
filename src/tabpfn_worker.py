"""TabPFN-TS worker (runs in its own venv: tabpfn-time-series, LOCAL mode on our GPU; no cloud client).
Called by tsfm.tabpfn_forecast through a subprocess, like timeh2o_worker.py.

python tabpfn_worker.py <in.npz> <out.npy>
in.npz: y [N, C] (NaN = missing), kc [N, K, C] known covariates over the context, kf [N, K, H] over the horizon,
        ts0 [N] origin timestamps (int64 ns, hour-end IST), names [K]
out.npy: [N, H] point forecast (mean of the predictive distribution, RMSE-optimal)
"""
import os
import sys

import numpy as np
import pandas as pd


def main(fi, fo):
    from tabpfn_time_series import TabPFNTSPipeline
    z = np.load(fi, allow_pickle=True)
    y, kc, kf, ts0, names = z["y"], z["kc"], z["kf"], z["ts0"], list(z["names"])
    N, C = y.shape
    H = kf.shape[-1]
    ctx, fut = [], []
    for i in range(N):
        t0 = pd.Timestamp(int(ts0[i]))
        tc = pd.date_range(end=t0, periods=C, freq="h")
        tf = pd.date_range(start=t0 + pd.Timedelta(hours=1), periods=H, freq="h")
        c = pd.DataFrame({"item_id": i, "timestamp": tc, "target": y[i]})
        f = pd.DataFrame({"item_id": i, "timestamp": tf})
        for k, nm in enumerate(names):
            c[nm] = kc[i, k]
            f[nm] = kf[i, k]
        ctx.append(c)
        fut.append(f)
    ctx = pd.concat(ctx, ignore_index=True)
    fut = pd.concat(fut, ignore_index=True)
    ckpt = os.environ.get("TABPFN_CKPT")  # e.g. the time-series fine-tuned TabPFN-TS-3 checkpoint
    kw = {"tabpfn_model_config": {"model_path": ckpt}} if ckpt else {}
    pipe = TabPFNTSPipeline(max_context_length=C, tabpfn_output_selection="mean", **kw)
    P = pipe.predict_df(ctx, future_df=fut).reset_index()
    P = P.sort_values(["item_id", "timestamp"])
    out = P["target"].to_numpy(dtype=np.float32).reshape(N, H)
    np.save(fo, out)
    print("tabpfn worker ok", out.shape, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("SKRUB_DATA_DIRECTORY", "/dev/shm/pm25_work/skrub")
    main(sys.argv[1], sys.argv[2])

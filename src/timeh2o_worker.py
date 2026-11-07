"""TimeH2O-v1 worker (runs in the separate Python 3.12 env; h2ofm needs Python >= 3.12).
python timeh2o_worker.py <in.npz> <out.npy>
in.npz: y (B, T) float32 with NaN for missing; optional fc (B, F, T + H) known-future covariates; H (horizon).
out.npy: (B, H) mean of the 9 trained quantiles (0.1..0.9) ~ RMSE-oriented point forecast.
HF token for the gated checkpoint from env HF_TOKEN.
"""
import os
import sys

import numpy as np
from h2ofm import TimeH2O

z = np.load(sys.argv[1])
y, H = z["y"], int(z["H"])
fc = z["fc"] if "fc" in z.files else None
m = TimeH2O.from_pretrained("h2oai/timeh2o-v1", token=os.environ.get("HF_TOKEN"), batch_size=256)
out = m.forecast_many(list(y), H, future_covariates=list(fc) if fc is not None else None, clip_nonnegative=True)
P = np.stack([np.asarray(f.quantiles, dtype=np.float32).reshape(H, -1).mean(-1) for f in out])
np.save(sys.argv[2], P)
print("timeh2o", P.shape, float(np.nanmean(P)))

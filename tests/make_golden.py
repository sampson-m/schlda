"""Regenerate tests/data/golden_v040.npz. Run on the v0.4.0 commit (008c4aa): python tests/make_golden.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fit import simulate  # noqa: E402
import schlda  # noqa: E402

RUN = dict(n_loops=60, burn_in=30, thin=5, seed=11, verbose=False)
CASES = {
    "random_m1_t1": dict(n_activity_topics=1, init="random", n_threads=1),
    "random_m2_t2": dict(n_activity_topics=2, init="random", n_threads=2),
    "random_m0_t1": dict(n_activity_topics=0, init="random", n_threads=1),
    "data_m1_t1": dict(n_activity_topics=1, init="data_driven", n_threads=1),
}
# v0.4.0 fits used alpha 0.1 on every topic; 0.6.0 defaults to alpha_identity = 1, so the golden runs pass it.
# The v0.4.0 cases with a depth-scaled or per-cell prior (data_m2_t2_kappa, data_m3_t1_vec) are no longer
# expressible and are not checked.
RUN = dict(RUN, alpha_identity=0.1)


def run(name, out):
    a, _, _ = simulate()
    kw = dict(CASES[name])
    m = kw.pop("n_activity_topics")
    res = schlda.fit_hlda(a, "cell_type", m, output_dir=Path(out) / name, **RUN, **kw)
    return res.theta.to_numpy(), res.beta.to_numpy()


if __name__ == "__main__":
    arrays = {}
    with tempfile.TemporaryDirectory() as d:
        for name in CASES:
            arrays[f"{name}_theta"], arrays[f"{name}_beta"] = run(name, d)
    out = Path(__file__).resolve().parent / "data" / "golden_v040.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({len(CASES)} cases)")

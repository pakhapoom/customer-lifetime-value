"""Step 3 - holdout evaluation on the test customers.

For each test customer, the realised discounted reward over the whole data window
(T = 53 months: the 36 trained on plus 17 more) is compared with the model's
T-month value V_T of their state at the snapshot, grouped by (segment, initial
state). Three estimates are compared:

  1. normal        normal rewards, normal transitions
  2. conservative  conservative (geometric-mean) rewards, normal transitions
  3. p - 1SE       normal rewards, every s -> s' rate lowered by one Poisson SE

    uv run src/evaluate.py
"""

import numpy as np
import pandas as pd

import mdp

MIN_N = 30  # groups with at least this many test customers are "reliable"

products, customers, snapshot, transactions = mdp.load_tables()
model = np.load("data/model.npz")
panel = mdp.build_panel(products, customers, snapshot, transactions)
T = panel["state"].shape[1] - 1
test = ~mdp.split(customers["customer_id"])

estimates = {
    "normal": mdp.finite_horizon(model["R_normal"], model["P"], T),
    "conservative": mdp.finite_horizon(model["R_conservative"], model["P"], T),
    "p - 1SE": mdp.finite_horizon(model["R_normal"], model["P_lower"], T),
}

# Realised: sum_t gamma^t x (that month's reward), t = months since the journey start.
ev = panel["events"]
pv = np.bincount(ev["c"], weights=ev["reward"] * mdp.GAMMA ** ev["m"], minlength=len(customers))
g, s0 = panel["segment"], panel["state"][:, 0]
keep = test & (model["n_s"][g, s0] > 0)  # initial state has an estimate
print(f"test customers {test.sum():,} | matched to a trained state {keep.sum():,} | T = {T} months\n")

df = pd.DataFrame({"g": g[keep], "s": s0[keep], "realised": pv[keep]})
obs = df.groupby(["g", "s"])["realised"].agg(n_test="size", observed="mean", sd="std").reset_index()
obs["SE"] = obs["sd"].fillna(0.0) / np.sqrt(obs["n_test"])
obs["lower"], obs["upper"] = obs["observed"] - 1.96 * obs["SE"], obs["observed"] + 1.96 * obs["SE"]
for name, V in estimates.items():
    obs[name] = V[obs["g"], obs["s"]]
obs["segment"] = np.array(model["segments"])[obs["g"]]
reliable = (obs["n_test"] >= MIN_N) & (obs["SE"] > 0)


def spearman(x, y):
    return x.rank().corr(y.rank()) if len(x) >= 2 else np.nan


def summary(d, r):
    rows = []
    for name in estimates:
        z = (d.loc[r, name] - d.loc[r, "observed"]) / d.loc[r, "SE"]
        rows.append(
            {
                "estimate": name,
                "groups": len(d),
                f"n>={MIN_N}": int(r.sum()),
                "L_T std RMS": np.sqrt(np.mean(z**2)) if r.any() else np.nan,
                "spearman": spearman(d[name], d["observed"]),
                f"spearman n>={MIN_N}": spearman(d.loc[r, name], d.loc[r, "observed"]),
                "below 95%": int((d.loc[r, name] < d.loc[r, "lower"]).sum()),
                "above 95%": int((d.loc[r, name] > d.loc[r, "upper"]).sum()),
                "pred/obs": (d[name] * d["n_test"]).sum() / (d["observed"] * d["n_test"]).sum(),
            }
        )
    return pd.DataFrame(rows)


fmt = {"L_T std RMS": "{:.2f}".format, "spearman": "{:.3f}".format, f"spearman n>={MIN_N}": "{:.3f}".format, "pred/obs": "{:.3f}".format}
print(f"Per segment (L_T = RMS of (estimate - observed) / SE over groups with n_test >= {MIN_N}; "
      "pred/obs = customer-weighted mean ratio):")
per_segment = pd.concat(
    [summary(d, reliable[d.index]).assign(segment=seg) for seg, d in obs.groupby("segment")], ignore_index=True
)
print(per_segment.set_index(["segment", "estimate"]).to_string(formatters=fmt))
print("\nOverall:")
print(summary(obs, reliable).set_index("estimate").to_string(formatters=fmt))

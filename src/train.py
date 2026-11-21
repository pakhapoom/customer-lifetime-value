"""Step 1 - training: train customers' first 36 months -> MDP per segment -> Bellman.

    uv run src/train.py   -> data/model.npz, data/actions.csv, data/state_values.csv
"""

import numpy as np
import pandas as pd

import mdp

products, customers, snapshot, transactions = mdp.load_tables()
panel = mdp.build_panel(products, customers, snapshot, transactions)
train = mdp.split(customers["customer_id"])
months = min(mdp.JOURNEY_MONTHS, panel["state"].shape[1] - 1)
model = mdp.fit(panel, train, months)

V_normal = mdp.solve(model["R_normal"], model["P"])
V_conservative = mdp.solve(model["R_conservative"], model["P"])
np.savez(
    "data/model.npz",
    segments=panel["segments"],
    groups=panel["groups"],
    **{k: v for k, v in model.items() if k != "actions"},
    V_normal=V_normal,
    V_conservative=V_conservative,
)
model["actions"].to_csv("data/actions.csv", index=False)

g, s = np.nonzero(model["n_s"])
names = mdp.state_names(panel["groups"])
values = pd.DataFrame(
    {
        "segment": np.array(panel["segments"])[g],
        "state": names[s],
        "customer_months": model["n_s"][g, s],
        "R_s_normal": model["R_normal"][g, s],
        "R_s_conservative": model["R_conservative"][g, s],
        "p_stay": model["P"][g, s, s],
        "CLV_normal": V_normal[g, s],
        "CLV_conservative": V_conservative[g, s],
    }
).sort_values(["segment", "CLV_normal"], ascending=[True, False])
values.to_csv("data/state_values.csv", index=False)

print(f"train customers {train.sum():,} of {len(train):,} | months {months} | gamma {mdp.GAMMA:.6f}")
print(f"states seen {len(values):,} (of {len(names)} per segment x {len(panel['segments'])} segments) "
      f"| (segment, state, action) rows {len(model['actions']):,}")
print("\nTop states by CLV (normal), per segment:")
print(values.groupby("segment").head(3).to_string(index=False, float_format="{:,.1f}".format))
print("\nsaved data/model.npz, data/actions.csv, data/state_values.csv")

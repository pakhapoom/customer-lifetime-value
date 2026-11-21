"""Step 2 - inference: CLV and most likely next purchase for every customer.

    uv run src/predict.py        -> data/predictions.csv
"""

import numpy as np
import pandas as pd

import mdp

products, customers, snapshot, transactions = mdp.load_tables()
model = np.load("data/model.npz")
actions = pd.read_csv("data/actions.csv")
panel = mdp.build_panel(products, customers, snapshot, transactions)

g, s = panel["segment"], panel["state"][:, -1]  # state as of now
pred = customers[["customer_id", "segment"]].assign(
    state=mdp.state_names(panel["groups"])[s],
    seen_in_training=model["n_s"][g, s] > 0,
    clv_normal=model["V_normal"][g, s].round(2),
    clv_conservative=model["V_conservative"][g, s].round(2),
)
best = actions.drop_duplicates(["segment", "state"])  # sorted by lambda, highest first
purchase = actions.groupby(["segment", "state"])["lambda"].sum().rename("p_purchase_next_month")
pred = (
    pred.merge(purchase, on=["segment", "state"], how="left")
    .merge(
        best[["segment", "state", "action", "lambda"]].rename(columns={"action": "most_likely_next_action", "lambda": "p_most_likely"}),
        on=["segment", "state"],
        how="left",
    )
    .fillna({"p_purchase_next_month": 0.0, "p_most_likely": 0.0, "most_likely_next_action": "{}"})
)
pred.to_csv("data/predictions.csv", index=False)
print(pred.head(10).to_string(index=False))
print(f"\n{len(pred):,} customers scored ({(~pred['seen_in_training']).sum():,} in a state unseen in training, CLV 0)"
      " -> data/predictions.csv")

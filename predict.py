"""Step 2 - inference: CLV and next best action for every current customer.

    uv run python predict.py        -> data/predictions.csv
"""

import numpy as np

from src import mdp

products, customers, holdings, offers = mdp.load_tables()
model = np.load("data/model.npz")
panel = mdp.build_panel(products, customers, holdings, offers)

h = panel["mask"][:, -1]  # holdings as of now
g = panel["segment"]
active = h > 0
held = mdp.bits_of(len(products))[h[active]] == 1
product_id = products["product_id"].to_numpy()
action_names = np.array(["none"] + [f"offer_{p}" for p in product_id])

pred = customers.loc[active, ["customer_id", "age", "monthly_income", "occupation"]].assign(
    holdings=[" ".join(product_id[row]) for row in held],
    monthly_revenue=held @ products["est_monthly_revenue"].to_numpy(),
    clv_current_policy=model["V_current"][g[active], h[active]].round(2),
    clv_optimal_policy=model["V_optimal"][g[active], h[active]].round(2),
    next_best_action=action_names[model["policy"][g[active], h[active]]],
)
pred.to_csv("data/predictions.csv", index=False)
print(pred.head(10).to_string(index=False))
print(f"\n{len(pred):,} active customers scored -> data/predictions.csv")

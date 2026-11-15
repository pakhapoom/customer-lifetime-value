"""Step 1 - training: tables in data/ -> learned MDP -> Bellman -> data/model.npz

    uv run src/train.py
"""

import numpy as np

import mdp

products, customers, holdings, offers = mdp.load_tables()
panel = mdp.build_panel(products, customers, holdings, offers)
model = mdp.fit(panel, products)

V_current, _ = mdp.evaluate_policy(model, model["pi"])  # CLV if the bank keeps acting as it has
V_optimal, policy = mdp.optimize(model)  # best achievable CLV + which offer achieves it

np.savez("data/model.npz", V_current=V_current, V_optimal=V_optimal, policy=policy)
months = int((panel["mask"][:, :-1] > 0).sum())
print(f"customer-months used: {months:,} | states: {V_current.size:,} | actions: {len(products) + 1}")
print("saved data/model.npz")

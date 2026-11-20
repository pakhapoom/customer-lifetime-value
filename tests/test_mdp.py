import numpy as np
import pandas as pd

import mdp, world

PRODUCTS = pd.DataFrame(
    {"product_id": ["P01", "P02", "P03"], "group_id": ["G1", "G1", "G2"], "one_time": [False, True, False]}
)


def test_solve_matches_closed_form():
    # One group. {} -> {G1} at rate p, {G1} is absorbing with reward r1.
    # V({G1}) = r1 / (1 - gamma);  V({}) = (r0 + gamma p V({G1})) / (1 - gamma (1 - p))
    r0, r1, p, gamma = 5.0, 10.0, 0.2, mdp.GAMMA
    R = np.array([[r0, r1]])
    P = np.array([[[1 - p, p], [0.0, 1.0]]])
    V = mdp.solve(R, P)
    assert np.isclose(V[0, 1], r1 / (1 - gamma))
    assert np.isclose(V[0, 0], (r0 + gamma * p * V[0, 1]) / (1 - gamma * (1 - p)))


def test_solve_equals_linear_solve_and_long_finite_horizon():
    customers, snapshot, transactions = world.simulate_history(n_customers=3000, n_months=24, seed=1)
    panel = mdp.build_panel(world.PRODUCTS, customers, snapshot, transactions)
    model = mdp.fit(panel, np.ones(len(customers), dtype=bool), 24)
    V = mdp.solve(model["R_normal"], model["P"])
    for g in range(len(panel["segments"])):
        H = V.shape[1]
        exact = np.linalg.solve(np.eye(H) - mdp.GAMMA * model["P"][g], model["R_normal"][g])
        assert np.allclose(V[g], exact)
    assert np.allclose(mdp.finite_horizon(model["R_normal"], model["P"], 1), model["R_normal"])
    assert np.allclose(mdp.finite_horizon(model["R_normal"], model["P"], 2000), V)


def test_build_panel_states_actions_and_rewards():
    customers = pd.DataFrame({"customer_id": ["C1", "C2"], "segment": ["S1", "S2"]})
    snapshot = pd.DataFrame({"customer_id": ["C1"], "group_id": ["G1"]})
    transactions = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C1"],
            "txn_date": pd.to_datetime(["2022-02-03", "2022-02-20", "2022-01-09", "2022-04-01"]),
            "product_id": ["P03", "P02", "P01", "P01"],
            "revenue": [120.0, 50.0, 12.0, 0.0],
        }
    )
    panel = mdp.build_panel(PRODUCTS, customers, snapshot, transactions)
    # G1 = bit 1, G2 = bit 2. Months Jan..Apr 2022 plus "now".
    assert panel["state"].tolist() == [[1, 1, 3, 3, 3], [0, 1, 1, 1, 1]]
    ev = panel["events"].set_index(["c", "m"])
    assert ev.loc[(0, 1), "action"] == "{G1/P02,G2/P03}"  # two products in a month = one compound action
    assert ev.loc[(0, 1), "next_state"] == 3
    assert np.isclose(ev.loc[(0, 1), "reward"], 120 * mdp.DEFAULT_MULTIPLIER + 50 * mdp.ONE_TIME_MULTIPLIER)
    assert ev.loc[(0, 3), "next_state"] == ev.loc[(0, 3), "state"] == 3  # bought within a held group
    assert ev.loc[(1, 0), "action"] == "{G1/P01}"


def test_fit_rates_rewards_and_transitions():
    # Segment S1, 4 customers in {} for one month. Two buy different G1 products
    # (two actions, same next state), one buys G2, one buys nothing.
    customers = pd.DataFrame({"customer_id": ["A", "B", "C", "D"], "segment": "S1"})
    snapshot = pd.DataFrame({"customer_id": [], "group_id": []})
    transactions = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "txn_date": pd.to_datetime(["2022-01-05"] * 3),
            "product_id": ["P01", "P02", "P03"],
            "revenue": [100.0, 40.0, 300.0],
        }
    )
    panel = mdp.build_panel(PRODUCTS, customers, snapshot, transactions)
    model = mdp.fit(panel, np.ones(4, dtype=bool), 1)
    a = model["actions"].set_index("action")
    assert np.allclose(a["lambda"], 0.25) and (a["n_s"] == 4).all()
    assert np.isclose(a["sd_lambda"].iloc[0], 1 / 4)
    P = model["P"][0]
    assert np.isclose(P[0, 1], 0.5)  # {} -> {G1}: sum of two actions
    assert np.isclose(P[0, 2], 0.25) and np.isclose(P[0, 0], 0.25)
    m = mdp.DEFAULT_MULTIPLIER
    assert np.isclose(model["R_normal"][0, 0], 0.25 * (100 * m + 40 + 300 * m))
    low = model["P_lower"][0]
    assert np.isclose(low[0, 1], max(0.5 - np.sqrt(2) / 4, 0.0))  # SEs of the two actions add in quadrature
    assert np.isclose(low[0, 2], 0.0) and np.isclose(low[0].sum(), 1.0)


def test_split_is_deterministic_and_about_70_30():
    ids = [f"C{i:06d}" for i in range(5000)]
    train = mdp.split(ids)
    assert (train == mdp.split(ids)).all()
    assert abs(train.mean() - mdp.TRAIN_RATIO) < 0.03


def test_pipeline_end_to_end_on_small_world():
    customers, snapshot, transactions = world.simulate_history(n_customers=4000, n_months=30, seed=3)
    panel = mdp.build_panel(world.PRODUCTS, customers, snapshot, transactions)
    train = mdp.split(customers["customer_id"])
    model = mdp.fit(panel, train, 24)
    for P in (model["P"], model["P_lower"]):
        assert np.allclose(P.sum(axis=2), 1.0) and (P >= 0).all()
        g, s, s2 = np.nonzero(P)
        assert ((s & s2) == s).all()  # only ever moves to a superset: the graph is a DAG
    assert (model["P_lower"] <= model["P"] + 1e-12)[:, ~np.eye(128, dtype=bool)].all()
    V = mdp.solve(model["R_normal"], model["P"])
    V_cons = mdp.solve(model["R_conservative"], model["P"])
    assert np.isfinite(V).all() and (V_cons <= V + 1e-9).all()  # geometric mean <= arithmetic mean
    assert (V[model["n_s"] == 0] == 0).all()  # unseen states carry no value

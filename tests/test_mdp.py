import numpy as np
import pandas as pd

from src import mdp, world


def test_policy_evaluation_matches_closed_form():
    # One product (revenue r, monthly retention q); never offered.
    # V(holding it) = r / (1 - gamma * q)
    r, q = 10.0, 0.9
    P = np.array([[[1.0, 0.0], [1 - q, q]]])
    model = {"P_none": P, "P_rej": P, "acc": np.zeros((1, 2, 1)), "R": np.array([[0.0, 0.0], [r, r - 2.0]])}
    V, _ = mdp.evaluate_policy(model, np.array([[[1.0, 0.0], [1.0, 0.0]]]))
    assert np.isclose(V[0, 1], r / (1 - mdp.GAMMA * q))
    assert V[0, 0] == 0.0


def test_optimize_offers_a_product_that_will_be_accepted():
    # Two products A (10/month) and B (50/month), nothing ever churns.
    # A-holders always accept B, so the best action when holding only A is "offer B".
    rev = np.array([10.0, 50.0])
    bits = mdp.bits_of(2)
    P = np.eye(4)[None]
    acc = np.zeros((1, 4, 2))
    acc[0, 1, 1] = 1.0
    R = np.repeat((bits @ rev)[:, None], 3, axis=1)
    R[:, 1:] -= mdp.OFFER_COST
    R[0] = 0.0
    valid = np.ones((4, 3), dtype=bool)
    valid[:, 1:] = bits == 0
    valid[0, 1:] = False
    V, policy = mdp.optimize({"P_none": P, "P_rej": P, "acc": acc, "R": R, "valid": valid})
    forever = 1 / (1 - mdp.GAMMA)
    assert policy[0].tolist() == [0, 2, 0, 0]
    assert np.isclose(V[0, 3], 60 * forever)
    assert np.isclose(V[0, 1], 10 - mdp.OFFER_COST + mdp.GAMMA * 60 * forever)


def test_build_panel_reconstructs_monthly_holdings():
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "age": [40, 25],
            "monthly_income": [30_000, 10_000],
            "signup_date": pd.to_datetime(["2022-01-15", "2022-02-03"]),
        }
    )
    holdings = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "product_id": ["P001", "P004", "P003"],
            "purchase_date": pd.to_datetime(["2022-01-15", "2022-03-10", "2022-02-03"]),
            "end_date": pd.to_datetime([None, "2022-05-20", "2022-04-01"]),
        }
    )
    offers = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "product_id": ["P004", "P010"],
            "offer_date": pd.to_datetime(["2022-03-10", "2022-04-05"]),
            "accepted": [True, False],
        }
    )
    panel = mdp.build_panel(world.PRODUCTS, customers, holdings, offers)
    # Months Jan..Jun 2022. P001 = bit 1, P003 = bit 4, P004 = bit 8.
    assert panel["mask"].tolist() == [[0, 1, 1, 9, 9, 1], [0, 0, 4, 4, 0, 0]]
    assert panel["action"].tolist() == [[0, 0, 4, 10, 0], [0, 0, 0, 0, 0]]
    assert panel["accepted"][0].tolist() == [False, False, True, False, False]
    assert panel["segment"].tolist() == [4, 0]


def test_pipeline_end_to_end_on_small_world():
    customers, holdings, offers = world.simulate_history(n_customers=1500, n_months=18, seed=3)
    panel = mdp.build_panel(world.PRODUCTS, customers, holdings, offers)
    held_now = mdp.bits_of(world.K)[panel["mask"][:, -1]].sum()
    assert held_now == holdings["end_date"].isna().sum()

    model = mdp.fit(panel, world.PRODUCTS)
    assert np.allclose(model["P_none"].sum(axis=2), 1.0)
    V_cur, _ = mdp.evaluate_policy(model, model["pi"])
    V_opt, policy = mdp.optimize(model)
    assert np.isfinite(V_cur).all() and np.allclose(V_cur[:, 0], 0.0, atol=1e-6)
    assert (V_opt >= V_cur - 1e-6).all()  # optimal is at least as good as current
    assert model["valid"][np.arange(1 << world.K), policy].all()  # never offers a held product

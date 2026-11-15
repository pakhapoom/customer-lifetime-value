"""The mock world: a 10-product retail-bank catalog, customers, hidden "true"
behaviour, and a simulator that writes the tables a real system would have.

The model in mdp.py never reads the hidden parameters below. They exist only
to generate history and to score predictions by simulating the future.
"""

import numpy as np
import pandas as pd

from mdp import GAMMA, OFFER_COST

START = "2022-01"  # first month of history
P_OFFER = 0.3  # behaviour policy: chance of making an offer in a month

CATEGORIES = ["deposit", "card", "loan", "insurance", "investment"]
TIERS = ["basic", "standard", "premium", "elite"]
OCCUPATIONS = ["salaried", "self_employed", "student", "retired"]

PRODUCTS = pd.DataFrame(
    [
        ("P001", "Savings Account", "deposit", "basic", 3.0),
        ("P002", "Fixed Deposit", "deposit", "standard", 8.0),
        ("P003", "Debit Card", "card", "basic", 2.0),
        ("P004", "Credit Card Classic", "card", "standard", 15.0),
        ("P005", "Credit Card Platinum", "card", "premium", 40.0),
        ("P006", "Personal Loan", "loan", "standard", 35.0),
        ("P007", "Home Loan", "loan", "elite", 120.0),
        ("P008", "Life Insurance", "insurance", "premium", 30.0),
        ("P009", "Mutual Fund", "investment", "premium", 45.0),
        ("P010", "Wealth Management", "investment", "elite", 150.0),
    ],
    columns=["product_id", "name", "category", "tier", "est_monthly_revenue"],
)
K = len(PRODUCTS)
_REV = PRODUCTS["est_monthly_revenue"].to_numpy()
_CAT = PRODUCTS["category"].map(CATEGORIES.index).to_numpy()
_TIER = PRODUCTS["tier"].map(TIERS.index).to_numpy()

# --- hidden ground truth ---
_CHURN = np.array([0.008, 0.030, 0.010, 0.020, 0.020, 0.040, 0.008, 0.012, 0.025, 0.015])  # monthly, per product
_APPEAL = np.array([-0.8, -2.0, -1.0, -1.6, -2.0, -2.2, -2.6, -2.2, -2.0, -2.4])  # base logit of saying yes
_AGE_CAT = np.array(  # age band (<30, 30-49, 50+) x category
    [[0.3, 0.5, 0.0, -0.6, -0.4], [0.0, 0.2, 0.5, 0.3, 0.2], [0.1, -0.3, -0.5, 0.5, 0.5]]
)
_OCC_CAT = np.array(  # occupation x category
    [[0, 0, 0, 0, 0], [0, 0, 0.4, 0, 0.2], [0, 0.2, -0.5, -0.3, -0.5], [0.2, -0.2, -0.6, 0.3, 0.2]]
)
_SYNERGY = np.array(  # holding a product in row category -> appetite for column category
    [
        [-0.8, 0.4, 0.3, 0.2, 0.3],
        [0.0, -0.5, 0.3, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.1],
        [0.0, 0.0, 0.0, 0.2, 0.3],
    ]
)
_LADDER = 0.5  # holding tier t-1 raises appetite for tier t
_INCOME_NEEDED = np.array([-2.0, -0.8, 0.2, 1.0])  # per tier, in std devs of log income
_ORGANIC = 0.02  # unprompted purchase prob = _ORGANIC * acceptance prob
_FATIGUE = 1.15  # a rejected offer raises churn that month
_AGE_CHURN = np.array([1.3, 1.0, 0.8])
_CAT_1H = np.eye(len(CATEGORIES), dtype=np.float32)[_CAT]
_TIER_1H = np.eye(len(TIERS), dtype=np.float32)[_TIER]


def make_customers(n, n_months, rng):
    age = np.clip(rng.normal(40, 14, n), 18, 80).astype(int)
    u = rng.random(n)
    occ = np.where(u < 0.75, 0, 1)  # salaried / self-employed
    occ = np.where((age < 25) & (u < 0.6), 2, occ)  # student
    occ = np.where((age > 60) & (u < 0.7), 3, occ)  # retired
    income = rng.lognormal(np.log(35_000), 0.6, n) * np.array([1.0, 1.0, 0.4, 0.6])[occ]
    signup = _dates(rng.integers(0, n_months - 1, n), rng)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "age": age,
            "monthly_income": income.round(-2),
            "occupation": np.array(OCCUPATIONS)[occ],
            "signup_date": signup,
        }
    )


def traits(customers):
    """Per-customer hidden traits: static appetite logit [n, K] and churn multiplier [n]."""
    age_i = np.digitize(customers["age"], [30, 50])
    occ_i = customers["occupation"].map(OCCUPATIONS.index).to_numpy()
    x = (np.log(customers["monthly_income"].to_numpy()) - np.log(35_000)) / 0.6
    afford = -1.5 * np.maximum(0.0, _INCOME_NEEDED[_TIER][None, :] - x[:, None])
    appetite = _APPEAL + _AGE_CAT[age_i][:, _CAT] + _OCC_CAT[occ_i][:, _CAT] + afford
    return appetite.astype(np.float32), _AGE_CHURN[age_i]


def step(h, appetite, churn_mult, action, rng):
    """One month. h: bool [n, K] held at the start. Returns (h_next, accepted)."""
    n = len(h)
    rows = np.arange(n)
    alive = h.any(axis=1)
    hf = h.astype(np.float32)
    has_cat = (hf @ _CAT_1H > 0).astype(np.float32)
    has_tier = (hf @ _TIER_1H > 0).astype(np.float32)
    ladder = np.concatenate([np.zeros((n, 1), np.float32), has_tier[:, :-1]], axis=1)
    logit = appetite + (has_cat @ _SYNERGY)[:, _CAT] + _LADDER * ladder[:, _TIER]
    p_yes = 1 / (1 + np.exp(-logit))

    k = np.maximum(action - 1, 0)
    offered = (action > 0) & alive & ~h[rows, k]
    accepted = offered & (rng.random(n) < p_yes[rows, k])

    stick = np.maximum(0.4, 0.85 ** (h.sum(axis=1) - 1))  # more products -> stickier
    fatigue = np.where(offered & ~accepted, _FATIGUE, 1.0)
    p_end = _CHURN * (churn_mult * stick * fatigue)[:, None]
    # One draw per (customer, product): held products may end, unheld may be bought.
    event = rng.random((n, K)) < np.where(h, p_end, _ORGANIC * p_yes)
    h_next = (h ^ event) & alive[:, None]
    h_next[rows[accepted], k[accepted]] = True
    return h_next, accepted


def behaviour_policy(h, rows, rng):
    """What the bank did historically: sometimes offer a random product not held."""
    k = rng.integers(0, K, len(h))
    offer = (rng.random(len(h)) < P_OFFER) & ~h[np.arange(len(h)), k]
    return np.where(offer, k + 1, 0)


def _dates(month, rng):
    day = rng.integers(0, 28, len(month)).astype("timedelta64[D]")
    return (np.datetime64(START, "M") + month.astype("timedelta64[M]")).astype("datetime64[D]") + day


def simulate_history(n_customers=20_000, n_months=36, seed=0):
    """Returns the mock tables: customers, holdings, offers.

    holdings: customer_id, product_id, purchase_date, end_date (empty = still held)
    offers:   customer_id, product_id, offer_date, accepted
    A customer whose last product ends has left the bank for good.
    """
    rng = np.random.default_rng(seed)
    customers = make_customers(n_customers, n_months, rng)
    signup = (customers["signup_date"].dt.year * 12 + customers["signup_date"].dt.month).to_numpy()
    signup = signup - (int(START[:4]) * 12 + int(START[5:]))
    appetite, churn_mult = traits(customers)

    h = np.zeros((n_customers, K), dtype=bool)
    since = np.full((n_customers, K), -1)  # purchase month of each open holding
    hold, offer = [], []
    for m in range(n_months):
        idx = np.flatnonzero(h.any(axis=1))
        a = behaviour_policy(h[idx], idx, rng)
        h_next, accepted = step(h[idx], appetite[idx], churn_mult[idx], a, rng)
        c, k = np.nonzero(h[idx] & ~h_next)  # ended
        hold.append((idx[c], k, since[idx[c], k], np.full(len(c), m)))
        since[idx[c], k] = -1
        c, k = np.nonzero(~h[idx] & h_next)  # bought
        since[idx[c], k] = m
        offer.append((idx[a > 0], a[a > 0] - 1, np.full((a > 0).sum(), m), accepted[a > 0]))
        h[idx] = h_next

        new = np.flatnonzero(signup == m)  # starter products
        first = rng.choice([0, 2, 3], len(new), p=[0.7, 0.15, 0.15])
        h[new, first] = True
        h[new[(first == 0) & (rng.random(len(new)) < 0.4)], 2] = True
        since[new] = np.where(h[new], m, -1)

    c, k = np.nonzero(since >= 0)  # still held at the end
    hold.append((c, k, since[c, k], np.full(len(c), -1)))
    c, k, bought, end = (np.concatenate(x) for x in zip(*hold))
    holdings = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[c],
            "product_id": PRODUCTS["product_id"].to_numpy()[k],
            "purchase_date": _dates(bought, rng),
            "end_date": np.where(end >= 0, _dates(np.maximum(end, 0), rng), np.datetime64("NaT", "D")),
        }
    ).sort_values(["purchase_date", "customer_id"], ignore_index=True)
    c, k, month, accepted = (np.concatenate(x) for x in zip(*offer))
    offers = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[c],
            "product_id": PRODUCTS["product_id"].to_numpy()[k],
            "offer_date": _dates(month, rng),
            "accepted": accepted,
        }
    ).sort_values(["offer_date", "customer_id"], ignore_index=True)
    return customers, holdings, offers


def rollout(h, appetite, churn_mult, policy, rng, horizon=480):
    """Simulate the true future under `policy(h, rows, rng) -> action` and return
    each row's realised discounted profit (revenue - offer costs)."""
    value = np.zeros(len(h))
    rows = np.arange(len(h))
    discount = 1.0
    for _ in range(horizon):
        a = np.where(h.any(axis=1), policy(h, rows, rng), 0)
        value[rows] += discount * (h @ _REV - OFFER_COST * (a > 0))
        h, _ = step(h, appetite, churn_mult, a, rng)
        keep = h.any(axis=1)
        if keep.mean() < 0.9:  # drop customers who have left
            h, appetite, churn_mult, rows = h[keep], appetite[keep], churn_mult[keep], rows[keep]
        if len(h) == 0:
            break
        discount *= GAMMA
    return value

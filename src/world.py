"""The mock world: a retail-bank catalog of 18 products in 7 product groups,
customers in 4 segments, hidden "true" purchase behaviour, and a simulator that
writes the tables a real system would have. Products, groups and segments are
anonymised as running numbers.

The model in mdp.py never reads the hidden parameters below. They exist only
to generate the history.
"""

import numpy as np
import pandas as pd

from mdp import JOURNEY_START

SEGMENTS = ["S1", "S2", "S3", "S4"]
GROUPS = [f"G{i}" for i in range(1, 8)]

_CATALOG = np.array(  # group, one-time?, base monthly purchase prob, mean booked revenue
    [
        (0, 0, 0.0060, 300),
        (0, 0, 0.0030, 800),
        (0, 0, 0.0020, 1_500),
        (1, 0, 0.0060, 1_200),
        (1, 0, 0.0030, 3_000),
        (1, 0, 0.0020, 500),
        (2, 0, 0.0015, 9_000),
        (2, 0, 0.0010, 15_000),
        (3, 0, 0.0030, 4_000),
        (3, 0, 0.0020, 6_000),
        (3, 0, 0.0020, 2_500),
        (3, 1, 0.0040, 150),
        (4, 0, 0.0020, 5_000),
        (4, 0, 0.0010, 12_000),
        (5, 0, 0.0005, 40_000),
        (5, 0, 0.0005, 25_000),
        (6, 0, 0.0010, 8_000),
        (6, 0, 0.0008, 6_000),
    ]
)
_GROUP = _CATALOG[:, 0].astype(int)
_ONE_TIME = _CATALOG[:, 1].astype(bool)
_BASE = _CATALOG[:, 2]
_REVENUE = _CATALOG[:, 3]
PRODUCTS = pd.DataFrame(
    {
        "product_id": [f"P{i + 1:02d}" for i in range(len(_CATALOG))],
        "group_id": np.array(GROUPS)[_GROUP],
        "one_time": _ONE_TIME,
    }
)
K = len(PRODUCTS)

# --- hidden ground truth ---
_SEG_SHARE = [0.45, 0.30, 0.17, 0.08]
_SEG_BUY = np.array([0.8, 1.0, 1.25, 1.5])  # purchase-rate multiplier
_SEG_REVENUE = np.array([1.0, 1.6, 2.6, 4.5])  # revenue multiplier
_SEG_HOLD = np.array(  # chance of holding each group at the snapshot
    [
        [0.80, 0.40, 0.05, 0.15, 0.05, 0.03, 0.02],
        [0.85, 0.55, 0.10, 0.25, 0.10, 0.06, 0.05],
        [0.90, 0.65, 0.15, 0.35, 0.20, 0.10, 0.10],
        [0.95, 0.70, 0.20, 0.45, 0.35, 0.15, 0.20],
    ]
)
_SYNERGY = np.array(  # log-lift: holding the row group -> appetite for the column group
    [
        [0.0, 0.5, 0.3, 0.3, 0.3, 0.2, 0.2],
        [0.2, 0.0, 0.3, 0.2, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.0, 0.5, 0.0, 0.3, 0.0],
        [0.1, 0.1, 0.0, 0.0, 0.3, 0.0, 0.2],
        [0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.4],
        [0.1, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0],
    ]
)
_SAME_GROUP = 0.5  # another product of a group already held is less likely
_DORMANT = 0.3  # customers who hold nothing rarely buy
_FRAILTY_SHAPE = 1.5  # per-customer activity ~ Gamma(mean 1): unobserved heterogeneity
_REV_SIGMA = 0.8  # lognormal spread of booked revenue
_ZERO_REVENUE = 0.05  # approved but never booked


def purchase_prob(held, seg, frailty):
    """Monthly probability of buying each product. held: bool [n, groups]."""
    lift = np.exp(held.astype(float) @ _SYNERGY)[:, _GROUP]
    same = np.where(held[:, _GROUP] & ~_ONE_TIME, _SAME_GROUP, 1.0)
    rate = (_SEG_BUY[seg] * frailty * np.where(held.any(axis=1), 1.0, _DORMANT))[:, None]
    return np.minimum(_BASE * rate * lift * same, 0.5)


def simulate_history(n_customers=50_000, n_months=53, seed=0):
    """Returns the mock tables: customers, holdings_snapshot, transactions.

    customers:          customer_id, segment
    holdings_snapshot:  customer_id, group_id   groups held just before the journey starts
    transactions:       customer_id, txn_date, product_id, revenue (booked revenue)
    """
    rng = np.random.default_rng(seed)
    seg = rng.choice(len(SEGMENTS), n_customers, p=_SEG_SHARE)
    frailty = rng.gamma(_FRAILTY_SHAPE, 1 / _FRAILTY_SHAPE, n_customers)
    ids = np.array([f"C{i:06d}" for i in range(n_customers)])
    customers = pd.DataFrame({"customer_id": ids, "segment": np.array(SEGMENTS)[seg]})

    held = rng.random((n_customers, len(GROUPS))) < _SEG_HOLD[seg]
    c, g = np.nonzero(held)
    snapshot = pd.DataFrame({"customer_id": ids[c], "group_id": np.array(GROUPS)[g]})

    bought = []
    for m in range(n_months):
        c, k = np.nonzero(rng.random((n_customers, K)) < purchase_prob(held, seg, frailty))
        bought.append((c, k, np.full(len(c), m)))
        held[c, _GROUP[k]] = True
    c, k, m = (np.concatenate(x) for x in zip(*bought))

    mean = _REVENUE[k] * _SEG_REVENUE[seg[c]]
    revenue = rng.lognormal(np.log(mean) - _REV_SIGMA**2 / 2, _REV_SIGMA)
    revenue[~_ONE_TIME[k] & (rng.random(len(k)) < _ZERO_REVENUE)] = 0.0
    day = rng.integers(0, 28, len(m)).astype("timedelta64[D]")
    date = (np.datetime64(JOURNEY_START, "M") + m.astype("timedelta64[M]")).astype("datetime64[D]") + day
    transactions = pd.DataFrame(
        {"customer_id": ids[c], "txn_date": date, "product_id": PRODUCTS["product_id"].to_numpy()[k], "revenue": revenue.round(2)}
    ).sort_values(["txn_date", "customer_id"], ignore_index=True)
    return customers, snapshot, transactions

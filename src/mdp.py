"""The MDP: build monthly states from raw tables, estimate the model, solve Bellman.

Segment g     a model key, not part of the state: every quantity below is
              estimated, and every Bellman equation solved, per segment.
State   s     the set of product groups the customer has ever held, as a bitmask
              over groups (7 groups -> 128 states). Groups are only ever added,
              so s only grows and the state graph is a DAG ordered by set size.
Action  a     the set of products bought during the month, e.g. {G2/P05,G4/P11}.
              Buying nothing is the implicit "no purchase" action.
Transition    deterministic given the action:  s' = s | groups(a).
              Several actions can lead from s to the same s' (any product of a
              group), and buying a product of a group already held keeps s' = s.
Reward  r     per transaction, booked revenue x multiplier, where the multiplier
              spreads the expected first-year revenue over the product lifetime
              capped at 36 months:  (1/12) * sum_{k<36} gamma^k  (1.0 for
              one-time products). R(s, a) = mean reward of the month.

    lambda(a|s) = sum_m N_m(s, a) / sum_m N_m(s)       monthly action rate
    R(s)        = sum_a lambda(a|s) R(s, a)
    P(s'|s)     = sum_{a: s|groups(a) = s'} lambda(a|s)      for s' != s
    P(s|s)      = 1 - sum_{s' != s} P(s'|s)                   no purchase, or same-group purchase

    V(s) = R(s) + gamma * sum_s' P(s'|s) V(s')
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

JOURNEY_START = "2022-01"  # first month of history; the holdings snapshot is taken just before
JOURNEY_MONTHS = 36  # training window
DAILY_DISCOUNT = 0.998890
GAMMA = DAILY_DISCOUNT**30  # monthly discount factor
REWARD_MONTHS = 36  # product lifetime cap used by the reward multiplier
TRAIN_RATIO = 0.7
SPLIT_SALT = "clv_mdp"


def annuity(months, gamma=GAMMA):
    """sum_{k=0}^{months-1} gamma^k"""
    return (1 - gamma**months) / (1 - gamma)


DEFAULT_MULTIPLIER = annuity(REWARD_MONTHS) / 12
ONE_TIME_MULTIPLIER = 1.0


def popcount(n_groups):
    """[2^G] number of groups in each state."""
    s = np.arange(1 << n_groups)
    return ((s[:, None] >> np.arange(n_groups)) & 1).sum(axis=1)


def state_names(groups):
    """[2^G] labels such as "{}" or "{G1,G3}"."""
    return np.array(
        ["{" + ",".join(g for i, g in enumerate(groups) if s >> i & 1) + "}" for s in range(1 << len(groups))]
    )


def load_tables(data_dir="data"):
    d = Path(data_dir)
    return (
        pd.read_csv(d / "products.csv"),
        pd.read_csv(d / "customers.csv"),
        pd.read_csv(d / "holdings_snapshot.csv"),
        pd.read_csv(d / "transactions.csv", parse_dates=["txn_date"]),
    )


def split(customer_ids, ratio=TRAIN_RATIO, salt=SPLIT_SALT):
    """Deterministic hash split: True = train, False = test."""
    bucket = [int(hashlib.md5(f"{c}{salt}".encode()).hexdigest(), 16) % 100 for c in customer_ids]
    return np.array(bucket) < ratio * 100


def build_panel(products, customers, snapshot, transactions):
    """Turn the tables into a monthly panel.

    state[c, m] is customer c's state at the *start* of month m: the groups in
    the snapshot plus the groups of everything bought before month m. The last
    column (after the last month) is the "as of now" state.

    events has one row per customer-month with at least one purchase: the
    compound action, the groups it adds, and the month's total reward.
    """
    groups = list(pd.unique(products["group_id"]))
    segments = sorted(customers["segment"].unique())
    cust_ix, group_ix = pd.Index(customers["customer_id"]), pd.Index(groups)
    group_bit = 1 << group_ix.get_indexer(products["group_id"])
    multiplier = np.where(products["one_time"], ONE_TIME_MULTIPLIER, DEFAULT_MULTIPLIER)

    init = np.zeros(len(customers), dtype=np.int64)
    np.bitwise_or.at(init, cust_ix.get_indexer(snapshot["customer_id"]), 1 << group_ix.get_indexer(snapshot["group_id"]))

    k = pd.Index(products["product_id"]).get_indexer(transactions["product_id"])
    start = pd.Period(JOURNEY_START, "M")
    t = transactions["txn_date"].dt
    tx = pd.DataFrame(
        {
            "c": cust_ix.get_indexer(transactions["customer_id"]),
            "m": (t.year - start.year) * 12 + t.month - start.month,
            "item": products["group_id"].to_numpy()[k] + "/" + products["product_id"].to_numpy()[k],
            "bit": group_bit[k],
            "reward": transactions["revenue"].fillna(0.0).to_numpy() * multiplier[k],
        }
    )
    M = int(tx["m"].max()) + 1 if len(tx) else 0
    by_month = tx.groupby(["c", "m"])
    events = pd.DataFrame(
        {
            "action": "{" + tx.drop_duplicates(["c", "m", "item"]).sort_values("item").groupby(["c", "m"])["item"].agg(",".join) + "}",
            "a_mask": tx.drop_duplicates(["c", "m", "bit"]).groupby(["c", "m"])["bit"].sum(),
            "reward": by_month["reward"].sum(),
        }
    ).reset_index()

    bought = np.zeros((len(customers), M), dtype=np.int64)
    np.bitwise_or.at(bought, (events["c"].to_numpy(), events["m"].to_numpy()), events["a_mask"].to_numpy())
    state = np.empty((len(customers), M + 1), dtype=np.int64)
    state[:, 0] = init
    state[:, 1:] = init[:, None] | np.bitwise_or.accumulate(bought, axis=1)
    events["state"] = state[events["c"], events["m"]]
    events["next_state"] = events["state"] | events["a_mask"]
    return {
        "groups": groups,
        "segments": segments,
        "segment": pd.Index(segments).get_indexer(customers["segment"]),
        "state": state,
        "events": events,
    }


def fit(panel, rows, n_months=JOURNEY_MONTHS):
    """Estimate the MDP per segment from customers `rows` (bool mask) over the
    first `n_months` months. Returns rewards R[g, s], transitions P[g, s, s'],
    the exposure n_s[g, s] (customer-months) and the per-(s, a) table."""
    G, H = len(panel["segments"]), 1 << len(panel["groups"])
    seg = panel["segment"]

    # N(s): customer-months spent in each state (the denominator of lambda).
    keys = seg[rows][:, None] * H + panel["state"][rows, :n_months]
    n_s = np.bincount(keys.ravel(), minlength=G * H).reshape(G, H)

    ev = panel["events"]
    ev = ev[rows[ev["c"]] & (ev["m"] < n_months)].assign(g=lambda d: seg[d["c"]], log1p=lambda d: np.log1p(d["reward"]))
    sa = (
        ev.groupby(["g", "state", "action"])
        .agg(next_state=("next_state", "first"), n_sa=("c", "size"), mean_reward=("reward", "mean"), log1p=("log1p", "mean"))
        .reset_index()
    )
    sa["n_s"] = n_s[sa["g"], sa["state"]]
    sa["lambda"] = sa["n_sa"] / sa["n_s"]
    sa["sd_lambda"] = np.sqrt(sa["n_sa"]) / sa["n_s"]  # Poisson standard error
    sa["geom_reward"] = np.expm1(sa.pop("log1p"))  # conservative: geometric mean

    g, s, s2, lam = (sa[c].to_numpy() for c in ["g", "state", "next_state", "lambda"])
    R_normal, R_conservative = np.zeros((G, H)), np.zeros((G, H))
    np.add.at(R_normal, (g, s), lam * sa["mean_reward"].to_numpy())
    np.add.at(R_conservative, (g, s), lam * sa["geom_reward"].to_numpy())

    move = s2 != s
    off, var = np.zeros((G, H, H)), np.zeros((G, H, H))
    np.add.at(off, (g[move], s[move], s2[move]), lam[move])
    np.add.at(var, (g[move], s[move], s2[move]), sa["sd_lambda"].to_numpy()[move] ** 2)
    names = state_names(panel["groups"])
    actions = sa.assign(segment=np.array(panel["segments"])[g], state=names[s], next_state=names[s2])
    actions = actions[["segment", "state", "action", "next_state", "n_sa", "n_s", "lambda", "sd_lambda", "mean_reward", "geom_reward"]]
    return {
        "R_normal": R_normal,
        "R_conservative": R_conservative,
        "P": _with_self_loops(off),
        # Every s -> s' rate lowered by one standard error; the removed mass stays in s.
        "P_lower": _with_self_loops(np.maximum(off - np.sqrt(var), 0.0)),
        "n_s": n_s,
        "actions": actions.sort_values(["segment", "state", "lambda"], ascending=[True, True, False], ignore_index=True),
    }


def _with_self_loops(off):
    P = off.copy()
    H = P.shape[1]
    P[:, np.arange(H), np.arange(H)] = 1.0 - off.sum(axis=2)
    if (P < -1e-9).any():
        raise ValueError("a state's outgoing transition rates sum to more than one")
    return P


def solve(R, P):
    """Bellman expectation equation, exactly and in one pass.

    Every transition goes to a strict superset s' of s or stays in s, so

        V(s) = [R(s) + gamma * sum_{s' != s} P(s'|s) V(s')] / (1 - gamma * P(s|s))

    can be solved from the largest states down: when a state is reached, every
    V(s') it depends on is already final. States of the same size never lead to
    each other, so each size is solved at once.
    """
    G, H = R.shape
    size = popcount(H.bit_length() - 1)
    stay = P[:, np.arange(H), np.arange(H)]
    V = np.zeros((G, H))
    for n in range(size.max(), -1, -1):
        L = size == n  # V[:, L] is still 0 here, so P @ V sums only over s' != s
        V[:, L] = (R[:, L] + GAMMA * np.einsum("gsh,gh->gs", P[:, L], V)) / (1 - GAMMA * stay[:, L])
    return V


def finite_horizon(R, P, months):
    """V_T(s) = sum_{t<T} gamma^t E[R(s_t) | s_0 = s], by T Bellman backups."""
    V = np.zeros_like(R)
    for _ in range(months):
        V = R + GAMMA * np.einsum("gsh,gh->gs", P, V)
    return V

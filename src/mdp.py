"""The MDP: build states from raw tables, learn the model, solve with Bellman.

State  (g, h)  g = demographic segment (age band x income band, 9 values)
               h = bitmask of the products held (2^K values, K = 10 -> 1024)
               h = 0 (holds nothing) means churned and is absorbing.
Action a = 0 "none", or k >= 1 "offer product k-1". Offering a product the
       customer already holds is not allowed.
Reward R(h, a) = sum of est. monthly revenue of held products - offer cost.

Transitions are learned in three pieces rather than one table per action:

    P(h'|g,h, none)    = P_none[g,h,h']              counted from no-offer months
    P(h'|g,h, offer k) = acc[g,h,k]     * (P_none[g,h,.] with product k added)
                       + (1-acc[g,h,k]) * P_rej[g,h,h']   counted from rejected-offer months
"""

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

GAMMA = 0.99  # monthly discount factor
OFFER_COST = 2.0
AGE_EDGES = [30, 50]
INCOME_EDGES = [25_000, 60_000]
N_SEGMENTS = 9


def segment_of(customers):
    age_i = np.digitize(customers["age"], AGE_EDGES)
    income_i = np.digitize(customers["monthly_income"], INCOME_EDGES)
    return age_i * 3 + income_i


def bits_of(n_products):
    """[2^K, K] table: bits_of(K)[h, k] == 1 if holdings mask h contains product k."""
    return (np.arange(1 << n_products)[:, None] >> np.arange(n_products)) & 1


def load_tables(data_dir="data"):
    d = Path(data_dir)
    return (
        pd.read_csv(d / "products.csv"),
        pd.read_csv(d / "customers.csv", parse_dates=["signup_date"]),
        pd.read_csv(d / "holdings.csv", parse_dates=["purchase_date", "end_date"]),
        pd.read_csv(d / "offers.csv", parse_dates=["offer_date"]),
    )


def build_panel(products, customers, holdings, offers):
    """Turn the event tables into a monthly panel.

    mask[c, m] is what customer c holds at the *start* of month m: products
    bought before month m and not ended before month m. An offer, purchase or
    end dated in month m moves the customer from mask[c, m] to mask[c, m+1].
    The last column (one month past the last event) is the "as of now" state.
    """
    cust_ix, prod_ix = pd.Index(customers["customer_id"]), pd.Index(products["product_id"])
    first = customers["signup_date"].min()
    origin = first.year * 12 + first.month

    def month(s):
        return (s.dt.year * 12 + s.dt.month).to_numpy() - origin

    c = cust_ix.get_indexer(holdings["customer_id"])
    bit = 1 << prod_ix.get_indexer(holdings["product_id"])
    bought = month(holdings["purchase_date"]).astype(int)
    end_f = month(holdings["end_date"])  # NaN while still held
    ended = ~np.isnan(end_f)
    end = end_f[ended].astype(int)
    off_c = cust_ix.get_indexer(offers["customer_id"])
    off_m = month(offers["offer_date"]).astype(int)
    M = int(max(bought.max(), end.max(initial=0), off_m.max(initial=0))) + 1

    # Difference array: +bit the month after purchase, -bit the month after the end.
    mask = np.zeros((len(customers), M + 2), dtype=np.int64)
    np.add.at(mask, (c, bought + 1), bit)
    np.add.at(mask, (c[ended], end + 1), -bit[ended])
    mask = mask.cumsum(axis=1)[:, : M + 1]

    action = np.zeros((len(customers), M), dtype=int)
    accepted = np.zeros((len(customers), M), dtype=bool)
    action[off_c, off_m] = prod_ix.get_indexer(offers["product_id"]) + 1
    accepted[off_c, off_m] = offers["accepted"].to_numpy()
    return {"segment": segment_of(customers), "mask": mask, "action": action, "accepted": accepted}


def _rate(index, hit, shape, prior, alpha):
    """Hit rate per cell, shrunk toward `prior` with `alpha` pseudo-counts."""
    n, y = np.zeros(shape), np.zeros(shape)
    np.add.at(n, index, 1)
    np.add.at(y, index, hit)
    return (y + alpha * prior) / (n + alpha)


def _transitions(g, h, h_next, bits, alpha):
    """P[g, h, h'] by counting, with two levels of shrinkage:

    segment row -> all-segments row -> product-factored prior.

    The prior treats products as independent: each held product ends at its
    own churn rate and each unheld product is bought at its own organic rate.
    It is what the model falls back on for holdings never seen in the data.
    """
    H, K = bits.shape
    b, b_next = bits[h], bits[h_next]
    churn = ((b == 1) & (b_next == 0)).sum(axis=0) / np.maximum((b == 1).sum(axis=0), 1)
    buy = ((b == 0) & (b_next == 1)).sum(axis=0) / np.maximum((b == 0).sum(axis=0), 1)
    per_product = [np.array([[1 - buy[k], buy[k]], [churn[k], 1 - churn[k]]]) for k in range(K)]
    prior = reduce(np.kron, per_product[::-1])  # [H, H]; bit 0 is least significant

    P = np.zeros((N_SEGMENTS, H, H))
    np.add.at(P, (g, h, h_next), 1)
    pooled = P.sum(axis=0)
    pooled = (pooled + alpha * prior) / (pooled.sum(axis=1, keepdims=True) + alpha)
    n = P.sum(axis=2, keepdims=True)
    P += alpha * pooled
    P /= n + alpha
    P[:, 0, :] = 0.0
    P[:, 0, 0] = 1.0  # churned is absorbing
    return P


def fit(panel, products, alpha=5.0, alpha_acc=20.0):
    """Estimate every piece of the MDP from the panel."""
    rev = products["est_monthly_revenue"].to_numpy()
    K = len(rev)
    bits = bits_of(K)
    H = len(bits)

    mask = panel["mask"]
    active = mask[:, :-1] > 0
    g = np.broadcast_to(panel["segment"][:, None], active.shape)[active]
    h, h_next = mask[:, :-1][active], mask[:, 1:][active]
    a, ok = panel["action"][active], panel["accepted"][active]

    none, rejected = a == 0, (a > 0) & ~ok
    P_none = _transitions(g[none], h[none], h_next[none], bits, alpha)
    P_rej = _transitions(g[rejected], h[rejected], h_next[rejected], bits, alpha)

    # Acceptance of offer k: the (segment, holdings, product) cell, shrunk toward
    # (segment appeal of k) x (how holdings h lift or lower appetite for k).
    off = a > 0
    go, ho, ko, yo = g[off], h[off], a[off] - 1, ok[off]
    acc_k = _rate((ko,), yo, (K,), yo.mean(), alpha_acc)
    acc_gk = _rate((go, ko), yo, (N_SEGMENTS, K), acc_k, alpha_acc)
    lift = _rate((ho, ko), yo, (H, K), acc_k, alpha_acc) / acc_k
    prior = np.clip(acc_gk[:, None, :] * lift[None], 0.0, 0.95)
    acc = _rate((go, ho, ko), yo, (N_SEGMENTS, H, K), prior, alpha_acc)
    acc[:, bits == 1] = 0.0

    R = np.repeat((bits @ rev)[:, None], K + 1, axis=1)
    R[:, 1:] -= OFFER_COST
    R[0] = 0.0

    valid = np.ones((H, K + 1), dtype=bool)
    valid[:, 1:] = bits == 0
    valid[0, 1:] = False

    # Behaviour policy pi[g, h, a], shrunk toward the overall action mix.
    overall = np.bincount(a, minlength=K + 1) * valid
    overall = overall / overall.sum(axis=1, keepdims=True)
    pi = np.zeros((N_SEGMENTS, H, K + 1))
    np.add.at(pi, (g, h, a), 1)
    pi = (pi + 10.0 * overall) / (pi.sum(axis=2, keepdims=True) + 10.0)

    return {"P_none": P_none, "P_rej": P_rej, "acc": acc, "R": R, "valid": valid, "pi": pi}


def evaluate_policy(model, pi):
    """Bellman expectation equation, solved exactly per segment:

        V = R_pi + gamma * P_pi V   =>   V = (I - gamma * P_pi)^-1 R_pi

    pi[g, h, a] are action probabilities. Returns V[g, h] and Q[g, h, a], where
    Q(s,a) = R(s,a) + gamma * E[V(s') | s, a].
    """
    R, acc_all = model["R"], model["acc"]
    G, H, K = acc_all.shape
    idx = np.arange(H)
    V, Q = np.zeros((G, H)), np.zeros((G, H, K + 1))
    for g in range(G):
        Pn, Pr, acc = model["P_none"][g], model["P_rej"][g], acc_all[g]
        P = pi[g][:, :1] * Pn + (pi[g][:, 1:] * (1 - acc)).sum(axis=1, keepdims=True) * Pr
        for k in range(K):
            w = pi[g][:, 1 + k] * acc[:, k]
            if w.any():  # accepted: next holdings as under "none", plus product k
                on = idx[(idx >> k) & 1 == 1]
                P[:, on] += w[:, None] * (Pn[:, on] + Pn[:, on ^ (1 << k)])
        V[g] = np.linalg.solve(np.eye(H) - GAMMA * P, (pi[g] * R).sum(axis=1))

        V_next = np.stack([V[g]] + [V[g][idx | (1 << k)] for k in range(K)], axis=1)
        PnV = Pn @ V_next
        Q[g, :, 0] = R[:, 0] + GAMMA * PnV[:, 0]
        Q[g, :, 1:] = R[:, 1:] + GAMMA * (acc * PnV[:, 1:] + (1 - acc) * (Pr @ V[g])[:, None])
    return V, Q


def optimize(model):
    """Bellman optimality equation  V*(s) = max_a Q*(s,a), by policy iteration:
    evaluate the policy exactly, switch to any strictly better action, repeat
    until no state can improve. Returns V*[g, h] and the policy[g, h].
    """
    G, H, K = model["acc"].shape
    policy = np.zeros((G, H), dtype=int)
    while True:
        V, Q = evaluate_policy(model, np.eye(K + 1)[policy])
        Q = np.where(model["valid"], Q, -np.inf)
        best = Q.argmax(axis=2)
        better = Q.max(axis=2) > np.take_along_axis(Q, policy[..., None], axis=2)[..., 0] + 1e-9
        if not better.any():
            return V, policy
        policy = np.where(better, best, policy)

"""Step 3 - score the model against the hidden truth (only possible with mock data).

Simulates the true future of a sample of current customers many times, under
(a) the bank's historical behaviour and (b) the learned policy, and compares
the realised discounted profit with what the model predicted.

    uv run python evaluate.py [--sample N] [--rollouts R]
"""

import argparse

import numpy as np
import pandas as pd

from src import mdp, world

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=3000)
parser.add_argument("--rollouts", type=int, default=30)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

products, customers, holdings, offers = mdp.load_tables()
model = np.load("data/model.npz")
panel = mdp.build_panel(products, customers, holdings, offers)
K = len(products)

now = panel["mask"][:, -1]
sample = rng.choice(np.flatnonzero(now > 0), args.sample, replace=False)
g, h = panel["segment"][sample], now[sample]
appetite, churn_mult = world.traits(customers.iloc[sample])

# Each sampled customer is simulated `rollouts` times.
rep = np.repeat(np.arange(args.sample), args.rollouts)
h0 = mdp.bits_of(K)[h[rep]] == 1
pow2 = 1 << np.arange(K)


def learned_policy(held, rows, rng):
    return model["policy"][g[rep][rows], held @ pow2]


def simulate(policy):
    v = world.rollout(h0, appetite[rep], churn_mult[rep], policy, rng)
    return v.reshape(args.sample, args.rollouts).mean(axis=1)


true_current = simulate(world.behaviour_policy)
true_learned = simulate(learned_policy)
pred_current = model["V_current"][g, h]
pred_optimal = model["V_optimal"][g, h]

print(f"{args.sample:,} customers x {args.rollouts} simulated futures each\n")
print("CLV under the current policy, by predicted-CLV quintile (mean per customer):")
q = pd.qcut(pred_current, 5, labels=False)
table = pd.DataFrame({"predicted": pred_current, "true": true_current}).groupby(q).mean()
table["error"] = table["predicted"] / table["true"] - 1
table.index = [f"Q{i + 1}" for i in table.index]
print(table.to_string(formatters={"predicted": "{:,.0f}".format, "true": "{:,.0f}".format, "error": "{:+.1%}".format}))
print(f"\noverall   predicted {pred_current.mean():,.0f} | true {true_current.mean():,.0f} "
      f"| error {pred_current.mean() / true_current.mean() - 1:+.1%}")
print(f"per-customer correlation(predicted, true): {np.corrcoef(pred_current, true_current)[0, 1]:.3f}")

print("\nLearned policy (mean CLV per customer):")
print(f"  true, current policy   {true_current.mean():>10,.0f}")
print(f"  true, learned policy   {true_learned.mean():>10,.0f}  ({true_learned.mean() / true_current.mean() - 1:+.1%})")
print(f"  model's own prediction {pred_optimal.mean():>10,.0f}  ({pred_optimal.mean() / true_learned.mean() - 1:+.1%} vs true)")

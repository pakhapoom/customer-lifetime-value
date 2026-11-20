"""Step 0 - write the mock tables to data/.

    uv run src/mock_data.py [--customers N] [--months M] [--seed S]
"""

import argparse
from pathlib import Path

import world

parser = argparse.ArgumentParser()
parser.add_argument("--customers", type=int, default=50_000)
parser.add_argument("--months", type=int, default=53, help="36 to train on + 17 more for evaluation")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

data = Path("data")
data.mkdir(exist_ok=True)
customers, snapshot, transactions = world.simulate_history(args.customers, args.months, args.seed)
world.PRODUCTS.to_csv(data / "products.csv", index=False)
customers.to_csv(data / "customers.csv", index=False)
snapshot.to_csv(data / "holdings_snapshot.csv", index=False)
transactions.to_csv(data / "transactions.csv", index=False)
print(
    f"products {len(world.PRODUCTS)} in {len(world.GROUPS)} groups | customers {len(customers):,} "
    f"| snapshot holdings {len(snapshot):,} | transactions {len(transactions):,} over {args.months} months"
)

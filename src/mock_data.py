"""Step 0 - write the mock tables to data/.

    uv run src/mock_data.py [--customers N] [--months M] [--seed S]
"""

import argparse
from pathlib import Path

import world

parser = argparse.ArgumentParser()
parser.add_argument("--customers", type=int, default=20_000)
parser.add_argument("--months", type=int, default=36)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

data = Path("data")
data.mkdir(exist_ok=True)
customers, holdings, offers = world.simulate_history(args.customers, args.months, args.seed)
world.PRODUCTS.to_csv(data / "products.csv", index=False)
customers.to_csv(data / "customers.csv", index=False)
holdings.to_csv(data / "holdings.csv", index=False)
offers.to_csv(data / "offers.csv", index=False)
print(f"products {len(world.PRODUCTS)} | customers {len(customers):,} | holdings {len(holdings):,} | offers {len(offers):,}")

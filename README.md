# Customer Lifetime Value as an MDP — lab

Learn a Markov decision process from (mock) product-holding history, then use the
Bellman equation to predict each customer's CLV and the next best offer.

```bash
uv sync
uv run python mock_data.py   # 0. write mock tables to data/
uv run python train.py       # 1. training:  tables -> MDP -> Bellman -> data/model.npz
uv run python predict.py     # 2. inference: CLV + next best action -> data/predictions.csv
uv run python evaluate.py    # 3. score against the hidden truth (mock only)
uv run pytest -q
```

## Mocked data (`data/`)

| file | columns | notes |
|---|---|---|
| `products.csv` | `product_id, name, category, tier, est_monthly_revenue` | 10 retail-bank products, P001–P010 (below) |
| `customers.csv` | `customer_id, age, monthly_income, occupation, signup_date` | 20,000 customers |
| `holdings.csv` | `customer_id, product_id, purchase_date, end_date` | one row per purchase; empty `end_date` = still held. A customer can hold several products at once |
| `offers.csv` | `customer_id, product_id, offer_date, accepted` | the bank's historical actions: an offer in about 25% of customer-months, for a random product not held |

`offers.csv` is what makes the effect of an action learnable. Purchases alone show what
customers did, not what the bank did.

| id | product | category | tier | revenue / month |
|---|---|---|---|---|
| P001 | Savings Account | deposit | basic | 3 |
| P002 | Fixed Deposit | deposit | standard | 8 |
| P003 | Debit Card | card | basic | 2 |
| P004 | Credit Card Classic | card | standard | 15 |
| P005 | Credit Card Platinum | card | premium | 40 |
| P006 | Personal Loan | loan | standard | 35 |
| P007 | Home Loan | loan | elite | 120 |
| P008 | Life Insurance | insurance | premium | 30 |
| P009 | Mutual Fund | investment | premium | 45 |
| P010 | Wealth Management | investment | elite | 150 |

Hidden behaviour in the mock world (`src/world.py`; the model never reads it):

- Each product has its own monthly churn rate (0.8%–4%).
- Saying yes to an offer depends on the product, age, occupation, income vs. tier, and
  **current holdings** (e.g. a deposit raises appetite for cards; a loan for insurance;
  holding one tier raises appetite for the next tier up).
- Customers also buy unprompted, at a low rate.
- More products held → each is stickier. A rejected offer raises churn that month.
- A customer whose last product ends has left for good.

## MDP formulation (`src/mdp.py`)

- **Time step** one month. Event dates are bucketed into months.
- **State** `(segment, holdings)`: segment = age band × income band (9); holdings = the
  exact set of products held (2^10 = 1,024). 9,216 states. Holding nothing = churned, absorbing.
- **Action** `none` or `offer_P001 … offer_P010` (11). Offering a held product is not allowed.
- **Reward** sum of `est_monthly_revenue` of held products − 2 per offer.
- **Discount** γ = 0.99 per month.
- **Transitions**, learned in three pieces:
  - `P_none`: next holdings after a no-offer month (counts).
  - `P_rej`: next holdings after a rejected offer (counts).
  - `acc[segment, holdings, product]`: probability the offer is accepted.
  - Offer k then = `acc` × (`P_none` with product k added) + (1 − `acc`) × `P_rej`.
- **Sparse data**: most of the 9,216 states are rarely or never seen, so every estimate is
  shrunk: segment row → all-segments row → a prior in which each product churns and is
  bought independently at its own rate.

CLV = V(state), from the Bellman equation:

- **current policy** — `V = R_π + γ P_π V`, solved exactly as a linear system per segment
- **optimal policy** — `V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]`, by policy iteration

## Results (seed 0: 20k customers, 36 months, 340k customer-months)

`evaluate.py` simulates the true future of 3,000 current customers 30 times each.

CLV under the current policy, mean per customer:

| predicted-CLV quintile | predicted | true | error |
|---|---|---|---|
| Q1 (lowest) | 2,336 | 1,840 | +27.0% |
| Q2 | 5,229 | 4,619 | +13.2% |
| Q3 | 8,807 | 8,821 | −0.2% |
| Q4 | 11,092 | 11,553 | −4.0% |
| Q5 (highest) | 16,453 | 17,988 | −8.5% |
| **overall** | **8,618** | **8,785** | **−1.9%** |

Per-customer correlation between predicted and true CLV: 0.93.

Learned policy, mean CLV per customer:

| | value |
|---|---|
| true, current policy | 8,785 |
| true, learned policy | 23,314 (+165%) |
| model's own prediction of the optimum | 27,629 (+18.5% too high) |

Reading these:

- The portfolio total is accurate and customers are ranked well, but predictions are
  pulled toward the middle: low-CLV customers are over-predicted and high-CLV ones
  under-predicted. Nine segments are coarser than the true drivers (exact income, occupation).
- The optimal-policy CLV is over-predicted. The `max` in the Bellman optimality equation
  favours actions whose estimates happen to be too high, and the best policy steers
  customers into holdings combinations the history barely covers.
- The +165% uplift reflects how well offers work in this invented world. It is not a
  real-world estimate.
- Customer churn in this world is low (8% of customers left within the window), so most
  of a CLV comes from products the customer is expected to buy later.

## Ideas to try next

- Finer or learned segments (or a regression model for acceptance and churn) to fix the
  pull toward the middle.
- Pessimism: optimise against a lower confidence bound to curb the optimal-policy over-prediction.
- Back-test: train on the first 24 months, compare with realised revenue in the last 12.
- More products: 2^K states stops scaling around K ≈ 12; beyond that the holdings need to
  be compressed (e.g. counts per tier) or the value function approximated.

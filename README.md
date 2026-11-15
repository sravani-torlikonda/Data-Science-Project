# IPL Auction Price Prediction: Is This Player Worth His Auction Money?

**Question:** Can we predict how much an IPL team will pay for a player at the auction, using only
what is known before the auction (base price, nationality and his past IPL performance), and use
the model to spot the most overpaid players and the biggest bargains?

**Data (included in `data/`):**

| File | What it is |
|---|---|
| `auction_2013_2023.csv` | Every player sold at IPL auctions 2013–2023 (Kaggle IPL auction dataset; prices in lakhs) |
| `auction_2025.csv` | IPL 2025 mega auction results (prices in crores; also lists retained and unsold players) |
| `auction_list_2025.csv` | Official TATA IPL 2025 auction list (nationality, role) |
| `ipl_ball_by_ball.csv.gz` | Every ball of every IPL match 2008–2026, from Cricsheet via the open `ritesh-ojha/IPL-DATASET` repo |
| `ipl_matches.csv`, `player_names_2024.csv` | Match dates (to get seasons) and full-name/short-name pairs for 2024 squads |

**Full report:** [`Project_Report.pdf`](Project_Report.pdf)

After cleaning: **1,230 players sold at 12 auctions** (2013–2023 and 2025; 2024 data was not available).

**Result (test auctions 2023 + 2025, 261 players):**

| Model | MAE (₹ crore) | R² (log price) |
|---|---|---|
| Typical-price baseline | 2.67 | −0.06 |
| Base-price baseline ("sells at base") | 2.22 | 0.33 |
| Linear Regression | 1.91 | 0.66 |
| Random Forest | 1.90 | 0.66 |
| **Gradient Boosting (chosen)** | **1.84** | **0.67** |

## How to run

Needs Python 3.9 or newer. The whole project is in **one file**, `ipl_auction_project.py`.

```bash
pip install -r requirements.txt
python ipl_auction_project.py
```

It takes under a minute (close each plot window to continue). In VS Code you can also click
**Run Cell** above each `# %%` line to go through it step by step.

All randomness uses a fixed seed (`SEED = 42`), so results repeat exactly.
Plots go to `outputs/figures/`, tables (including the overpaid/bargain lists) to `outputs/`.

## What is inside `ipl_auction_project.py`

| Step | What it does |
|---|---|
| Step 1 | Settings and helper functions: cleaning, season stats, name matching, feature building |
| Step 2 | Look at the three sources and find the problems |
| Step 3 | One row per sold player; train on 2013–2022 auctions, test on 2023 + 2025 |
| Step 4 | 5 EDA plots with insights |
| Step 5 | Cleaning report, name matching, features, correlations |
| Step 6 | 2 baselines + 3 models, 5-fold cross-validation grouped by auction year, tuning |
| Step 7 | Test results, error analysis, overpaid/bargain players 2025, feature importance |

## Data sources

- IPL auction data 2013–2023 and 2025: Kaggle IPL auction datasets
- TATA IPL 2025 Auction List: Kaggle
- Ball-by-ball data: [Cricsheet](https://cricsheet.org) via [ritesh-ojha/IPL-DATASET](https://github.com/ritesh-ojha/IPL-DATASET) (MIT licence)

# WNBA Moneyline Prediction Model

Machine learning ensemble for WNBA moneyline value betting.
Same architecture as the NBA model — adapted for WNBA data and season structure.

---

## Key Differences from NBA Model

| | NBA | WNBA |
|--|-----|------|
| League ID | `00` | `10` |
| Season format | `2024-25` | `2024` |
| Season dates | Oct–Apr | May–Oct |
| Teams | 30 | 12 |
| Games/season | ~1,230 | ~240 |
| Data API | stats.nba.com | stats.wnba.com |

---

## Quick Start

```bash
python -m venv wnba_model
wnba_model\Scripts\activate       # Windows
source wnba_model/bin/activate    # Mac/Linux

pip install -r requirements.txt

# Full pipeline (first time — takes ~30 min)
python run_pipeline.py --fetch --features --train

# Daily predictions
python predict.py

# Dashboard
python -m streamlit run dashboard_app.py
```

---

## Pipeline Commands

```bash
python run_pipeline.py --fetch              # Fetch WNBA game data
python run_pipeline.py --features           # Build feature matrix
python run_pipeline.py --train              # Train ensemble
python predict.py                           # Today's predictions
python predict.py --season 2025             # Specific season
python predict.py --odds "LAS:-150,IND:+130;NYL:-200,CON:+170"
```

---

## Backtesting

```bash
# Scrape historical odds (once per season)
python src/sbr_scraper.py --from 2024-05-14 --to 2024-09-22 --out data/odds/sbr_2024.csv

# Backtest
python backtest_real_odds.py --csv data/odds/sbr_2024.csv --file_season 2024 --edge 15

# All seasons
python backtest_real_odds.py --all --edge 15
```

---

## WNBA Season Dates

| Season | Start | End |
|--------|-------|-----|
| 2025 | ~May 16, 2025 | ~Sep 21, 2025 |
| 2024 | May 14, 2024 | Sep 22, 2024 |
| 2023 | May 19, 2023 | Sep 10, 2023 |
| 2022 | May 6, 2022 | Sep 18, 2022 |
| 2021 | May 14, 2021 | Sep 19, 2021 |
| 2020 | Jul 25, 2020 | Oct 6, 2020 (bubble) |
| 2019 | May 25, 2019 | Sep 8, 2019 |
| 2018 | May 18, 2018 | Sep 9, 2018 |

---

## WNBA Team Abbreviations

| Abbr | Team |
|------|------|
| ATL | Atlanta Dream |
| CHI | Chicago Sky |
| CON | Connecticut Sun |
| DAL | Dallas Wings |
| IND | Indiana Fever |
| LAS | Las Vegas Aces |
| LA  | Los Angeles Sparks |
| MIN | Minnesota Lynx |
| NYL | New York Liberty |
| PHO | Phoenix Mercury |
| SEA | Seattle Storm |
| WAS | Washington Mystics |

---

## Betting Filters (config/settings.py)

| Filter | Default | Notes |
|--------|---------|-------|
| Min edge | 15% | Optimise after backtesting |
| Underdogs only | True | Start conservative |
| Min odds | +141 | Near-even odds unreliable |
| Max odds | +500 | Longshots too volatile |
| Max edge cap | 30% | Overconfidence filter |

**Note:** WNBA markets are less efficient than NBA — you may find larger edges
and could potentially lower the min edge threshold after backtesting.

---

## Deployment

Same as NBA model — Streamlit Cloud + GitHub Actions.
GitHub Action runs at 11pm UTC (7am MYT) daily, May–October only:

```yaml
# To restrict to WNBA season only, update cron in .github/workflows/daily_predictions.yml
# May–October: months 5–10
on:
  schedule:
    - cron: '0 23 * 5-10 *'   # Only runs May through October
```

---

## License

Private. All rights reserved.

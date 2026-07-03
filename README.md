# NBA All-Star Predictor

Binary classification model predicting NBA All-Star selections 
using per-game player statistics from 1980–2026.

## Dataset
NBA player stats from Basketball Reference via Kaggle.
33,000+ player seasons across 75+ years.

## Approach
- Merged All-Star selection data as binary target (6.5% positive rate)
- Features: points, assists, rebounds, steals, blocks, 
  minutes, FG%, FT% per game
- Handled class imbalance with balanced class weights
- Models: Logistic Regression and Random Forest
- Evaluation: ROC-AUC, precision/recall, confusion matrix

## Results
| Model | ROC-AUC | F1 (default) | F1 (tuned threshold) |
|-------|---------|--------------|----------------------|
| Logistic Regression | 0.977 | 0.52 | 0.67 |
| Random Forest | 0.974 | 0.63 | 0.67 |

Optimal thresholds: LR=0.902, RF=0.340

## Key Finding
Minutes per game was the strongest predictor (0.30 importance), 
slightly edging points (0.29). This suggests the model partially 
learns playing time as a proxy for All-Star candidacy — starters 
accumulate the stats needed for selection. Logistic Regression 
maximises recall at the cost of precision; Random Forest is more 
conservative but more precise.

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Structure
```
├── data/               # Raw CSVs (not tracked)
├── notebooks/
│   └── 01_eda.ipynb   # Full analysis
└── results/            # Output visualisations
```
Link to the Python notebook file: https://github.com/NikhilSubramani/NBA-player-metrics-analysis/blob/main/NBA-player-analysis/Notebooks/initial-eda.ipynb

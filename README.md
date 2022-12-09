# COMP4601 - Assignment 2

## Description

Within this assignment, we perform some experimental analysis on a historical movie review dataset. The goal of the assignment is to determine which algorithm (user-based or itembased) and parameter combination (top-K neighbours or similarity threshold, and associated values) provide the best recommendation solution for the given dataset.

## Contributors

* Tom Lam (SN: 101114541)
* Minh Thang Cao (SN: 101147025)

## Getting Started

### Dependencies

* `python3`
* `pip`

### Installing

```
pip install -r requirements.txt
```

### Executing program

Run validator. This takes a few days for an average computer to complete. The result of this has already been included in `assignment2-mae-results.json`
```
python a2.py
```

Generate graphs from the result file
```
python graphs/graph-generator.py
```
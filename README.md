# 🏆 Time Series Forecasting Benchmark

### Foundation Models · AutoML · Modern Tabular ML

> **Walmart Weekly Sales** · Multi-series (7 departments) · Horizon = 5 weeks · Strict out-of-time validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Why this notebook exists

Most forecasting tutorials stop at a single model and a single series.  
This repository is a **production-oriented tournament** that answers three practical questions:

1. **How much accuracy do you really gain** by moving from a hand-crafted recursive LightGBM to a global Numba engine, a zero-shot foundation model, or a full AutoML ensemble?
2. **Which signal-processing diagnostics** (Welch PSD, grouped PACF, STL) actually change the feature set you ship?
3. **What is the true latency / training cost** of each paradigm under identical data and metrics?

The notebook is deliberately written so you can copy any single contender into your own pipeline in minutes.

---

## Contenders

| # | Architecture | Type | Training | Inference |
|---|--------------|------|----------|-----------|
| 1 | **skforecast** | Local recursive LightGBM + rolling features | Fast | Sequential |
| 2 | **Manual recursive loop** | Explicit step-by-step (`feature-engine` + LGBM) | Fast | Sequential |
| 3 | **mlforecast** | Global LightGBM (Numba engine) | Very fast | Vectorised |
| 4 | **Google TimesFM 500M** | Zero-shot Transformer foundation model | None | GPU / CPU |
| 5 | **AutoGluon-TimeSeries** | Weighted ensemble (Chronos + DL + GBDTs) | Medium | Fast |

All models are evaluated with the same **WAPE / MAE / RMSE / R²** on a pure out-of-time hold-out of the last 5 weeks of every department.

---

## Methodological highlights

### Signal diagnostics that actually matter
- **Welch Power Spectral Density** → isolates the 52-week, 26-week and ≈4.3-week cycles that drive Fourier design.
- **Grouped PACF** with theoretical ±2/√N bands → selects leakage-safe lags (Lag ≥ H).
- **STL decomposition** → demonstrates why *rigid* seasonal indices fail and why **Fourier harmonics + calendar signatures** are superior for irregular retail cycles.

### Feature engineering
A hybrid space that combines:
- Fourier sine/cosine terms for periods {1, 4.33, 9, 14, 26, 52} (order ≤ 2)
- Time-signature attributes (week-of-year, month, quarter, …)
- Explicit holiday flags
- Modular Scikit-Learn transformers (`Flagnan3`, `Flagout2`, `TargetEncoderFastPandas2`, …)

### Evaluation utilities
Three-panel diagnostic plots (test zoom · full context · bias scatter) are available for both single-department and fully aggregated views.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/franciscogamarra10/timeseries-forecasting-benchmark.git
cd timeseries-forecasting-benchmark

# 2. Create environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or just run the install cell inside the notebook

# 3. Launch
jupyter lab TimeSeries_Forecasting_Benchmark.ipynb
```

The first code cell installs everything needed (`pytimetk`, `skforecast`, `mlforecast`, `autogluon.timeseries`, `timesfm`, …).  
A single T4 GPU accelerates TimesFM; every other model runs happily on CPU.

---

## Repository layout

```
├── TimeSeries_Forecasting_Benchmark.ipynb   # main notebook
├── README.md                                # this file
├── CHANGES.md                               # detailed changelog of the conversion
├── requirements.txt                         # optional pinned deps
└── (your custom modules)
    ├── pandas_transformers.py
    ├── dashboards2.py
    └── loaders.py
```

> The custom modules (`pandas_transformers`, `dashboards2`, `loaders`) are my own production feature-engineering library.  
> They can be swapped for equivalent `feature-engine` / `sklearn` / skrub` steps without changing the modelling logic.

---

## Key take-aways (spoiler-free)

- Multi-scale **Fourier features** consistently improve tree-based models once the spectral analysis has been performed.
- A **global LightGBM / mlforecast** baseline is already competitive and extremely fast.
- **TimesFM** gives a strong zero-shot reference with zero training cost.
- **AutoGluon** remains the accuracy champion when wall-time and transparency are secondary.
- Always evaluate with a **strict out-of-time split** and a scale-free metric (WAPE).

---

## Citation

If this benchmark helps your research or production work, please star the repository and cite:

```
@misc{timeseries-forecasting-benchmark,
  author = {Francisco Gamarra},
  title  = {Time Series Forecasting Benchmark: Foundation Models vs AutoML vs Tabular ML},
  year   = {2025},
  url    = {https://github.com/franciscogamarra10/timeseries-forecasting-benchmark}
}
```

---

## License

MIT – use it, fork it, ship it.

# Forecasting Optimizer

Forecasting Optimizer is an experimental, configuration-driven framework for
comparing complete time-series forecasting pipelines. An outer Optuna study
selects preprocessing, feature-engineering, and model choices; individual model
adapters can also tune their own hyperparameters with cross-validation.

The project is an installable `forecasting_optimizer` package with source under
`src/`. It uses uv for dependency resolution, environment management, command
execution, locking, and package builds.

## Pipeline

Each optimization trial follows the same stages:

1. **Data landing** calls the active experiment's zero-argument loader and
   stores the returned pandas DataFrame.
2. **Exploratory analysis** adds missing timestamps at the configured frequency
   while preserving existing timestamps, summarizes missing values, and
   optionally renders plots.
3. **Preprocessing** runs the experiment hook, creates train/validation/test
   splits, and applies the selected imputer.
4. **Feature engineering** runs the experiment hook, encodes categorical
   values, and applies configured outlier handling.
5. **Modeling** tunes, fits, and evaluates the selected estimator, then records
   its predictions and performance metrics.
6. **Optimization** scores the configured dataset and reruns the best trial.

## Model adapters

| Family | Implementations | Status |
| --- | --- | --- |
| Linear | Linear regression, Ridge, Lasso, Elastic Net | Concrete adapters |
| Boosted trees | XGBoost, LightGBM | Concrete adapters |
| Statistical | SARIMAX through `pmdarima.auto_arima` | Concrete adapter |
| Neural sequence | CNN, GRU, LSTM | Concrete adapters |
| Scaffolds | Prophet, NeuralProphet, Transformer, BERT | Registered but incomplete |

The H2O class is not registered with the model dispatcher. For outlier
handling, only Isolation Forest currently removes observations; the boxplot,
z-score, and one-class SVM names are placeholders.

A feature-scaling helper exists, but the current feature-engineering execution
path does not call it. Selected scalers are instead passed into model adapters,
whose scaling behavior varies by model family.

## Repository layout

Except for the package root itself, paths in this table are relative to
`src/forecasting_optimizer/`.

| Path | Purpose |
| --- | --- |
| `src/forecasting_optimizer/` | Contains the installable Python package. |
| `experiment_settings.py` | Selects the active experiment module. |
| `experiment_configs/` | Declares datasets, splits, hooks, models, and search settings. |
| `projects/` | Contains source-specific loaders and feature hooks. |
| `eda/` | Adds configured timestamps and calculates exploratory summaries. |
| `data_preprocessing/` | Splits and imputes landed data. |
| `feature_engineering/` | Generates, encodes, and filters features. |
| `modeling/` | Implements model adapters, tuning, fitting, and evaluation. |
| `optimization/` | Coordinates the outer Optuna study. |
| `util/` | Provides persistence, distribution, sequence, and BigQuery helpers. |

## Setup

Python 3.9 through 3.13 and
[uv](https://docs.astral.sh/uv/getting-started/installation/) are required. uv
creates and maintains the project virtual environment automatically.

```bash
uv sync
mkdir -p output_data tmp/stocks
```

`pyproject.toml` is the dependency source of truth and `uv.lock` records the
resolved environment. Add dependencies with `uv add <package>` and development
dependencies with `uv add --dev <package>`.

The stock and contact-forecasting support modules create BigQuery clients when
they are imported, so they require valid Google Cloud Application Default
Credentials even when their local-data branches are used. Remote loaders also
require access to the configured project and datasets.

## Configuration

Set `EXPERIMENT_CONSTANTS_MODULE_NAME` in `experiment_settings.py` to one of the
modules in `experiment_configs/`, then review that module's settings. The core
fields read directly by the pipeline are:

- `DATA_LOADING_FUNCTION`: a zero-argument callable returning a time-indexed
  pandas DataFrame;
- `TARGET_COL`, plus `TRAIN_VAL_SPLIT_DATE` and `VAL_TEST_SPLIT_DATE` (the
  split-date values may be `None` to use calculated defaults);
- `TIMESERIES_INTERVAL_VALUE`, `TIMESERIES_INTERVAL_UNIT`, and
  `TIMESERIES_TARGET_FILL_GAP_VALUE`; and
- `PREPROCESSING_FUNCTION`, which may be `None` or a callable with the
  signature `(trial, dataframe)`.

When provided, `FEATURE_ENGINEERING_FUNCTION` must use that same callback
signature; it falls back to a no-op when absent. Model lists,
performance-dataset selection, imputers, scalers, cross-validation, and Optuna
settings have framework fallbacks. The optimizer constructs its one-hot encoder
directly. The `CATEGORICAL_FEATURE_ENCODER` declarations in example
configurations are not read by the current execution path.

The feature-stage outlier choice comes directly from `OUTLIER_METHODS`.
`REMOVE_OUTLIERS` does not gate that path, so use `OUTLIER_METHODS = [None]` to
disable feature-stage row removal.

The bundled configurations are examples tied to their original environments:

- `stocks` is selected by default and reads `tmp/stocks/PLTR.csv` and
  `tmp/stocks/nasdaq.csv` when local mode is enabled;
- `e2e` reads `tmp/base_data.pickle` or a private BigQuery dataset;
- `contact_forecasting` reads `output_data/cf_raw_data.pickle`; its disabled
  remote branch queries private BigQuery data; and
- `slf` imports a supplier-forecasting project from a hard-coded external path
  and needs interval settings and hook-signature adaptation.

Adapt a configuration and its loader before expecting a bundled experiment to
run in a new environment. The default command-line runner also uses a fixed
experiment ID and requests 200 outer trials; review `__main__.py` before a
long-running experiment.

Some retained helpers use the removed NumPy `np.int`/`np.float` aliases, and
the stock feature hook uses the removed pandas `DatetimeIndex.week` attribute.
Modern releases therefore raise `AttributeError` on those legacy paths; the
package migration leaves their executable behavior unchanged.

## Run

Create all required local data/cache directories, then run the installed
console command from the repository root:

```bash
uv run forecasting-optimizer
```

The equivalent module invocation is `uv run python -m forecasting_optimizer`.
Running from the repository root keeps the bundled configurations' writable
`output_data/` and `tmp/` paths in the expected location.

The optimizer can also be called directly from Python. Run it from a working
directory containing the configured runtime files and writable output paths:

```python
from forecasting_optimizer import Optimizer

optimizer = Optimizer(experiment_id="my-experiment")
best_parameters = optimizer.execute_optimization(
    direction="minimize",
    n_trials=20,
)
```

Constructing `Optimizer` immediately loads and prepares the active experiment's
data.

## Output artifacts

The framework writes HDF DataFrames and pickle summaries beneath
`output_data/`, namespaced by experiment ID. Typical artifacts include landed,
EDA, processed split, engineered split, prediction, performance, and final
result files. Working outer-trial summaries are stored at
`tmp/optimizer_best_model_params_dict.pickle`.

Candidate trials reuse the experiment ID, so generic stage artifacts for train,
validation, and test splits are overwritten as later trials run.

`<experiment-id>_final_result.pickle` contains the selected parameter summary;
it is not a serialized fitted estimator or executable pipeline. Only load
pickle files from trusted sources because unpickling can execute code.

## Development and style

Formatting, docstrings, and comments were aligned with the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
where that was possible without changing executable behavior. Full style
conformance is not claimed: behavior-sensitive legacy constructs remain,
including catch-all exception handlers, mutable defaults, f-string logging,
and historical names. Formatting and lint settings live in `pyproject.toml`.

```bash
uv sync
uv lock --check
uv run black --check src
# Diagnostic only: the unchanged legacy findings currently make this nonzero.
uv run pylint src/forecasting_optimizer
uv run python -m compileall -q src/forecasting_optimizer
uv build --no-sources
```

There is no automated test suite yet. Before changing pipeline behavior, add
focused tests for the relevant experiment hook or model adapter.

## License

Forecasting Optimizer is distributed under the BSD 3-Clause License. See
`LICENSE.txt`.

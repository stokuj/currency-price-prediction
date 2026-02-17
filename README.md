# CryptoCurrencyPP

Desktop application for cryptocurrency price forecasting with recurrent neural networks (LSTM, GRU, and LSTM+GRU).

## Overview

The project uses a Tkinter GUI (MVC structure) to:
- select a market (`BTC`, `ETH`, `DOGE`, `LTC`),
- choose a model architecture,
- train on online or local CSV data,
- visualize predicted vs. actual prices,
- calculate simple gain over a selected window.

Training runs in a background thread. Progress and ETA are shown in:
- terminal logs,
- GUI status line.

## Tech Stack

- Python `>=3.10,<3.13`
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn
- Matplotlib
- yfinance + pandas-datareader
- Tkinter (Azure theme)
- uv (dependency and environment management)

## Project Structure

```text
.
|-- app.py                 # Application entry point (Controller)
|-- app/
|   |-- model.py           # Data access, training, prediction, gain
|   |-- view.py            # Tkinter GUI
|   |-- azure.tcl          # GUI theme
|   |-- test.csv           # Default local CSV for offline mode
|   |-- assets/
|   `-- images/
|-- DATA/
|   `-- download.py        # Utility script for fetching CSV data
|-- DOC/                   # Report and presentation materials
|-- pyproject.toml
`-- uv.lock
```

## Quick Start

1. Install dependencies and run the app:

```bash
uv sync
uv run python app.py
```

## Using the App

1. Choose currency, data source, and model type.
2. Configure `Prediction days`, `Future days`, and `Plot range`.
3. Click `Train`.
4. Monitor progress in terminal and GUI status line.
5. Click `Plot` to show the chart.
6. Click `GAIN` to compute percent change.

## Data and Date Semantics

- `Prediction days`: lookback window size (how many past observations are used as input).
- `Future days`: forecast horizon in days (for example, `1` means predict one day ahead).
- `Plot range`: number of dated points displayed on the chart.
- Chart X-axis now uses concrete calendar dates (`YYYY-MM-DD`).
- Predicted series is shifted by `Future days`, so each predicted point is plotted at its target date.

## Data Modes

- Online mode: `Yahoo` is used for crypto downloads. Selecting `Stooq` or `Naver` shows an "unsupported" warning and auto-switches back to `Yahoo`.
- Offline mode: disable online switch and choose a CSV file.
  Required column: `Close`.
  Recommended column for proper timeline: `Date`.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Notes

- If a data source returns empty data, the app shows a user-facing error dialog.
- This project is educational and should not be treated as trading advice.


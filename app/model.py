import datetime as dt
import os
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler

# Hide TensorFlow INFO logs in console output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, LSTMCell, RNN
from tensorflow.keras.models import Sequential


class Model:
    """Model class used for program logic."""

    def __init__(self, controller):
        self.controller = controller

        self.ready_to_do_plot = False
        self.crypto_currency = "BTC"
        self.against_currency = "USD"
        self.data_source = "yahoo"
        self.use_online_db = True

        self.start = dt.datetime(2016, 1, 1)
        self.end = dt.datetime.now()
        self.test_start = dt.datetime(2022, 1, 23)

        self.prediction_days = 30
        self.future_day = 1

        self.filename = os.path.join(os.path.dirname(__file__), "test.csv")

        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_train = []
        self.y_train = []

        self.model_id = 1
        self.model = Sequential()

    def _symbol(self):
        return f"{self.crypto_currency}-{self.against_currency}"

    def _provider_symbol(self):
        if self.data_source == "stooq":
            # Stooq expects symbols like BTCUSD instead of BTC-USD.
            return f"{self.crypto_currency}{self.against_currency}"
        return self._symbol()

    def _extract_close_series(self, frame, context):
        # yfinance can return MultiIndex columns (Price/Ticker).
        if isinstance(frame.columns, pd.MultiIndex):
            first_level = frame.columns.get_level_values(0)
            if "Close" in first_level:
                close_df = frame.xs("Close", axis=1, level=0)
            elif "close" in first_level:
                close_df = frame.xs("close", axis=1, level=0)
            else:
                raise ValueError(f"Missing 'Close' column for {context}.")

            if close_df.empty:
                raise ValueError(f"Empty 'Close' series for {context}.")

            # Pick the first ticker column for single-symbol use.
            close_series = close_df.iloc[:, 0]
            close_series.name = "Close"
            return close_series

        if "Close" in frame.columns:
            return frame["Close"].rename("Close")
        if "close" in frame.columns:
            return frame["close"].rename("Close")

        raise ValueError(f"Missing 'Close' column for {context}.")

    def _normalize_close(self, df, context):
        if df is None or df.empty:
            raise ValueError(f"No data returned for {context}.")

        frame = df.copy()

        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.dropna(subset=["Date"])
            if frame.empty:
                raise ValueError(f"Invalid 'Date' column for {context}.")
            frame = frame.set_index("Date")

        close_series = self._extract_close_series(frame, context)
        close_series = pd.to_numeric(close_series, errors="coerce").dropna()
        if close_series.empty:
            raise ValueError(f"Empty 'Close' series for {context}.")

        normalized = pd.DataFrame({"Close": close_series})

        if not isinstance(normalized.index, pd.DatetimeIndex):
            parsed_index = pd.to_datetime(normalized.index, errors="coerce")
            if isinstance(parsed_index, pd.DatetimeIndex) and not parsed_index.isna().all():
                normalized.index = parsed_index

        if not isinstance(normalized.index, pd.DatetimeIndex):
            start_date = dt.date.today() - dt.timedelta(days=len(normalized) - 1)
            normalized.index = pd.date_range(start=start_date, periods=len(normalized), freq="D")

        if normalized.index.hasnans:
            normalized = normalized.loc[~normalized.index.isna()]
        if normalized.empty:
            raise ValueError(f"No valid dated rows for {context}.")

        return normalized.sort_index()

    def _download_online_data(self, start, end):
        symbol = self._provider_symbol()

        if self.data_source == "yahoo":
            df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
            return self._normalize_close(df, f"{symbol} from yahoo")

        if self.data_source == "stooq":
            try:
                df = web.DataReader(symbol, "stooq", start, end)
                return self._normalize_close(df, f"{symbol} from stooq")
            except Exception as exc:
                raise ValueError(f"Failed to download data from 'stooq' for {symbol}: {exc}")

        if self.data_source == "naver":
            raise ValueError(
                "Naver source is not available for these cryptocurrency pairs. "
                "Use Yahoo or Stooq."
            )

        raise ValueError(f"Unsupported data source: {self.data_source}")

    def _load_offline_data(self):
        if not os.path.isfile(self.filename):
            raise ValueError(f"Offline file not found: {self.filename}")

        try:
            df = pd.read_csv(self.filename)
        except Exception as exc:
            raise ValueError(f"Failed to read offline file '{self.filename}': {exc}")

        return self._normalize_close(df, f"offline file {self.filename}")

    def _get_data(self, start, end):
        if self.use_online_db:
            return self._download_online_data(start, end)
        return self._load_offline_data()

    def train(self, progress_callback=None):
        """Main function for training."""
        self.x_train, self.y_train = [], []
        self.model = Sequential()

        self.data = self._get_data(self.start, self.end)
        close_values = self.data["Close"].to_numpy().reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_values)

        if len(scaled_data) <= self.prediction_days + self.future_day:
            raise ValueError("Not enough rows to train with current prediction/future day settings.")

        for x in range(self.prediction_days, len(scaled_data) - self.future_day):
            self.x_train.append(scaled_data[x - self.prediction_days : x, 0])
            self.y_train.append(scaled_data[x + self.future_day, 0])

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        if self.model_id == 1:
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))
        elif self.model_id == 2:
            self.model.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(units=50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))
        else:
            self.model.add(RNN(cell=LSTMCell(50), return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.25))
            self.model.add(GRU(units=50, return_sequences=True))
            self.model.add(Dropout(0.10))
            self.model.add(GRU(units=50))
            self.model.add(Dropout(0.10))
            self.model.add(Dense(units=1))

        class EpochProgressCallback(Callback):
            def __init__(self, epochs, callback):
                super().__init__()
                self.epochs = epochs
                self.callback = callback
                self._durations = []
                self._start = None

            def on_epoch_begin(self, epoch, logs=None):
                self._start = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                duration = time.perf_counter() - self._start
                self._durations.append(duration)
                remaining = self.epochs - (epoch + 1)
                avg_epoch = sum(self._durations) / len(self._durations)
                eta_seconds = int(avg_epoch * remaining)
                loss = None if logs is None else logs.get("loss")
                if self.callback:
                    self.callback(epoch + 1, self.epochs, loss, eta_seconds)

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        epochs = 100
        progress = EpochProgressCallback(epochs=epochs, callback=progress_callback)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[progress],
        )

        self.ready_to_do_plot = True

    def plot(self):
        """Main function for plotting results."""
        if self.data is None or self.data.empty:
            raise ValueError("No training data available. Train the model first.")

        if self.use_online_db:
            test_end = dt.datetime.now() + dt.timedelta(days=self.future_day)
            test_data = self._download_online_data(self.test_start, test_end)
        else:
            test_data = self._load_offline_data()

        actual_prices = test_data["Close"].to_numpy()
        test_dates = pd.to_datetime(test_data.index)

        total_dataset = pd.concat((self.data["Close"], test_data["Close"]), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - self.prediction_days :].to_numpy()
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = self.scaler.fit_transform(model_inputs)

        x_test = []
        for x in range(self.prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - self.prediction_days : x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction_prices = self.model.predict(x_test, verbose=0)
        prediction_prices = self.scaler.inverse_transform(prediction_prices).ravel()

        prediction_dates = test_dates + pd.to_timedelta(self.future_day, unit="D")

        fig, ax = plt.subplots()
        ax.plot(test_dates, actual_prices, color="cyan", label="Actual Prices")
        ax.plot(prediction_dates, prediction_prices, color="indigo", label="Predicted Prices")
        ax.set_title(
            f"{self.crypto_currency}-{self.against_currency} | "
            f"prediction +{self.future_day} day(s)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        plt.show()

    def gain(self):
        """Calculate gain percentage."""
        if self.use_online_db:
            start_date_gain = dt.datetime.now() - dt.timedelta(days=self.prediction_days)
            df = self._download_online_data(start_date_gain, dt.datetime.now())
        else:
            df = self._load_offline_data()

        if len(df) < 2:
            raise ValueError("Not enough data points to calculate gain.")

        x = float(df["Close"].iloc[0])
        y = float(df["Close"].iloc[-1])
        return (y / x) * 100

    def set_currency(self, i):
        mapping = {1: "BTC", 2: "ETH", 3: "DOGE", 4: "LTC"}
        if i in mapping:
            self.crypto_currency = mapping[i]

    def set_data_source(self, i):
        if i == 1:
            self.data_source = "yahoo"
            return True

        # For crypto pairs in this app, keep UI options but fallback to Yahoo.
        if i in (2, 3):
            self.data_source = "yahoo"
            return False

        return True

    def set_model_id(self, i):
        if i in (1, 2, 3):
            self.model_id = i

    def set_file_path(self, path):
        if path:
            self.filename = path

    def set_switch_state(self, i):
        self.use_online_db = i == 1

    def set_prediction_days(self, i):
        self.prediction_days = int(i)

    def set_predition_days(self, i):
        """Backward-compatible alias for typo in old API."""
        self.set_prediction_days(i)

    def set_future_days(self, i):
        self.future_day = int(i)

    def set_test_start_date(self, i):
        self.test_start = dt.datetime.now() - dt.timedelta(days=int(i))


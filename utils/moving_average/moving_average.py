import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas.plotting import register_matplotlib_converters
from pathlib import Path
import sys

rootDir = Path(os.getcwd()).parent
utilsDir = rootDir / "utils"
sys.path.append(str(utilsDir))

register_matplotlib_converters()

from utils import load_data


class MovingAverage:
    def __init__(self, config: dict):
        self.config = config

    def prepare_data(self, filepath):
        self.data = load_data(filepath)
        self.preprocess_data()

    def preprocess_data(self):
        if self.config["TEST_DAYS"]:
            self.train_data = self.data[: -self.config["TEST_DAYS"]]
            self.test_data = self.data[-self.config["TEST_DAYS"] :]
        else:
            self.train_data = self.data
            self.test_data = self.data

    def plot_data(self):
        test_dates = self.test_data.index[self.config["WINDOW_SIZE"] :]
        actual_vals = self.test_data["ProratedQuantity"][self.config["WINDOW_SIZE"] :]
        rolling_mean = (
            self.test_data["ProratedQuantity"]
            .rolling(window=self.config["WINDOW_SIZE"])
            .mean()
        )
        predicted_vals = rolling_mean[self.config["WINDOW_SIZE"] :]
        residuals = self.test_data["ProratedQuantity"] - rolling_mean
        res_vals = residuals[self.config["WINDOW_SIZE"] :]
        z_scores = (residuals - residuals.mean()) / residuals.std()
        anomalies = self.test_data[z_scores.abs() > self.config["THRESHOLD"]]
        upper_bound = predicted_vals + self.config["THRESHOLD"] * res_vals.std()
        lower_bound = predicted_vals - self.config["THRESHOLD"] * res_vals.std()

        plt.figure(figsize=(16, 7), facecolor="lightgray")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %#d"))
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x / 1e6:,.2f}M")
        )

        plt.plot(
            test_dates,
            actual_vals,
            color="blue",
            linewidth=1.75,
            label="Actual",
        )

        plt.plot(
            test_dates,
            predicted_vals,
            color="goldenrod",
            linewidth=2,
            alpha=0.75,
            label="Predicted",
        )

        if self.config["SHOW_BOUNDS"]:
            plt.plot(
                test_dates,
                upper_bound,
                color="goldenrod",
                linestyle="--",
                alpha=0.5,
                label="↑/↓ Thresholds",
            )
            plt.plot(
                test_dates,
                lower_bound,
                color="goldenrod",
                linestyle="--",
                alpha=0.5,
            )
            plt.fill_between(
                test_dates,
                lower_bound,
                upper_bound,
                color="goldenrod",
                alpha=0.125,
            )

        if self.config["PLOT_ANOMALIES"]:
            plt.scatter(
                anomalies.index,
                anomalies["ProratedQuantity"],
                color="red",
                s=80,
                label="Anomalies",
            )

        tenant = self.test_data["Tenant"].iloc[0]
        title = f"Daily Billing Data for {tenant} (past {len(self.test_data[self.config["WINDOW_SIZE"] : ]) + 1} days)"
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Date", fontsize=14, fontweight="bold", labelpad=15)
        plt.ylabel("ProratedQuantity", fontsize=14, fontweight="bold", labelpad=15)
        plt.xticks(
            self.test_data.index[self.config["WINDOW_SIZE"] :: 4],
            rotation=45,
            fontsize=9,
        )
        min_val = min(min(actual_vals), min(predicted_vals))
        max_val = max(max(actual_vals), max(predicted_vals))
        num_ticks = 8
        yticks = np.linspace(min_val, max_val, num_ticks)
        plt.yticks(yticks, fontsize=9)
        plt.legend(frameon=False, fontsize=10, loc="upper right")
        plt.show()

    def run_model(self, filepath):
        self.prepare_data(filepath)
        self.plot_data()

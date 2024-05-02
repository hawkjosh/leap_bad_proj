import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas.plotting import register_matplotlib_converters
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


rootDir = Path(os.getcwd()).parent
utilsDir = rootDir / "utils"
sys.path.append(str(utilsDir))

register_matplotlib_converters()

from utils import load_data


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config["DROPOUT"])
        self.d_model = config["D_MODEL"]
        pe = torch.zeros(5000, config["D_MODEL"])
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config["D_MODEL"], 2).float()
            * (-np.log(10000.0) / config["D_MODEL"])
        )
        decay = torch.exp(-config["DECAY_RATE"] * position)
        pe[:, 0::2] = torch.sin(position * div_term) * decay
        pe[:, 1::2] = torch.cos(position * div_term) * decay
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.d_model).float()) + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Linear(in_features=1, out_features=config["D_MODEL"])
        self.pos_encoder = PositionalEncoding(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"], nhead=config["N_HEAD"]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["NUM_LAYERS"], enable_nested_tensor=False
        )
        self.decoder = nn.Linear(in_features=config["D_MODEL"], out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x


class Transformer:
    def __init__(self, config):
        self.config = config
        self.model = TransformerModel(self.config)
        self.scaler = MinMaxScaler()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

    def prepare_data(self, filepath):
        self.data = load_data(filepath)
        self.preprocess_data()
        self.create_sequences()
        self.create_loaders()

    def preprocess_data(self):
        if self.config["TEST_DAYS"]:
            train_data = self.data[: -self.config["TEST_DAYS"]]
            self.test_data = self.data[-self.config["TEST_DAYS"] :]
        else:
            train_data = self.data
            self.test_data = self.data

        self.pq_train = train_data["ProratedQuantity"].to_numpy().reshape(-1, 1)
        self.pq_test = self.test_data["ProratedQuantity"].to_numpy().reshape(-1, 1)

        self.scaler = MinMaxScaler()
        self.pq_train = self.scaler.fit_transform(self.pq_train).flatten().tolist()
        self.pq_test = self.scaler.transform(self.pq_test).flatten().tolist()

    def create_sequences(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        for i in range(len(self.pq_train) - self.config["SEQ_SIZE"]):
            window = self.pq_train[i : i + self.config["SEQ_SIZE"]]
            after_window = self.pq_train[i + self.config["SEQ_SIZE"]]
            self.x_train.append(window)
            self.y_train.append(after_window)
        for i in range(len(self.pq_test) - self.config["SEQ_SIZE"]):
            window = self.pq_test[i : i + self.config["SEQ_SIZE"]]
            after_window = self.pq_test[i + self.config["SEQ_SIZE"]]
            self.x_test.append(window)
            self.y_test.append(after_window)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).view(
            -1, self.config["SEQ_SIZE"], 1
        )
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).view(
            -1, self.config["SEQ_SIZE"], 1
        )
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def create_loaders(self):
        train_dataset = TensorDataset(self.x_train, self.y_train)
        test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def train_model(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["LEARN_RATE"], weight_decay=0.01
        )

        for epoch in range(self.config["NUM_EPOCHS"]):
            self.model.train()
            for batch in self.train_loader:
                x_batch, y_batch = batch

                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_losses = []
            self.predictions = []
            with torch.no_grad():
                for batch in self.test_loader:
                    x_batch, y_batch = batch
                    outputs = self.model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    self.predictions.extend(outputs.squeeze().tolist())

            val_loss = np.mean(val_losses)

            if (epoch + 1) == 1:
                print("Training...", end="\n\n\n")
                print("Epoch    |  Loss")
            if (epoch + 1) % (self.config["NUM_EPOCHS"] // 10) == 0:
                if (epoch + 1) < 100:
                    print(
                        f"{epoch + 1}/{self.config["NUM_EPOCHS"]}   |  {val_loss:.5f}",
                    )
                else:
                    print(
                        f"{epoch + 1}/{self.config["NUM_EPOCHS"]}  |  {val_loss:.5f}",
                        end="\n\n\n",
                    )

        rmse = np.sqrt(
            np.mean(
                (
                    self.scaler.inverse_transform(
                        np.array(self.predictions).reshape(-1, 1)
                    )
                    - self.scaler.inverse_transform(self.y_test.numpy().reshape(-1, 1))
                )
                ** 2
            )
        )

        print(f"RMSE = {rmse:.5f}")

    def post_train_plot(self):
        test_dates = self.test_data.index[self.config["WINDOW_SIZE"] :]
        actual_vals = self.scaler.inverse_transform(self.y_test.numpy())
        predicted_vals = self.scaler.inverse_transform(
            np.array(self.predictions).reshape(-1, 1)
        )
        residuals = np.abs(predicted_vals - actual_vals)

        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        threshold = residuals_mean + self.config["THRESHOLD"] * residuals_std

        anomaly_dates = test_dates[(residuals > threshold).flatten()]
        anomaly_vals = actual_vals[(residuals > threshold).flatten()]

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
            color="coral",
            linewidth=2,
            alpha=0.75,
            label="Predicted",
        )

        if self.config["SHOW_BOUNDS"]:
            upper_bound = predicted_vals + threshold
            lower_bound = predicted_vals - threshold
            plt.plot(
                test_dates,
                upper_bound,
                color="coral",
                linestyle="--",
                alpha=0.5,
                label="↑/↓ Thresholds",
            )
            plt.plot(
                test_dates,
                lower_bound,
                color="coral",
                linestyle="--",
                alpha=0.5,
            )
            plt.fill_between(
                test_dates,
                upper_bound.flatten(),
                lower_bound.flatten(),
                color="coral",
                alpha=0.125,
            )

        if self.config["PLOT_ANOMALIES"]:
            plt.scatter(
                anomaly_dates,
                anomaly_vals,
                color="red",
                s=80,
                label="Anomalies",
            )

        tenant = self.data["Tenant"].iloc[0]
        title = f"Daily Billing Data for {tenant} (past {len(self.test_data[self.config["WINDOW_SIZE"] : ]) + 1} days)"
        plt.title(title, fontsize=16, fontweight="bold", pad=20)

        plt.xlabel("Date", fontsize=14, fontweight="bold", labelpad=15)
        plt.ylabel("ProratedQuantity", fontsize=14, fontweight="bold", labelpad=15)
        plt.xticks(
            self.test_data.index[self.config["WINDOW_SIZE"] :: 4],
            rotation=45,
            fontsize=9,
        )
        min_val = min(np.min(actual_vals), np.min(predicted_vals))
        max_val = max(np.max(actual_vals), np.max(predicted_vals))
        num_ticks = 8
        yticks = np.linspace(min_val, max_val, num_ticks)
        plt.yticks(yticks, fontsize=9)
        plt.legend(frameon=False, fontsize=10, loc="upper right")
        plt.show()

    def run_model_training(self, filepath):
        self.prepare_data(filepath)
        self.train_model()
        self.post_train_plot()

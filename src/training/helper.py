from src.data import ForecastDataset, slicing_window
from src.models import RNNForecastor
from src.training import Trainer
import torch
from torch.utils.data import DataLoader
import pandas as pd

def build_model_and_trainer(config, num_brands):
    data_cfg = config.data
    model_cfg = config.model

    config.set("model.num_brands", num_brands)
    config.set("model.input_dim", len(data_cfg["feature_cols"]))

    model = RNNForecastor(
        num_brands=num_brands,
        brand_emb_dim=model_cfg["brand_emb_dim"],
        input_dim=model_cfg["input_dim"],
        hidden_size=model_cfg["hidden_size"],
        n_layers=model_cfg["n_layers"],
        dropout_prob=model_cfg["dropout_prob"],
        output_dim=model_cfg["output_dim"],
    )

    criterion = torch.nn.HuberLoss(delta=0.8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training["learning_rate"],
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device(config.training["device"]),
        log_interval=config.training["log_interval"],
        save_dir=None,   # controlled outside
    )

    return model, trainer


def train_model_for_cutoff(
    data: pd.DataFrame,
    config,
    train_cutoff: pd.Timestamp,
):
    data_cfg = config.data
    window_cfg = config.window

    # -------------------------------
    # Windowing
    # -------------------------------
    X, y, _, cat = slicing_window(
        data,
        input_size=window_cfg["input_size"],
        horizon=window_cfg["horizon"],
        feature_cols=data_cfg["feature_cols"],
        target_col="residual",
        baseline_col="baseline",
        brand_id_col=data_cfg["brand_id_col"],
        time_col=data_cfg["time_col"],
        label_end_date=train_cutoff,
        return_dates=False,
    )

    if len(X) == 0:
        raise ValueError(f"No training windows before {train_cutoff}")

    dataset = ForecastDataset(X, y, cat)
    loader = DataLoader(
        dataset,
        batch_size=config.training["batch_size"],
        shuffle=True,
    )

    # -------------------------------
    # Model + trainer
    # -------------------------------
    num_brands = data[data_cfg["brand_id_col"]].nunique()
    model, trainer = build_model_and_trainer(config, num_brands)

    # -------------------------------
    # Train
    # -------------------------------
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        epochs=config.training["epochs"],
        save_best=False,
        verbose=False,
    )

    return model

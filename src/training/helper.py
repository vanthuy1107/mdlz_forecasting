from src.data import ForecastDataset, slicing_window
from src.models import RNNForecastor
from src.training import Trainer
from src.utils.losses import QuantileLoss, WeightedMSE
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

    # criterion = torch.nn.HuberLoss(delta=0.8)
    # criterion = QuantileLoss(q=0.7)
    criterion = torch.nn.MSELoss()
    # criterion = WeightedMSE(alpha=1.0)
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
    checkpoint_path=None,
    checkpoint_save_dir=None,
    month=None,
):
    """
    Build (or resume) a model and fine-tune it up to *train_cutoff*.

    Warm-start behaviour
    --------------------
    When *checkpoint_path* is supplied the model and optimizer are
    initialised from that file before training begins, so each monthly
    retraining is a *fine-tuning* step rather than a cold restart.
    This is much faster and usually yields better performance because
    the model retains the patterns it already learned.

    After training, if *checkpoint_save_dir* and *month* are both given,
    the updated weights are saved as ``ckpt_<YYYY-MM>.pth`` in that
    directory so the next month can pick them up.

    Args:
        data (pd.DataFrame):            Feature-engineered training data.
        config:                         Config object with data/window/training sections.
        train_cutoff (pd.Timestamp):    Only windows whose label ends before this
                                        date are used for training.
        checkpoint_path (str | Path | None):
                                        Path to a ``.pth`` file written by
                                        ``Trainer.save_walkforward_checkpoint``.
                                        Pass ``None`` to start from random weights
                                        (first month, or intentional cold-start).
        checkpoint_save_dir (str | Path | None):
                                        Directory in which to persist the newly
                                        trained checkpoint.  Nothing is saved when
                                        this is ``None``.
        month (pd.Timestamp | None):    The forecast month; used to name the
                                        saved checkpoint file.  Required when
                                        *checkpoint_save_dir* is set.

    Returns:
        nn.Module: Trained (or fine-tuned) model in eval mode.
    """
    from pathlib import Path

    data_cfg = config.data
    window_cfg = config.window
    print(f"  Features: {data_cfg['feature_cols']}")

    # -------------------------------
    # Windowing
    # -------------------------------
    X, y, brand = slicing_window(
        data,
        input_size=window_cfg["input_size"],
        horizon=window_cfg["horizon"],
        feature_cols=data_cfg["feature_cols"],
        target_col=data_cfg["target_col"],
        brand_id_col=data_cfg["brand_id_col"],
        time_col=data_cfg["time_col"],
        label_end_date=train_cutoff,
        return_dates=False,
    )

    if len(X) == 0:
        raise ValueError(f"No training windows before {train_cutoff}")

    dataset = ForecastDataset(X, y, brand)
    loader = DataLoader(
        dataset,
        batch_size=config.training["batch_size"],
        shuffle=False,
    )

    # -------------------------------
    # Model + trainer
    # Always build a fresh architecture first (guarantees correct shapes),
    # then overwrite weights from the checkpoint if one is provided.
    # -------------------------------
    num_brands = data[data_cfg["brand_id_col"]].nunique()
    model, trainer = build_model_and_trainer(config, num_brands)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            prior_month = trainer.load_walkforward_checkpoint(checkpoint_path)
            print(f"  ✅ Resumed from checkpoint: {checkpoint_path.name} (month={prior_month})")
        else:
            print(f"  ⚠️  Checkpoint not found at {checkpoint_path} — starting from scratch")

    # -------------------------------
    # Train (fine-tune if warm-started)
    # -------------------------------
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        epochs=config.training["epochs"],
        save_best=False,
        verbose=False,
    )

    # -------------------------------
    # Persist checkpoint for next month
    # -------------------------------
    if checkpoint_save_dir is not None and month is not None:
        saved_path = trainer.save_walkforward_checkpoint(month, checkpoint_save_dir)
        print(f"  💾 Checkpoint saved → {saved_path}")

    return model

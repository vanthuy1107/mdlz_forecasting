"""
MVP Training Script – uses preprocessed data only
(Refactored to use training.helper)
"""

import os
import time
import json
import pickle
import torch
import pandas as pd

from config import load_config
from src.utils import seed_everything, SEED
from src.training.helper import train_model_for_cutoff

seed_everything(SEED)


def train_model(config):
    print("=" * 80)
    print("TRAINING MODEL (PREPROCESSED DATA)")
    print("=" * 80)

    preprocess_dir = "data_preprocess"

    # --------------------------------------------------
    # 1. Load preprocessed data
    # --------------------------------------------------
    data = pd.read_csv(
        os.path.join(preprocess_dir, "data_preprocessed.csv"),
        parse_dates=[config.data["time_col"]],
    )

    with open(os.path.join(preprocess_dir, "feature_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    train_cutoff = artifacts["train_cutoff"]
    brand2id = artifacts["brand2id"]
    num_brands = artifacts["num_brands"]

    print(f"[INFO] Training cutoff: {train_cutoff}")
    print(f"[INFO] Num brands: {num_brands}")

    # --------------------------------------------------
    # 2. Train using helper
    # --------------------------------------------------
    start = time.time()

    model = train_model_for_cutoff(
        data=data,
        config=config,
        train_cutoff=train_cutoff,
    )

    training_time = time.time() - start

    # --------------------------------------------------
    # 3. Save model
    # --------------------------------------------------
    model_dir = config.output["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config._config,
            "num_brands": num_brands,
            "brand2id": brand2id,
            "train_cutoff": str(train_cutoff),
        },
        model_path,
    )

    # --------------------------------------------------
    # 4. Save metadata
    # --------------------------------------------------
    metadata = {
        "num_brands": num_brands,
        "brand2id": brand2id,
        "training_time_seconds": training_time,
        "config": config._config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Training complete")
    print(f"  - Time: {training_time / 60:.2f} minutes")
    print(f"  - Model saved to: {model_path}")

    return training_time


def main():
    config = load_config()

    output_dir = "outputs/mvp_test"
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    config.set("output.output_dir", output_dir)
    config.set("output.model_dir", model_dir)

    train_model(config)


if __name__ == "__main__":
    main()

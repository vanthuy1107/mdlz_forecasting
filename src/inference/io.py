from src.models import RNNForecastor
from pathlib import Path
import json
import pickle
import torch

def load_model_for_test(model_path: str, config):
    """Load trained model from checkpoint and scaler."""
    # Load metadata first to get the training-time model/data config
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # ------------------------------------------------------------------
    # 1) Recover num_brands and full model architecture from metadata
    # ------------------------------------------------------------------
    num_brands = None
    model_config = None
    feature_cols = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get full model_config (includes num_brands, input_dim, etc.)
        model_config = metadata.get('model_config', {})
        if 'num_brands' in model_config:
            num_brands = model_config['num_brands']

        # Also recover the exact feature column list used during training
        data_config = metadata.get("data_config", {})
        feature_cols = data_config.get("feature_cols")
    
    # Fallback to config if metadata doesn't have it
    if num_brands is None:
        model_config = config.model
        num_brands = model_config.get('num_brands')
    
        if num_brands is None:
            raise ValueError("num_brands must be found in model metadata or config")

    # If we have the training-time feature list, push it into the live config
    # so that window creation uses the exact same ordering and dimensionality.
    if feature_cols is not None:
        config.set("data.feature_cols", list(feature_cols))
    
    print(f"  - Loading model with num_brands={num_brands} (from trained model)")
    
    # ------------------------------------------------------------------
    # 2) Build model with the *exact* architecture used during training
    #    (input_dim, hidden_size, n_layers, etc. come from metadata)
    # -----------------------------------------------------------------
    model = RNNForecastor(
        num_brands=num_brands,
        brand_emb_dim=model_config['brand_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    ckpt = torch.load(model_path, map_location=device)

    cfg = ckpt.get("config", {})
    model_cfg = cfg["model"]
    assert model_cfg["input_dim"] == model.input_dim
    assert model_cfg["brand_emb_dim"] == model.brand_emb_dim
    assert model_cfg["num_brands"] == model.num_brands

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded from: {model_path}")
    print(f"  - Best validation loss: {ckpt.get('best_val_loss', 'N/A'):.4f}")
    
    # Load scaler from same directory as model (model_dir already defined above)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from: {scaler_path}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, test will be in scaled space")
    
    return model, device, scaler
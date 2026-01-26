"""Model loading utilities for prediction.

This module handles loading trained models, scalers, and metadata,
ensuring consistent category ID mappings from training time.
"""
import json
import pickle
import re
from pathlib import Path
import torch

from src.models import RNNWithCategory


def load_model_for_prediction(model_path: str, config):
    """Load trained model from checkpoint and scaler.
    
    Args:
        model_path: Path to the model checkpoint file (best_model.pth)
        config: Configuration object
    
    Returns:
        Tuple of (model, device, scaler, trained_category_filter, trained_cat2id)
        - model: Loaded PyTorch model in eval mode
        - device: PyTorch device
        - scaler: StandardScaler if available, else None
        - trained_category_filter: Category the model was trained on (if single-category)
        - trained_cat2id: Training-time category to ID mapping
    """
    # Load metadata first to get the training-time model/data config
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # ------------------------------------------------------------------
    # 1) Recover num_categories and full model architecture from metadata
    # ------------------------------------------------------------------
    num_categories = None
    trained_model_config = None
    trained_feature_cols = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get full model_config (includes num_categories, input_dim, etc.)
        trained_model_config = metadata.get('model_config', {})
        if 'num_categories' in trained_model_config:
            num_categories = trained_model_config['num_categories']

        # Also recover the exact feature column list used during training
        trained_data_config = metadata.get("data_config", {})
        trained_feature_cols = trained_data_config.get("feature_cols")
    
    # Fallback to config if metadata doesn't have it
    if num_categories is None:
        model_config = config.model
        num_categories = model_config.get('num_categories')
    
    if num_categories is None:
        raise ValueError("num_categories must be found in model metadata or config")

    # Get category_filter from training metadata to know which category(ies) model was trained on
    trained_category_filter = None
    trained_cat2id = None  # Training-time category mapping
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if 'data_config' in metadata and 'category_filter' in metadata['data_config']:
            trained_category_filter = metadata['data_config']['category_filter']
        
        # Try to extract category mapping from log_summary
        log_summary = metadata.get('log_summary', '')
        # Look for "Category mapping: {...}" in log_summary
        match = re.search(r"Category mapping: ({[^}]+})", log_summary)
        if match:
            try:
                # Parse the dictionary string from log_summary
                trained_cat2id_str = match.group(1)
                # Convert single quotes to double quotes for JSON parsing
                trained_cat2id_str = trained_cat2id_str.replace("'", '"')
                trained_cat2id = json.loads(trained_cat2id_str)
            except:
                pass

    # If we have the training-time feature list, push it into the live config
    # so that window creation uses the exact same ordering and dimensionality.
    if trained_feature_cols is not None:
        config.set("data.feature_cols", list(trained_feature_cols))
    
    print(f"  - Loading model with num_categories={num_categories} (from trained model)")
    if trained_category_filter:
        print(f"  - Model was trained on category: {trained_category_filter}")
    else:
        print(f"  - Model was trained on: all categories (num_categories={num_categories})")
    if trained_cat2id:
        print(f"  - Training-time category mapping: {trained_cat2id}")
    
    # ------------------------------------------------------------------
    # 2) Build model with the *exact* architecture used during training
    #    (input_dim, hidden_size, n_layers, etc. come from metadata)
    # ------------------------------------------------------------------
    if trained_model_config is not None:
        # Override config.model with training-time values for safety
        model_config = config.model
        for k, v in trained_model_config.items():
            model_config[k] = v
    else:
        model_config = config.model

    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim'],
        use_layer_norm=model_config.get('use_layer_norm', True),
    )
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded from: {model_path}")
    print(f"  - Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Load scaler from same directory as model (model_dir already defined above)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from: {scaler_path}")
        print(f"    Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, predictions will be in scaled space")
    
    return model, device, scaler, trained_category_filter, trained_cat2id

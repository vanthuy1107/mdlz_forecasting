"""Model definitions for the MDLZ warehouse forecasting project.

This module provides:
 - RNNWithCategory: LSTM model that conditions on product category
 - RNNForecastor: baseline RNN/LSTM model without explicit category conditioning

The implementations follow the architecture described in MODEL_LOGIC.md.

Day-of-Week Encoding:
The model currently uses cyclical encoding (sin/cos) for day-of-week features
(day_of_week_sin, day_of_week_cos) which ensures the model understands that
Sunday (6) is adjacent to Monday (0). This is the recommended approach.

Alternative: Day-of-Week Embedding
For an embedding-based approach, you would:
1. Extract day_of_week (0-6) from features before passing to LSTM
2. Create an Embedding layer: nn.Embedding(7, day_emb_dim)
3. Embed the day_of_week and concatenate with other features
4. This allows the model to learn a multi-dimensional representation of each day

Example implementation:
    # In __init__:
    self.day_embedding = nn.Embedding(7, day_emb_dim)  # 7 days
    
    # In forward:
    day_of_week = x_seq[:, :, day_of_week_idx].long()  # Extract from features
    day_emb = self.day_embedding(day_of_week)  # (B, T, day_emb_dim)
    # Remove day_of_week from x_seq and concatenate day_emb instead
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List


class RNNWithCategory(nn.Module):
    """Category-aware LSTM forecaster.

    Args:
        num_categories: Number of unique product categories.
        cat_emb_dim: Dimension of category embedding vectors.
        input_dim: Number of time-series features per timestep (without category).
        hidden_size: LSTM hidden state size.
        n_layers: Number of LSTM layers.
        output_dim: Output dimension (e.g., 1 for single-step forecast).
    """

    def __init__(
        self,
        num_categories: int,
        cat_emb_dim: int,
        input_dim: int,
        hidden_size: int,
        n_layers: int,
        output_dim: int,
        use_layer_norm: bool = True,
        dropout_prob: float = 0.0,
        category_specific_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.num_categories = num_categories
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob
        self.category_specific_params = category_specific_params or {}

        # Category embedding
        self.cat_embedding = nn.Embedding(num_categories, cat_emb_dim)

        # Initialize hidden state from category embedding
        self.h0_fc = nn.Linear(cat_emb_dim, hidden_size)

        # LSTM over concatenated [time_features, category_embedding]
        # Note: Dropout in LSTM is applied between layers (not per timestep)
        self.lstm = nn.LSTM(
            input_size=input_dim + cat_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob if n_layers > 1 else 0.0,  # Dropout only between layers
        )

        # Final prediction layer: outputs entire forecast horizon at once (direct multi-step)
        # This prevents exposure bias from recursive forecasting
        self.fc = nn.Linear(hidden_size, output_dim)

        # Optional Layer Normalization on the final hidden state to
        # reduce sensitivity to absolute scale and improve stability.
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        
        # Optional dropout before final layer (category-specific dropout handled in forward)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(
        self, 
        x_seq: Tensor, 
        x_cat: Tensor,
        apply_seasonal_mask: bool = False,
        is_active_season_idx: Optional[int] = None
    ) -> Tensor:
        """Forward pass.

        Args:
            x_seq: Input sequence of shape (B, T, D) with D == input_dim.
            x_cat: Category IDs of shape (B,) or (B, 1).
            apply_seasonal_mask: Whether to apply seasonal active-window masking.
            is_active_season_idx: Index of is_active_season feature in x_seq (if apply_seasonal_mask=True).

        Returns:
            Tensor of shape (B, output_dim) with predictions for entire forecast horizon.
            For direct multi-step forecasting, output_dim = horizon (e.g., 30 days).
        """
        # Ensure category IDs are 1D long tensors
        if x_cat.dim() > 1:
            x_cat = x_cat.squeeze(-1)
        x_cat = x_cat.long()

        batch_size, seq_len, _ = x_seq.shape

        # Category embedding: (B, cat_emb_dim)
        cat_vec = self.cat_embedding(x_cat)

        # Hidden state initialization from category embedding
        # h0_single: (B, hidden_size)
        h0_single = torch.tanh(self.h0_fc(cat_vec))
        # h0: (num_layers, B, hidden_size)
        h0 = h0_single.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # Cell state initialized to zeros
        c0 = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_size,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )

        # Expand category embedding across time dimension and concatenate
        # cat_seq: (B, T, cat_emb_dim)
        cat_seq = cat_vec.unsqueeze(1).expand(-1, seq_len, -1)
        # x: (B, T, input_dim + cat_emb_dim)
        x = torch.cat([x_seq, cat_seq], dim=-1)

        # LSTM forward
        out, _ = self.lstm(x, (h0.to(x_seq.device), c0))
        # Take last timestep output: (B, hidden_size)
        last_out = out[:, -1, :]

        # Apply LayerNorm if enabled
        if self.layer_norm is not None:
            last_out = self.layer_norm(last_out)

        # Apply category-specific dropout if configured
        # Note: This is a simplified approach. For per-category dropout, we'd need
        # to apply different dropout rates per sample, which is complex.
        # For now, we use the global dropout_prob.
        if self.dropout is not None:
            last_out = self.dropout(last_out)

        # Final prediction: outputs entire forecast horizon at once
        # Shape: (B, output_dim) where output_dim = horizon (e.g., 30)
        pred = self.fc(last_out)
        
        # Apply seasonal active-window masking if requested
        # This enforces hard-zero constraint for off-season periods
        if apply_seasonal_mask and is_active_season_idx is not None:
            # Extract is_active_season from the last timestep of input sequence
            # Shape: (B,)
            is_active = x_seq[:, -1, is_active_season_idx] > 0.5  # Binary threshold
            
            # For multi-step forecasting, we need to expand the mask to match horizon
            # For now, we apply the same mask to all steps in the horizon
            # In practice, you'd want to extract is_active_season for each future timestep
            # from the forecast window, but that requires additional feature engineering
            if pred.ndim == 2:
                is_active = is_active.unsqueeze(-1).expand_as(pred)
            
            # Apply hard-zero constraint: set predictions to 0 when not in active season
            pred = pred * is_active.float()
        
        return pred


class RNNForecastor(nn.Module):
    """Baseline RNN/LSTM forecaster without explicit category conditioning.

    Args:
        embedding_dim: Number of input features per timestep.
        hidden_size: Hidden state size of the RNN/LSTM.
        out_dim: Output dimension (e.g., 1 for single-step forecast).
        n_layers: Number of RNN/LSTM layers.
        dropout_prob: Dropout probability applied before the final layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        out_dim: int,
        n_layers: int = 1,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.n_layers = n_layers

        # Core sequence model
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x_seq: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x_seq: Input sequence of shape (B, T, D) with D == embedding_dim.
            x_cat: Optional category tensor (ignored, present for API compatibility).

        Returns:
            Tensor of shape (B, out_dim) with predictions.
        """
        # LSTM forward
        out, _ = self.rnn(x_seq)
        # Take last timestep output
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)

        return self.fc(last_out)


class HybridLSTM(nn.Module):
    """Multi-Input Hybrid LSTM with Dual-Branch Architecture.
    
    This model implements a two-branch approach:
    - Branch A (LSTM Path): Processes temporal sequences to capture short-term dynamics
    - Branch B (Dense Path): Processes static/meta features for domain knowledge injection
    
    The outputs from both branches are concatenated and passed through fusion layers
    to learn complex interactions between temporal trends and contextual information.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: Temporal Sequence (T days x D features)             │
    │         + Static Features (Brand, Day Stats, Holidays, etc.)│
    └────────────┬────────────────────────────┬───────────────────┘
                 │                            │
         Branch A: LSTM                Branch B: Dense
         (Temporal Path)               (Static/Meta Path)
                 │                            │
         [LSTM Layers]                  [MLP Layers]
         Hidden State H             Contextual Embedding C
                 │                            │
                 └────────────┬───────────────┘
                              │
                    [Concatenate: H || C]
                              │
                    [Fusion Dense Layers]
                              │
                         [Output Layer]
                              │
                    Forecast (Horizon Days)
    
    Args:
        num_categories: Number of unique product categories (for embedding).
        cat_emb_dim: Dimension of category embedding vectors.
        temporal_input_dim: Number of temporal features per timestep in the sequence.
        static_input_dim: Number of static/meta features (1D vector).
        hidden_size: LSTM hidden state size (Branch A output dimension).
        n_layers: Number of LSTM layers in Branch A.
        dense_hidden_sizes: List of hidden layer sizes for Branch B (e.g., [64, 32]).
        fusion_hidden_sizes: List of hidden layer sizes for fusion layers (e.g., [64]).
        output_dim: Output dimension (forecast horizon, e.g., 30 days).
        dropout_prob: Dropout probability for regularization.
        use_layer_norm: Whether to apply LayerNorm on LSTM output and fusion layers.
    """
    
    def __init__(
        self,
        num_categories: int,
        cat_emb_dim: int,
        temporal_input_dim: int,
        static_input_dim: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dense_hidden_sizes: List[int] = [64, 32],
        fusion_hidden_sizes: List[int] = [64],
        output_dim: int = 30,
        dropout_prob: float = 0.2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_categories = num_categories
        self.cat_emb_dim = cat_emb_dim
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.use_layer_norm = use_layer_norm
        
        # ═══════════════════════════════════════════════════════════
        # Branch A: Temporal Sequence Processor (LSTM)
        # ═══════════════════════════════════════════════════════════
        # Input: Sequential window of T days (e.g., T=30) containing historical CBM values
        # Objective: Extract latent "momentum" and autocorrelation patterns from recent past
        
        # Category embedding for temporal branch (optional, can be used to initialize h0)
        self.cat_embedding = nn.Embedding(num_categories, cat_emb_dim)
        
        # Initialize LSTM hidden state from category embedding
        self.h0_fc = nn.Linear(cat_emb_dim, hidden_size)
        
        # LSTM: Processes temporal sequence
        # Input: (batch, time_steps, temporal_input_dim + cat_emb_dim)
        # Output: (batch, time_steps, hidden_size)
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim + cat_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob if n_layers > 1 else 0.0,
        )
        
        # Optional LayerNorm on LSTM output
        self.lstm_layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        
        # ═══════════════════════════════════════════════════════════
        # Branch B: Contextual Knowledge Base (Dense Stream)
        # ═══════════════════════════════════════════════════════════
        # Input: 1D Vector of hand-crafted features (static features)
        # Objective: Hard-code "operational constraints" and "global trends"
        #           (e.g., Sundays usually have zero volume, holiday proximity effects)
        
        # Build Dense (MLP) layers for static features
        dense_layers = []
        prev_size = static_input_dim
        for hidden_size_dense in dense_hidden_sizes:
            dense_layers.append(nn.Linear(prev_size, hidden_size_dense))
            if use_layer_norm:
                dense_layers.append(nn.LayerNorm(hidden_size_dense))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size_dense
        
        self.dense_branch = nn.Sequential(*dense_layers)
        dense_output_dim = dense_hidden_sizes[-1] if dense_hidden_sizes else static_input_dim
        
        # ═══════════════════════════════════════════════════════════
        # Feature Fusion & Interaction Layer
        # ═══════════════════════════════════════════════════════════
        # The Merge: Concatenate LSTM's final hidden state with Dense branch output
        # Integration Dense Layers: "Bottleneck" layer that acts as inference engine
        #   to decide if "Temporal Trend" should be overridden by "Operational Reality"
        
        fusion_input_dim = hidden_size + dense_output_dim
        
        # Build Fusion layers
        fusion_layers = []
        prev_size = fusion_input_dim
        for fusion_hidden_size in fusion_hidden_sizes:
            fusion_layers.append(nn.Linear(prev_size, fusion_hidden_size))
            if use_layer_norm:
                fusion_layers.append(nn.LayerNorm(fusion_hidden_size))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout_prob))
            prev_size = fusion_hidden_size
        
        self.fusion_layers = nn.Sequential(*fusion_layers)
        fusion_output_dim = fusion_hidden_sizes[-1] if fusion_hidden_sizes else fusion_input_dim
        
        # ═══════════════════════════════════════════════════════════
        # Output Layer
        # ═══════════════════════════════════════════════════════════
        # Final prediction layer: outputs entire forecast horizon at once
        # This prevents exposure bias from recursive forecasting
        
        self.output_layer = nn.Linear(fusion_output_dim, output_dim)
    
    def forward(
        self,
        x_temporal: Tensor,  # (batch, time_steps, temporal_input_dim)
        x_static: Tensor,    # (batch, static_input_dim)
        x_cat: Tensor,       # (batch,) or (batch, 1) - category IDs
    ) -> Tensor:
        """Forward pass through the hybrid model.
        
        Args:
            x_temporal: Temporal sequence input (batch, T, D_temporal).
            x_static: Static/meta features input (batch, D_static).
            x_cat: Category IDs (batch,) or (batch, 1).
        
        Returns:
            Tensor of shape (batch, output_dim) with predictions for forecast horizon.
        """
        # Ensure category IDs are 1D long tensors
        if x_cat.dim() > 1:
            x_cat = x_cat.squeeze(-1)
        x_cat = x_cat.long()
        
        batch_size, seq_len, _ = x_temporal.shape
        
        # ═══════════════════════════════════════════════════════════
        # Branch A: LSTM Path (Temporal Processing)
        # ═══════════════════════════════════════════════════════════
        
        # Category embedding for hidden state initialization
        cat_vec = self.cat_embedding(x_cat)  # (batch, cat_emb_dim)
        
        # Initialize LSTM hidden state from category embedding
        h0_single = torch.tanh(self.h0_fc(cat_vec))  # (batch, hidden_size)
        h0 = h0_single.unsqueeze(0).repeat(self.n_layers, 1, 1)  # (n_layers, batch, hidden_size)
        
        # Cell state initialized to zeros
        c0 = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_size,
            device=x_temporal.device,
            dtype=x_temporal.dtype,
        )
        
        # Expand category embedding across time and concatenate with temporal features
        cat_seq = cat_vec.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, T, cat_emb_dim)
        x_lstm = torch.cat([x_temporal, cat_seq], dim=-1)  # (batch, T, temporal_input_dim + cat_emb_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_lstm, (h0.to(x_temporal.device), c0))  # (batch, T, hidden_size)
        
        # Take last timestep output
        lstm_final = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply LayerNorm if enabled
        if self.lstm_layer_norm is not None:
            lstm_final = self.lstm_layer_norm(lstm_final)
        
        # ═══════════════════════════════════════════════════════════
        # Branch B: Dense Path (Static/Meta Processing)
        # ═══════════════════════════════════════════════════════════
        
        dense_out = self.dense_branch(x_static)  # (batch, dense_output_dim)
        
        # ═══════════════════════════════════════════════════════════
        # Feature Fusion
        # ═══════════════════════════════════════════════════════════
        
        # Concatenate LSTM output and Dense output
        fused = torch.cat([lstm_final, dense_out], dim=-1)  # (batch, hidden_size + dense_output_dim)
        
        # Pass through fusion layers (interaction learning)
        fused = self.fusion_layers(fused)  # (batch, fusion_output_dim)
        
        # ═══════════════════════════════════════════════════════════
        # Output Prediction
        # ═══════════════════════════════════════════════════════════
        
        pred = self.output_layer(fused)  # (batch, output_dim)
        
        return pred


__all__ = ["RNNWithCategory", "RNNForecastor", "HybridLSTM"]


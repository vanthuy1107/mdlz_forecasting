"""Model definitions for the MDLZ warehouse forecasting project.

This module provides:
 - RNNWithCategory: LSTM model that conditions on product category
 - RNNForecastor: baseline RNN/LSTM model without explicit category conditioning

The implementations follow the architecture described in MODEL_LOGIC.md.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


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
    ) -> None:
        super().__init__()

        self.num_categories = num_categories
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Category embedding
        self.cat_embedding = nn.Embedding(num_categories, cat_emb_dim)

        # Initialize hidden state from category embedding
        self.h0_fc = nn.Linear(cat_emb_dim, hidden_size)

        # LSTM over concatenated [time_features, category_embedding]
        self.lstm = nn.LSTM(
            input_size=input_dim + cat_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        # Final prediction layer (sequence-to-one)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x_seq: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_seq: Input sequence of shape (B, T, D) with D == input_dim.
            x_cat: Category IDs of shape (B,) or (B, 1).

        Returns:
            Tensor of shape (B, output_dim) with predictions.
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

        # Final prediction
        pred = self.fc(last_out)
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


__all__ = ["RNNWithCategory", "RNNForecastor"]


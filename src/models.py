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


class RNNForecastor(nn.Module):
    def __init__(
        self, num_categories, cat_emb_dim, 
        input_dim, hidden_size,
        n_layers, output_dim=30
    ):
        super().__init__()
        self.n_layers = n_layers

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim + cat_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

        self.h0_fc = nn.Linear(cat_emb_dim, hidden_size)
        #self.fc = nn.Linear(hidden_size, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x_seq, x_cat):
        # x_seq: (B, T, F)
        # x_cat: (B,)
               
        B, T, _ = x_seq.shape

        x_cat = x_cat.long()  # Ensure cat is long for embedding
        cat_vec = self.cat_emb(x_cat)              # (B, E)

        # init hidden
        h0 = self.h0_fc(cat_vec)                   # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        c0 = torch.zeros_like(h0)

        # expand category over time
        cat_seq = cat_vec.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x_seq, cat_seq], dim=-1)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        y = self.fc(out)
        return y


__all__ = ["RNNForecastor"]


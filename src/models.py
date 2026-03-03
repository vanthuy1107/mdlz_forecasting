"""
RNN-based forecasting model with brand embeddings.
"""

import torch
import torch.nn as nn


class RNNForecastor(nn.Module):
    def __init__(
        self, num_brands, brand_emb_dim=4, 
        input_dim=28, hidden_size=64,
        n_layers=2, 
        dropout_prob=0.2,
        output_dim=7
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_brands = num_brands
        self.brand_emb_dim = brand_emb_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers

        self.brand_emb = nn.Embedding(num_brands, brand_emb_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim + brand_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=self.dropout_prob if n_layers > 1 else 0
        )

        self.h0_fc = nn.Linear(brand_emb_dim, hidden_size)
        self.c0_fc = nn.Linear(brand_emb_dim, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x_seq, x_brand):
        # x_seq: (B, T, F)
        # x_brand: (B,)

        B, T, _ = x_seq.shape

        x_brand = x_brand.long()
        brand_vec = self.brand_emb(x_brand)   # (B, E)

        # -------------------------------------------------
        # 1️⃣ Expand brand embedding across time
        # -------------------------------------------------
        brand_expand = brand_vec.unsqueeze(1).repeat(1, T, 1)  # (B, T, E)

        # -------------------------------------------------
        # 2️⃣ Concatenate with sequence features
        # -------------------------------------------------
        x_seq = torch.cat([x_seq, brand_expand], dim=-1)  # (B, T, F+E)

        # -------------------------------------------------
        # 3️⃣ Initialize hidden state/long-term using brand
        # -------------------------------------------------
        h0 = self.h0_fc(brand_vec)               # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        c0 = torch.zeros_like(h0)
        # c0 = self.c0_fc(brand_vec)
        # c0 = c0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # -------------------------------------------------
        # 4️⃣ LSTM
        # -------------------------------------------------
        out, _ = self.lstm(x_seq, (h0, c0))
        out = out[:, -1, :]

        # -------------------------------------------------
        # 5️⃣ Output
        # -------------------------------------------------
        y = self.fc(out)
        return y


__all__ = ["RNNForecastor"]


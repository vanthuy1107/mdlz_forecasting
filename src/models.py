"""
RNN-based forecasting model with brand embeddings.
"""

import torch
import torch.nn as nn


class RNNForecastor(nn.Module):
    def __init__(
        self, num_brands, brand_emb_dim, 
        input_dim, hidden_size,
        n_layers, output_dim=28
    ):
        super().__init__()
        self.n_layers = n_layers

        self.brand_emb = nn.Embedding(num_brands, brand_emb_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim + brand_emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

        self.h0_fc = nn.Linear(brand_emb_dim, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x_seq, x_brand):
        # x_seq: (B, T, F)
        # x_brand: (B,)
               
        B, T, _ = x_seq.shape

        x_brand = x_brand.long()  # Ensure brand is long for embedding
        brand_vec = self.brand_emb(x_brand)              # (B, E)

        # init hidden
        h0 = self.h0_fc(brand_vec)                   # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        c0 = torch.zeros_like(h0)

        # expand brand over time
        brand_seq = brand_vec.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x_seq, brand_seq], dim=-1)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        y = self.fc(out)
        return y


__all__ = ["RNNForecastor"]


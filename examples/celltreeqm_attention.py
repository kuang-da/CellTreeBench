import torch
import torch.nn as nn
import logging
from feature_gates_minimal import FeatureGates


class BaseCellEncoder(nn.Module):
    """Abstract base class for per-cell encoders."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class LinearCellEncoder(BaseCellEncoder):
    """Matches the original linear projection behaviour."""

    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, proj_dim)
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SiteTokenEncoder(BaseCellEncoder):
    """Site-aware encoder that shares parameters across sites before aggregation."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        site_alphabet_size: int = 22,
        site_embed_dim: int = 64,
        site_encoder_heads: int = 4,
        site_encoder_layers: int = 2,
        site_dropout: float = 0.1,
        site_chunk_size: int | None = 32,
    ):
        super().__init__()
        if input_dim % site_alphabet_size != 0:
            raise ValueError(
                "input_dim must be divisible by site_alphabet_size when using the site encoder"
            )
        self.input_dim = input_dim
        self.site_alphabet_size = site_alphabet_size
        self.num_sites = input_dim // site_alphabet_size

        self.embedding = nn.Linear(site_alphabet_size, site_embed_dim)
        self.embedding_norm = nn.LayerNorm(site_embed_dim)
        self.embedding_dropout = nn.Dropout(site_dropout)

        if site_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=site_embed_dim,
                nhead=site_encoder_heads,
                dim_feedforward=site_embed_dim * 2,
                dropout=site_dropout,
                norm_first=True,
            )
            self.site_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=site_encoder_layers
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, site_embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.site_encoder = None
            self.cls_token = None

        self.output_linear = nn.Linear(site_embed_dim, proj_dim)
        self.site_chunk_size = None if site_chunk_size in (None, 0) else max(1, int(site_chunk_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError("Unexpected feature dimension for site encoder")

        batch = x.shape[0]
        x = x.view(batch, self.num_sites, self.site_alphabet_size)

        if self.site_chunk_size is None or batch <= self.site_chunk_size:
            x = self.embedding(x)

            if self.site_encoder is not None:
                cls_tokens = self.cls_token.expand(batch, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            x = self.embedding_norm(x)
            x = self.embedding_dropout(x)

            if self.site_encoder is not None:
                x = x.transpose(0, 1)
                x = self.site_encoder(x)
                x = x.transpose(0, 1)
                pooled = x[:, 0, :]
            else:
                pooled = x.mean(dim=1)

            return self.output_linear(pooled)

        pooled_chunks = []
        for chunk in torch.split(x, self.site_chunk_size, dim=0):
            chunk = self.embedding(chunk)

            if self.site_encoder is not None:
                cls_tokens = self.cls_token.expand(chunk.size(0), -1, -1)
                chunk = torch.cat((cls_tokens, chunk), dim=1)

            chunk = self.embedding_norm(chunk)
            chunk = self.embedding_dropout(chunk)

            if self.site_encoder is not None:
                chunk = chunk.transpose(0, 1)
                chunk = self.site_encoder(chunk)
                chunk = chunk.transpose(0, 1)
                pooled = chunk[:, 0, :]
            else:
                pooled = chunk.mean(dim=1)

            pooled_chunks.append(pooled)

        pooled = torch.cat(pooled_chunks, dim=0)
        return self.output_linear(pooled)


class _MAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(q, k, k)
        x = self.ln1(q + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)


class _SAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.mab = _MAB(dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class _PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, dropout: float):
        super().__init__()
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive")
        self.seeds = nn.Parameter(torch.zeros(1, num_seeds, dim))
        nn.init.xavier_normal_(self.seeds)
        self.mab = _MAB(dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seeds = self.seeds.expand(batch, -1, -1)
        return self.mab(seeds, x)


class SiteSetPoolEncoder(BaseCellEncoder):
    """Set Transformer style encoder with shared site parameters."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        site_alphabet_size: int = 22,
        site_embed_dim: int = 64,
        site_encoder_heads: int = 4,
        site_encoder_layers: int = 2,
        site_dropout: float = 0.1,
        site_pma_seeds: int = 4,
        site_chunk_size: int | None = 32,
    ):
        super().__init__()
        if input_dim % site_alphabet_size != 0:
            raise ValueError(
                "input_dim must be divisible by site_alphabet_size when using the site encoder"
            )
        self.input_dim = input_dim
        self.site_alphabet_size = site_alphabet_size
        self.num_sites = input_dim // site_alphabet_size

        self.embedding = nn.Linear(site_alphabet_size, site_embed_dim)
        self.embedding_norm = nn.LayerNorm(site_embed_dim)
        self.embedding_dropout = nn.Dropout(site_dropout)
        self.site_chunk_size = None if site_chunk_size in (None, 0) else max(1, int(site_chunk_size))

        if site_encoder_layers > 0:
            self.sab_layers = nn.ModuleList(
                [_SAB(site_embed_dim, site_encoder_heads, site_dropout) for _ in range(site_encoder_layers)]
            )
        else:
            self.sab_layers = nn.ModuleList()

        self.pma = _PMA(site_embed_dim, site_encoder_heads, site_pma_seeds, site_dropout)
        self.output_linear = nn.Linear(site_embed_dim * site_pma_seeds, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError("Unexpected feature dimension for site encoder")

        batch = x.shape[0]
        x = x.view(batch, self.num_sites, self.site_alphabet_size)

        if self.site_chunk_size is None or batch <= self.site_chunk_size:
            x = self.embedding(x)
            x = self.embedding_norm(x)
            x = self.embedding_dropout(x)

            if len(self.sab_layers) > 0:
                for layer in self.sab_layers:
                    x = layer(x)

            pooled = self.pma(x)
            pooled = pooled.reshape(batch, -1)
            return self.output_linear(pooled)

        pooled_chunks = []
        for chunk in torch.split(x, self.site_chunk_size, dim=0):
            chunk = self.embedding(chunk)
            chunk = self.embedding_norm(chunk)
            chunk = self.embedding_dropout(chunk)

            if len(self.sab_layers) > 0:
                encoded = chunk
                for layer in self.sab_layers:
                    encoded = layer(encoded)
            else:
                encoded = chunk

            pooled = self.pma(encoded)
            pooled_chunks.append(pooled.reshape(encoded.size(0), -1))

        pooled = torch.cat(pooled_chunks, dim=0)
        return self.output_linear(pooled)


class CellTreeQMAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        output_dim,
        dropout_data=0.2,
        dropout_metric=0.2,
        norm_method=None,
        proj_dim=1024,
        gate_mode="none",
        gate_embed_dim=32,
        gate_hidden_dim=64,
        tau=1.0,
        device="cpu",
        init_ones=False,
        cell_encoder_type="linear",
        site_alphabet_size=22,
        site_embed_dim=64,
        site_encoder_heads=4,
        site_encoder_layers=2,
        site_dropout=0.1,
        site_pma_seeds=4,
        site_chunk_size: int | None = 1,
    ):
        """
        Encoder with optional feature gating for CellTree Quartet Matching.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension of the transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            output_dim (int): Output feature dimension.
            dropout_data (float): Dropout for data projection.
            dropout_metric (float): Dropout for output.
            norm_method (str): Normalization method ('batch_norm', 'layer_norm', or None).
            proj_dim (int): Projection dimension for transformer input.
            gate_mode (str): Feature gating mode ('none', 'sigmoid', 'softmax', 'gumbel').
            gate_embed_dim (int): Embedding dimension for feature gating.
            gate_hidden_dim (int): Hidden dimension for feature gating.
            tau (float): Temperature parameter for Gumbel softmax.
            device (str): Device for computation.
            init_ones (bool): Whether to initialize gates to favor "on" state.
            cell_encoder_type (str): Type of cell encoder to use ('linear', 'site_token', 'site_setpool').
            site_alphabet_size (int): Size of the site alphabet.
            site_embed_dim (int): Embedding dimension for the site encoder.
            site_encoder_heads (int): Number of attention heads for the site encoder.
            site_encoder_layers (int): Number of layers for the site encoder.
            site_dropout (float): Dropout for the site encoder.
            site_pma_seeds (int): Number of seed vectors for set pooling.
            site_chunk_size (int | None): Optional mini-batch size for site encoders (chunk along cell axis).
        """
        super().__init__()
        self.norm_method = norm_method

        logging.info(f"Input dim: {input_dim}")
        logging.info(f"Projection dim: {proj_dim}")
        logging.info(f"Gating mode: {gate_mode}")
        logging.info(f"Cell encoder type: {cell_encoder_type}")

        # Feature gates
        if gate_mode == "none":
            self.feature_gate = None
        elif gate_mode == "linear":
            self.feature_gate = None
            self.G = nn.Parameter(torch.eye(input_dim))
        else:
            self.feature_gate = FeatureGates(
                input_dim,
                embed_dim=gate_embed_dim,
                hidden_dim=gate_hidden_dim,
                tau=tau,
                mode=gate_mode,
                device=device,
                init_ones=init_ones,
            )

        # Per-cell encoder
        encoder_type = cell_encoder_type.lower()
        if encoder_type == "linear":
            logging.info(f"Using LinearCellEncoder")
            self.cell_encoder = LinearCellEncoder(input_dim, proj_dim)
        
        elif encoder_type == "site_token":
            logging.info("Using SiteTokenEncoder (CLS pooling over site tokens)")
            self.cell_encoder = SiteTokenEncoder(
                input_dim=input_dim,
                proj_dim=proj_dim,
                site_alphabet_size=site_alphabet_size,
                site_embed_dim=site_embed_dim,
                site_encoder_heads=site_encoder_heads,
                site_encoder_layers=site_encoder_layers,
                site_dropout=site_dropout,
                site_chunk_size=site_chunk_size,
            )
        elif encoder_type == "site_setpool":
            logging.info(f"Using SetTransformerCellEncoder")
            self.cell_encoder = SiteSetPoolEncoder(
                input_dim=input_dim,
                proj_dim=proj_dim,
                site_alphabet_size=site_alphabet_size,
                site_embed_dim=site_embed_dim,
                site_encoder_heads=site_encoder_heads,
                site_encoder_layers=site_encoder_layers,
                site_dropout=site_dropout,
                site_pma_seeds=site_pma_seeds,
                site_chunk_size=site_chunk_size,
            )
        else:
            raise ValueError(f"Unknown cell_encoder_type '{cell_encoder_type}'")

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_data,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Linear(proj_dim, output_dim)

        # Normalization
        if norm_method == "batch_norm":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm_method == "layer_norm":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None

        # Dropout
        self.dropout1 = nn.Dropout(dropout_data)
        self.dropout2 = nn.Dropout(dropout_metric)

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) or (N, C).

        Returns:
            torch.Tensor: Output tensor.
        """
        if len(x.shape) == 3:
            batched = True
            B, N, C = x.shape
            x = x.view(B * N, C)
        else:
            batched = False

        # Apply feature gates
        if self.feature_gate is not None:
            x = self.feature_gate(x)
        elif hasattr(self, 'G'):
            x = x @ self.G

        # Encode per-cell features
        x = self.cell_encoder(x)
        x = self.dropout1(x)

        # Transformer encoder
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        # Output layer
        x = self.output_layer(x)

        # Apply normalization
        if self.norm is not None:
            x = self.norm(x)

        # Apply dropout
        x = self.dropout2(x)
        
        # x = torch.nn.functional.normalize(x, p=2, dim=-1)

        if batched:
            x = x.view(B, N, -1)

        return x

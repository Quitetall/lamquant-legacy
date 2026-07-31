"""Learned entropy models for improved rANS coding.

Path 2: SNN-driven learned entropy
    The SNN already runs for adaptive FSQ level selection. A tiny linear
    head on the SNN hidden state predicts per-channel token probabilities,
    giving rANS better CDF tables than static frequency counts.

    Cost: ~640 ternary params = ~160 bytes firmware
    Gain: ~10-20% bitrate reduction (context-adaptive CDF)

Path 4: Conditional entropy model (base station only)
    A small autoregressive transformer on the base station predicts
    P(token_t | token_{t-1}, ..., token_0). The rANS decoder uses
    these conditional probabilities.

    Cost: ~200K params, runs on base station CPU (not MCU)
    Gain: ~30-40% bitrate reduction (conditional < marginal entropy)

Usage:
    # SNN-driven (runs on MCU alongside SNN)
    entropy_head = SNNEntropyHead(snn_dim=40, n_channels=32, n_levels=5)
    cdfs = entropy_head(snn_hidden_state)  # [32, 5] per-channel CDF

    # Conditional (runs on base station)
    cond_model = ConditionalEntropyModel(n_channels=32, n_levels=5)
    cdfs = cond_model.predict_next(token_history)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SNNEntropyHead(nn.Module):
    """SNN-driven per-channel token probability predictor.

    Attaches to the MambaSNN's hidden state and predicts per-channel
    CDF tables for rANS encoding. The SNN already runs for FSQ level
    selection — this head is almost free.

    Architecture: Linear(snn_dim, n_channels * n_levels)
    Firmware cost: snn_dim * n_channels * n_levels * 0.25 bytes (ternary)
    For snn_dim=40, n_channels=32, n_levels=5: 6400 params = 1600 bytes

    The predicted CDF replaces the static frequency table in rANS,
    making the entropy coding context-adaptive.
    """

    def __init__(self, snn_dim=40, n_channels=32, n_levels=5):
        super().__init__()
        self.n_channels = n_channels
        self.n_levels = n_levels
        self.head = nn.Linear(snn_dim, n_channels * n_levels)
        # Initialize to uniform (log-uniform = zeros)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, snn_hidden):
        """Predict per-channel token probabilities from SNN state.

        Args:
            snn_hidden: [B, snn_dim] or [B, T, snn_dim] SNN hidden state
                       (pooled or per-timestep)
        Returns:
            probs: [B, n_channels, n_levels] probability distribution per channel
            cdfs:  [B, n_channels, n_levels] cumulative distribution (for rANS)
        """
        if snn_hidden.ndim == 3:
            # Pool over time
            snn_hidden = snn_hidden.mean(dim=1)

        logits = self.head(snn_hidden)  # [B, n_channels * n_levels]
        logits = logits.reshape(-1, self.n_channels, self.n_levels)

        probs = F.softmax(logits, dim=-1)  # [B, C, L]
        cdfs = probs.cumsum(dim=-1)        # [B, C, L]

        return probs, cdfs

    def to_frequency_table(self, probs, total_freq=4096):
        """Convert probabilities to integer frequency table for rANS.

        Args:
            probs: [n_channels, n_levels] probabilities
            total_freq: rANS frequency table precision (default 4096)
        Returns:
            freq: [n_channels, n_levels] integer frequencies summing to total_freq
        """
        if probs.ndim == 3:
            probs = probs[0]  # take first batch element

        freq = (probs * total_freq).round().long()
        # Ensure each channel sums to exactly total_freq
        for c in range(self.n_channels):
            diff = total_freq - freq[c].sum()
            freq[c, freq[c].argmax()] += diff
            # Ensure no zeros (rANS requires freq >= 1)
            freq[c] = freq[c].clamp(min=1)
            diff = total_freq - freq[c].sum()
            freq[c, freq[c].argmax()] += diff

        return freq.cpu().numpy().astype(np.int32)


class ConditionalEntropyModel(nn.Module):
    """Autoregressive conditional entropy model for base station decoding.

    A small transformer predicts P(token_t | token_{t-1}, ..., token_0)
    for each of 32 latent channels. The rANS decoder uses these conditional
    probabilities instead of static marginal tables.

    Runs on the base station (CPU is fine for 79 timesteps × 32 channels).
    NOT deployed to MCU — the encoder still uses static rANS tables.

    Architecture:
        Token embedding (n_levels → dim) per channel
        4 transformer layers (dim=64, 4 heads, causal mask)
        Linear readout → per-channel next-token logits

    For [32, 79] latent: 32 channels × 79 timesteps = 2,528 predictions.
    At dim=64, 4 layers: ~200K params. Runs in <1ms on CPU.
    """

    def __init__(self, n_channels=32, n_levels=5, dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_levels = n_levels
        self.dim = dim

        # Per-channel token embedding
        self.tok_embed = nn.Embedding(n_levels, dim)
        self.ch_embed = nn.Embedding(n_channels, dim)
        self.pos_embed = nn.Embedding(128, dim)  # max 128 timesteps

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-channel readout
        self.readout = nn.Linear(dim, n_levels)

    def forward(self, tokens):
        """Predict next-token probabilities for all positions.

        Args:
            tokens: [B, C, T] integer token indices (0 to n_levels-1)
        Returns:
            logits: [B, C, T, n_levels] next-token logits
        """
        B, C, T = tokens.shape

        # Embed tokens
        tok_emb = self.tok_embed(tokens)  # [B, C, T, dim]
        ch_idx = torch.arange(C, device=tokens.device)
        ch_emb = self.ch_embed(ch_idx).unsqueeze(0).unsqueeze(2)  # [1, C, 1, dim]
        pos_idx = torch.arange(T, device=tokens.device)
        pos_emb = self.pos_embed(pos_idx).unsqueeze(0).unsqueeze(0)  # [1, 1, T, dim]

        x = tok_emb + ch_emb + pos_emb  # [B, C, T, dim]

        # Flatten channels into batch for transformer
        x = x.reshape(B * C, T, self.dim)  # [B*C, T, dim]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)

        # Run transformer
        x = self.transformer(x, mask=mask, is_causal=True)  # [B*C, T, dim]

        # Readout
        logits = self.readout(x)  # [B*C, T, n_levels]
        logits = logits.reshape(B, C, T, self.n_levels)

        return logits

    def predict_probabilities(self, tokens):
        """Get conditional probabilities for rANS decoding.

        Args:
            tokens: [C, T] integer token indices
        Returns:
            probs: [C, T, n_levels] conditional probabilities
        """
        with torch.no_grad():
            logits = self.forward(tokens.unsqueeze(0))  # [1, C, T, L]
            probs = F.softmax(logits[0], dim=-1)        # [C, T, L]
        return probs

    def compute_conditional_entropy(self, tokens):
        """Compute conditional entropy of the token sequence.

        Useful for measuring the theoretical compression limit with
        this model vs static marginal entropy.

        Args:
            tokens: [C, T] integer token indices
        Returns:
            entropy_bits: total conditional entropy in bits
            marginal_bits: total marginal entropy in bits
        """
        with torch.no_grad():
            logits = self.forward(tokens.unsqueeze(0))  # [1, C, T, L]
            probs = F.softmax(logits[0], dim=-1)        # [C, T, L]

            C, T, L = probs.shape

            # Conditional entropy: H = -sum(P(x) * log2(P(x)))
            # Use the predicted probability of the actual token
            actual = tokens.long()  # [C, T]
            cond_bits = 0
            for c in range(C):
                for t in range(1, T):  # skip first (no context)
                    p = probs[c, t - 1, actual[c, t]].item()  # predicted prob of actual token
                    if p > 0:
                        cond_bits -= np.log2(p)

            # Marginal entropy
            marginal_bits = 0
            for c in range(C):
                counts = torch.bincount(actual[c], minlength=L).float()
                p_marginal = counts / counts.sum()
                for l in range(L):
                    if p_marginal[l] > 0:
                        marginal_bits -= T * p_marginal[l].item() * np.log2(p_marginal[l].item())

        return cond_bits, marginal_bits

    def training_loss(self, tokens):
        """Cross-entropy loss for training the entropy model.

        The model learns to predict token_t from token_{t-1}, ..., token_0.
        Trained on encoded latent tokens from the encoder.

        Args:
            tokens: [B, C, T] integer token indices
        Returns:
            loss: cross-entropy loss (scalar)
        """
        logits = self.forward(tokens)  # [B, C, T, L]
        B, C, T, L = logits.shape

        # Shift: predict token_t from position t-1
        pred = logits[:, :, :-1, :]  # [B, C, T-1, L]
        target = tokens[:, :, 1:]    # [B, C, T-1]

        return F.cross_entropy(pred.reshape(-1, L), target.reshape(-1).long())

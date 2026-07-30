"""Post-encoding token compression: RLE + context-adaptive entropy.

These are codec-layer optimizations between FSQ and rANS that don't touch
the encoder or decoder neural networks. Zero quality loss — they just
squeeze more bits out of the existing token stream.

Path 1: Token RLE
    During quiescence, the latent is nearly constant. Instead of transmitting
    79 timesteps where 70 are identical, run-length encode the token sequence.

    Current:  [tok, tok, tok, tok, tok2, tok, ...]  79 tokens
    RLE:      [(tok, 4), (tok2, 1), (tok, 2), ...]  ~20 entries

    Quiescent: 79 → ~15-25 unique tokens (60-80% reduction)
    Active:    79 → ~50-60 entries (modest reduction)

    The decoder upsamples back to 79 before feeding ConvNeXt/iSTFT.

Path 2: Context-adaptive entropy coding
    SNN hidden states encode temporal context. A tiny linear head predicts
    per-channel token probabilities, giving rANS better CDF tables than
    static frequency counts.

    SNN state [40] → Linear(40, 32×L) → per-channel CDF
    Cost: 40 × 32 × L ternary params = ~640 bytes firmware
    Gain: ~10-20% bitrate reduction

Usage:
    from lamquant_codec.token_compression import token_rle_encode, token_rle_decode
    from lamquant_codec.token_compression import estimate_rle_savings

    # Measure potential savings
    savings = estimate_rle_savings(latent_tokens)

    # Compress
    rle_payload = token_rle_encode(tokens_2d)  # [32, 79] int -> bytes
    tokens_back = token_rle_decode(rle_payload, n_channels=32, T=79)
"""

import struct
import numpy as np


def token_rle_encode(tokens, n_channels=32, T=79):
    """Run-length encode quantized token matrix [channels, T].

    For each channel, consecutive identical tokens are collapsed to
    (value, count) pairs. The output is a compact byte stream.

    Format per channel:
      [n_runs: uint8] [value_0: int8] [count_0: uint8] [value_1: int8] [count_1: uint8] ...

    Args:
        tokens: [C, T] numpy array of quantized token indices
    Returns:
        bytes: RLE-compressed payload
    """
    tokens = np.asarray(tokens)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    C, T_actual = tokens.shape

    output = bytearray()
    total_runs = 0

    for ch in range(C):
        row = tokens[ch]
        runs = []
        cur_val = int(row[0])
        cur_count = 1

        for t in range(1, T_actual):
            if int(row[t]) == cur_val and cur_count < 255:
                cur_count += 1
            else:
                runs.append((cur_val, cur_count))
                cur_val = int(row[t])
                cur_count = 1
        runs.append((cur_val, cur_count))

        # Pack: n_runs + (value, count) pairs
        output.append(min(len(runs), 255))
        for val, count in runs[:255]:
            output.append(val & 0xFF)  # int8 as uint8
            output.append(count & 0xFF)
        total_runs += len(runs)

    return bytes(output), total_runs


def token_rle_decode(data, n_channels=32, T=79):
    """Decode RLE-compressed tokens back to [C, T] matrix.

    Args:
        data: bytes from token_rle_encode
        n_channels: number of channels
        T: target temporal length
    Returns:
        [C, T] numpy array of token indices
    """
    tokens = np.zeros((n_channels, T), dtype=np.int32)
    pos = 0

    for ch in range(n_channels):
        n_runs = data[pos]
        pos += 1
        t = 0
        for _ in range(n_runs):
            val = data[pos]
            if val > 127:
                val -= 256  # unsigned to signed
            count = data[pos + 1]
            pos += 2
            tokens[ch, t:t + count] = val
            t += count

    return tokens


def estimate_rle_savings(latent, fsq_levels=5):
    """Estimate RLE compression savings on a latent tensor.

    Args:
        latent: [B, C, T] or [C, T] float tensor in [-1, 1]
        fsq_levels: FSQ quantization level
    Returns:
        dict with savings metrics
    """
    if hasattr(latent, 'numpy'):
        latent = latent.detach().cpu().numpy()
    if latent.ndim == 3:
        latent = latent[0]

    C, T = latent.shape
    L = fsq_levels

    # Quantize to FSQ indices
    step = 2.0 / L
    tokens = np.clip(((latent + 1.0) / step).astype(np.int32), 0, L - 1)

    # Count runs
    total_tokens = C * T
    total_runs = 0
    for ch in range(C):
        runs = 1
        for t in range(1, T):
            if tokens[ch, t] != tokens[ch, t - 1]:
                runs += 1
        total_runs += runs

    # Uncompressed: C * T * ceil(log2(L)) bits
    bits_per_token = np.ceil(np.log2(L))
    uncompressed_bits = total_tokens * bits_per_token

    # RLE: total_runs * (value_bits + count_bits)
    # value: ceil(log2(L)) bits, count: 8 bits
    rle_bits = total_runs * (bits_per_token + 8)

    # Also compute the byte-level sizes
    uncompressed_bytes = int(np.ceil(uncompressed_bits / 8))
    rle_bytes = C + total_runs * 2  # header (n_runs per ch) + 2 bytes per run

    savings_pct = (1 - rle_bytes / max(uncompressed_bytes, 1)) * 100

    return {
        'total_tokens': total_tokens,
        'total_runs': total_runs,
        'compression_ratio': total_tokens / max(total_runs, 1),
        'uncompressed_bytes': uncompressed_bytes,
        'rle_bytes': rle_bytes,
        'savings_pct': savings_pct,
        'avg_run_length': total_tokens / max(total_runs, 1),
    }


def hexagonal_q2d2_quantize(pair_a, pair_b, n_points=19):
    """Hexagonal Q2D2: quantize channel pair on hexagonal grid.

    19-point hexagonal grid covers the same area as 5×5=25 square grid
    but with 24% fewer points → log2(19)=4.25 bits vs log2(25)=4.64 bits.

    The 19 points are: center + 6 inner ring + 12 outer ring (hex packing).

    Args:
        pair_a, pair_b: [T] arrays in [-1, 1]
    Returns:
        indices: [T] int indices (0-18)
        recon_a, recon_b: [T] dequantized values
    """
    # Hexagonal grid points (center + 2 rings)
    # Ring 1: 6 points at radius 0.5
    # Ring 2: 12 points at radius 1.0
    # Total: 1 + 6 + 12 = 19
    grid = [(0, 0)]  # center
    for k in range(6):
        angle = k * np.pi / 3
        grid.append((0.5 * np.cos(angle), 0.5 * np.sin(angle)))
    for k in range(12):
        angle = k * np.pi / 6
        grid.append((np.cos(angle), np.sin(angle)))
    grid = np.array(grid)  # [19, 2]

    # Scale grid to [-1, 1] range
    grid = grid / grid.max()

    # Find nearest grid point for each timestep
    points = np.stack([pair_a, pair_b], axis=-1)  # [T, 2]
    dists = np.sum((points[:, None, :] - grid[None, :, :]) ** 2, axis=-1)  # [T, 19]
    indices = dists.argmin(axis=-1)  # [T]

    recon_a = grid[indices, 0]
    recon_b = grid[indices, 1]

    return indices, recon_a, recon_b


def estimate_hex_q2d2_savings(latent, L=5):
    """Compare hexagonal Q2D2 vs independent FSQ bitrate.

    Args:
        latent: [C, T] float in [-1, 1]
    Returns:
        dict with comparison metrics
    """
    if hasattr(latent, 'numpy'):
        latent = latent.detach().cpu().numpy()
    if latent.ndim == 3:
        latent = latent[0]

    C, T = latent.shape
    n_pairs = C // 2

    # Independent FSQ: C channels × log2(L) bits/channel
    independent_bits = C * T * np.log2(L)

    # Square Q2D2: n_pairs × log2(L²) = same bits (no savings from square)
    square_bits = n_pairs * T * np.log2(L * L)

    # Hexagonal Q2D2: n_pairs × log2(19) bits
    hex_bits = n_pairs * T * np.log2(19)

    return {
        'independent_bits': independent_bits,
        'independent_bytes': int(np.ceil(independent_bits / 8)),
        'square_q2d2_bits': square_bits,
        'hex_q2d2_bits': hex_bits,
        'hex_q2d2_bytes': int(np.ceil(hex_bits / 8)),
        'hex_savings_pct': (1 - hex_bits / independent_bits) * 100,
        'bits_per_pair_independent': 2 * np.log2(L),
        'bits_per_pair_hex': np.log2(19),
    }

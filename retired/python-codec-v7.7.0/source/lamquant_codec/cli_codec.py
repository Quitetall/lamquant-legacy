"""
CLI entry point: lamquant-decode

Encode/decode EEG files using the LamQuant codec. All output uses the
standard EEGPacket format so the GUI and benchmarks consume the same schema.

Usage:
    # Encode
    lamquant decode -c model.ckpt -i eeg.npy --encode -o compressed.bin
    lamquant decode -c model.ckpt -i eeg.npy --encode --subband -o compressed.bin

    # Decode to file
    lamquant decode -c model.ckpt -i compressed.bin -o recon.npy
    lamquant decode -c model.ckpt -i compressed.bin --subband -o recon.npy

    # Decode to JSON (for GUI bridge)
    cat window.bin | lamquant decode -c model.ckpt --stdin --json-out
    cat window.bin | lamquant decode -c model.ckpt --stdin --json-out --subband

    # Lossless mode (no checkpoint needed)
    lamquant decode -i eeg.npy --lossless -o compressed.bin --encode
    lamquant decode -i compressed.bin --lossless -o recon.npy

JSON output follows the EEGPacket schema:
    {
        "signal": [[ch0_samples], [ch1_samples], ...],
        "sample_rate": 250,
        "n_channels": 21,
        "mode": "neural",
        "compressed_bytes": 384,
        "raw_bytes": 105000,
        "metadata": {"quality_mode": 2, ...}
    }
"""
import argparse
import json
import sys
import numpy as np

from lamquant_codec.codec_types import EEGPacket
from lamquant_codec.benchmark import Benchmark


def _packet_to_json(packet: EEGPacket) -> dict:
    """Convert EEGPacket to JSON-serializable dict."""
    return {
        'signal': packet.signal.tolist(),
        'sample_rate': packet.sample_rate,
        'n_channels': packet.n_channels,
        'n_samples': packet.n_samples,
        'mode': packet.mode,
        'compressed_bytes': packet.compressed_bytes,
        'raw_bytes': packet.raw_bytes,
        'metadata': packet.metadata,
    }


def _encode_neural(args):
    """Encode EEG → compressed bytes (neural codec)."""
    import torch
    from lamquant_codec import SubbandCodec, TernaryCodec

    # allow_pickle=False — .npy files from untrusted sources can carry
    # pickle payloads that execute arbitrary code on load.
    # (V4 Pro Finding 1 of 7523b3b1 review.)
    eeg = np.load(args.input, allow_pickle=False).astype(np.float32)
    if eeg.ndim == 2:
        eeg = eeg[np.newaxis, ...]
    if eeg.ndim != 3:
        raise SystemExit(
            f"Expected 2D [channels, samples] or 3D [batch, channels, "
            f"samples] input; got {eeg.ndim}D shape {eeg.shape}."
        )

    if args.subband:
        # Encoder.encode is single-trial; refuse multi-trial input
        # rather than silently dropping batches 1..N.
        # (V4 Flash Finding 1 of 7523b3b1 review.)
        if eeg.shape[0] != 1:
            raise SystemExit(
                f"--subband encode is single-trial. Got batch={eeg.shape[0]}. "
                f"Loop over trials and call encode_neural per trial, or "
                f"pre-flatten to shape [channels, samples]."
            )
        # Drive the Encoder so the adaptive-FSQ (LMQ3) path is reached.
        # The legacy direct `codec.compress(...)` path only produces
        # LMQ1 packets and never sees the SNN. Routing through Encoder
        # picks up the production SNN from the PCCP registry pin (or
        # --snn-checkpoint) and emits LMQ3 unless --no-adaptive-fsq is
        # set.
        from lamquant_codec.fileformat import Encoder
        encoder = Encoder(
            checkpoint=args.checkpoint,
            quality=args.quality,
            snn_checkpoint=args.snn_checkpoint,
            adaptive=not args.no_adaptive_fsq,
        )
        signal_2d = eeg[0] if eeg.ndim == 3 else eeg
        compressed, _levels = encoder.encode(signal_2d)
    else:
        x = torch.tensor(eeg)
        codec = TernaryCodec.from_checkpoint(args.checkpoint)
        latent = codec.encode(x)
        compressed = codec.compress(latent)

    with open(args.output, 'wb') as f:
        f.write(compressed)

    cr = eeg.nbytes / len(compressed)
    mode = 'subband' if args.subband else 'gen7.0'
    wire = compressed[:4].decode('ascii', errors='replace') if args.subband else 'n/a'
    print(f"Encoded ({mode}): {eeg.shape} -> {len(compressed)} bytes "
          f"(CR={cr:.1f}x, wire={wire})")
    return 0


def _decode_neural(args, compressed):
    """Decode compressed bytes → EEGPacket (neural codec)."""
    import torch
    from lamquant_codec import SubbandCodec, TernaryCodec

    if args.subband:
        codec = SubbandCodec.from_checkpoint(args.checkpoint)
        latent, quality, lpc_bytes, detail_bytes = codec.decompress(compressed)
        with torch.no_grad():
            recon = codec.model.decode(latent, target_len=313, quantize=True)
        signal = recon[0].numpy()
        return EEGPacket.from_reconstruction(
            signal=signal,
            compressed_bytes=len(compressed),
            mode='neural',
            metadata={'quality_mode': int(quality), 'codec': 'subband'},
        )
    else:
        codec = TernaryCodec.from_checkpoint(args.checkpoint)
        latent = codec.decompress(compressed)
        recon = codec.decode(latent)
        signal = recon[0].numpy() if hasattr(recon[0], 'numpy') else np.asarray(recon[0])
        return EEGPacket.from_reconstruction(
            signal=signal,
            compressed_bytes=len(compressed),
            mode='neural',
            metadata={'codec': 'gen7.0'},
        )


def _encode_lossless(args):
    """Encode EEG → compressed bytes (lossless codec)."""
    from lamquant_codec import LosslessCodec

    eeg = np.load(args.input).astype(np.float64)
    if eeg.ndim == 3:
        eeg = eeg[0]
    codec = LosslessCodec(klt_matrix=None, n_levels=3)
    compressed = codec.compress(eeg)
    with open(args.output, 'wb') as f:
        f.write(compressed)
    cr = eeg.nbytes / len(compressed)
    print(f"Encoded (lossless): {eeg.shape} -> {len(compressed)} bytes (CR={cr:.1f}x)")
    return 0


def _decode_lossless(compressed):
    """Decode compressed bytes → EEGPacket (lossless codec)."""
    from lamquant_codec import LosslessCodec

    codec = LosslessCodec(klt_matrix=None, n_levels=3)
    recon = codec.decompress(compressed)
    return EEGPacket.from_lossless(
        signal=recon,
        compressed_bytes=len(compressed),
        metadata={'codec': 'lossless'},
    )


def main():
    parser = argparse.ArgumentParser(
        prog='lamquant-decode',
        description='LamQuant EEG codec — encode/decode with standard EEGPacket output',
    )
    parser.add_argument('--checkpoint', '-c', help='Model checkpoint path (required for neural modes)')
    parser.add_argument('--input', '-i', help='Input file (.npy or .bin); omit with --stdin')
    parser.add_argument('--output', '-o', help='Output file; omit with --json-out')
    parser.add_argument('--encode', action='store_true', help='Encode mode (input=EEG, output=compressed)')
    parser.add_argument('--subband', action='store_true', help='Use Gen 7.1 subband codec')
    parser.add_argument('--lossless', action='store_true', help='Use Mode 2 lossless codec (no checkpoint needed)')
    parser.add_argument('--snn-checkpoint', type=str, default=None,
                        dest='snn_checkpoint',
                        help='MambaSNN .pt for adaptive FSQ (default: '
                             'registry pin from pccp/registry.yaml). Ignored '
                             'when --no-adaptive-fsq is set.')
    parser.add_argument('--no-adaptive-fsq', action='store_true',
                        dest='no_adaptive_fsq',
                        help='Force uniform LMQ1 FSQ (opt out of '
                             'SNN-driven adaptive FSQ). Adaptive is ON '
                             'by default.')
    parser.add_argument('--quality', type=int, default=2, choices=[0, 1, 2],
                        help='Quality mode: 0=alerting, 1=monitoring, 2=clinical (default: 2)')
    parser.add_argument('--stdin', action='store_true', help='Read input bytes from stdin')
    parser.add_argument('--json-out', action='store_true', help='Write EEGPacket JSON to stdout')
    parser.add_argument('--benchmark', action='store_true',
                        help='If --encode, also decode and print quality report')
    args = parser.parse_args()

    # Refuse contradictory flags rather than silently ignoring one.
    # (V4 Pro Finding 4 + V4 Flash Finding 5 of 7523b3b1 review.)
    if args.snn_checkpoint is not None and args.no_adaptive_fsq:
        parser.error(
            "--snn-checkpoint conflicts with --no-adaptive-fsq "
            "(uniform LMQ1 ignores the SNN). Pick one."
        )

    # Auto-detect mode from file extension
    if args.input and not args.stdin:
        if args.input.endswith('.lml'):
            args.lossless = True
        elif args.input.endswith('.lmq'):
            args.subband = True
    if args.output:
        if args.output.endswith('.lml'):
            args.lossless = True
        elif args.output.endswith('.lmq'):
            args.subband = True

    # --- Encode path ---
    if args.encode:
        if not args.output:
            print('error: --output is required in encode mode', file=sys.stderr)
            return 2
        if args.lossless:
            return _encode_lossless(args)
        if not args.checkpoint:
            print('error: --checkpoint is required for neural encode', file=sys.stderr)
            return 2
        return _encode_neural(args)

    # --- Decode path ---
    # Read compressed input
    if args.stdin:
        compressed = sys.stdin.buffer.read()
    elif args.input:
        with open(args.input, 'rb') as f:
            compressed = f.read()
    else:
        print('error: --input or --stdin is required', file=sys.stderr)
        return 2

    # Decode to EEGPacket
    if args.lossless:
        packet = _decode_lossless(compressed)
    else:
        if not args.checkpoint:
            print('error: --checkpoint is required for neural decode', file=sys.stderr)
            return 2
        packet = _decode_neural(args, compressed)

    # Output
    if args.json_out:
        json.dump(_packet_to_json(packet), sys.stdout)
        sys.stdout.write('\n')
        sys.stdout.flush()
        return 0

    if not args.output:
        print('error: --output or --json-out is required', file=sys.stderr)
        return 2

    np.save(args.output, packet.signal)
    print(f"Decoded: {packet.compressed_bytes} bytes -> {packet.signal.shape} "
          f"(mode={packet.mode})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

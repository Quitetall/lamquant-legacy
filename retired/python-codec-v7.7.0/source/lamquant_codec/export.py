"""
CLI entry point: lamquant-export

Usage:
    lamquant-export --checkpoint model.ckpt --output firmware/firmware_export/
"""
import argparse
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'firmware'))
from export_firmware import (
    export_to_header, compute_firmware_crc,
    export_fsq_lattice, export_toeplitz_seeds,
)


def main():
    parser = argparse.ArgumentParser(description='Export LamQuant model to firmware headers')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--output', '-o', default=os.path.join(ROOT_DIR, 'firmware', 'firmware_export'),
                        help='Output directory for .h files')
    parser.add_argument('--encoder-only', action='store_true', default=True,
                        help='Only export encoder weights (default: true)')
    args = parser.parse_args()

    import torch
    try:
        from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    except ImportError as exc:
        raise SystemExit(
            "lamquant-export needs the neural encoder definition, which now "
            "lives in LamQuant-Neural (pip install lamquant-neural). This "
            "firmware-header exporter reads the trained encoder weights."
        ) from exc

    # Auto-detect model variant from checkpoint
    model = TernaryMobileNetV5_Subband.from_checkpoint(args.checkpoint, device='cpu')
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    header_path = os.path.join(args.output, 'focal_net_weights.h')
    export_to_header(model, header_path, encoder_only=args.encoder_only)

    crc_path = os.path.join(args.output, 'firmware_crc.h')
    compute_firmware_crc(header_path, crc_path)

    export_fsq_lattice(model, os.path.join(args.output, 'fsq_lattice.h'))
    export_toeplitz_seeds(os.path.join(args.output, 'toep_seeds.h'))

    print(f"\n[*] All headers exported to: {args.output}/")


if __name__ == '__main__':
    main()

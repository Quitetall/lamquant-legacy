#!/usr/bin/env python3
"""Entry point for `oh` and `lamquant` CLI commands."""
import sys


def main():
    try:
        from lamquant_codec.cli import main as _main
        _main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

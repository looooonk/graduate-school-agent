from __future__ import annotations

import argparse

from tests.tui_demo import render_demo_snapshot, run_demo_tui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the Rich TUI with fake data and no API calls.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Print one static text render instead of opening a live preview.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML to load. Defaults to ./config.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for repeatable fake event ordering.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Console width for --snapshot output.",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.35,
        help="Average seconds between fake live events.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=5.0,
        help="Seconds to keep the final live frame visible.",
    )
    args = parser.parse_args()

    if args.snapshot:
        print(
            render_demo_snapshot(
                config_path=args.config,
                width=args.width,
                seed=args.seed,
            )
        )
    else:
        run_demo_tui(
            config_path=args.config,
            seed=args.seed,
            frame_delay=args.frame_delay,
            hold_seconds=args.hold,
        )


if __name__ == "__main__":
    main()

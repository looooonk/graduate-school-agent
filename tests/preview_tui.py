from __future__ import annotations

import argparse

from grad_agent.tui import render_demo_snapshot, run_demo_tui


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
        "--width",
        type=int,
        default=120,
        help="Console width for --snapshot output.",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.35,
        help="Seconds between fake live events.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=5.0,
        help="Seconds to keep the final live frame visible.",
    )
    args = parser.parse_args()

    if args.snapshot:
        print(render_demo_snapshot(width=args.width))
    else:
        run_demo_tui(frame_delay=args.frame_delay, hold_seconds=args.hold)


if __name__ == "__main__":
    main()

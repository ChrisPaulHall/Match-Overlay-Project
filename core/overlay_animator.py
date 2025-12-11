import random, time, argparse
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import MatcherConfig
from overlay import (
    init,
    build_overlay_payload,
    build_networth_ticker_payload,
    build_trades_ticker_payload,
    write_overlay,
    bioguide_to_info,
)

config = MatcherConfig()
init(config)

def show_member(bioguide):
    """Show a specific member's overlay"""
    payload = build_overlay_payload(
        bioguide,
        filename=None,
        face_score=0.93,
        full_sim=0.90,
        last_sim=0.88,
        combined_score=0.92,
    )
    write_overlay(payload)

def show_senators_ticker():
    """Show scrolling ticker of senators sorted by net worth"""
    payload = build_networth_ticker_payload(preferred_first="senate")
    write_overlay(payload)

def show_representatives_ticker():
    """Show scrolling ticker of representatives sorted by net worth"""
    payload = build_networth_ticker_payload(preferred_first="house")
    write_overlay(payload)

def show_trades_ticker():
    """Show scrolling ticker of recent congressional stock trades"""
    payload = build_trades_ticker_payload(limit=50)
    write_overlay(payload)

def main():
    parser = argparse.ArgumentParser(
        description="Animate overlay with different modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random member animation (default)
  python core/overlay_animator.py

  # Show senators ticker indefinitely
  python core/overlay_animator.py --senators

  # Show representatives ticker indefinitely
  python core/overlay_animator.py --representatives

  # Show recent trades ticker indefinitely
  python core/overlay_animator.py --trades

  # Cycle through specific members
  python core/overlay_animator.py --interval 20
        """
    )
    parser.add_argument(
        '--senators',
        action='store_true',
        help='Show senators ticker sorted by net worth (runs indefinitely)'
    )
    parser.add_argument(
        '--representatives', '--reps',
        action='store_true',
        dest='representatives',
        help='Show representatives ticker sorted by net worth (runs indefinitely)'
    )
    parser.add_argument(
        '--trades',
        action='store_true',
        help='Show recent trades ticker (runs indefinitely)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Seconds between member changes (default: 45)'
    )

    args = parser.parse_args()

    # Ticker modes - these run indefinitely showing a single ticker
    if args.senators:
        print("Starting senators net worth ticker (press Ctrl+C to stop)...")
        show_senators_ticker()
        while True:
            time.sleep(60)  # Keep alive, ticker auto-scrolls in browser
        return

    if args.representatives:
        print("Starting representatives net worth ticker (press Ctrl+C to stop)...")
        show_representatives_ticker()
        while True:
            time.sleep(60)  # Keep alive, ticker auto-scrolls in browser
        return

    if args.trades:
        print("Starting recent trades ticker (press Ctrl+C to stop)...")
        show_trades_ticker()
        while True:
            time.sleep(60)  # Keep alive, ticker auto-scrolls in browser
        return

    # Default mode: cycle through members
    print(f"Starting member animation (interval: {args.interval}s, press Ctrl+C to stop)...")
    play_list = ["P000197"]
    members = list(bioguide_to_info.keys())

    if not members:
        print("Warning: No members found in bioguide_to_info")
        return

    while True:
        if play_list:
            bioguide = play_list.pop(0)
            print(f"Showing member: {bioguide}")
            show_member(bioguide)
        else:
            bioguide = random.choice(members)
            print(f"Showing random member: {bioguide}")
            show_member(bioguide)
        time.sleep(args.interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")

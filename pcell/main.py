"""Main module for pcell package."""

from pathlib import Path

import matplotlib.pyplot as plt

from pcell.visualization import plot_trajectory


def main():
    """Main entry point."""
    # Path to the CSV file
    csv_path = Path(__file__).parent / "assets" / "behavior" / "WL25_25_12_01_behavior_position.csv"

    # Plot LED trajectory
    plot_trajectory(csv_path, bodypart="LED", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("led_trajectory.png", dpi=150)
    print("Plot saved to led_trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()

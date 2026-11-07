
import json
from pathlib import Path
import matplotlib.pyplot as plt


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "comparison_report.json"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiment_results():
    """Load experiment results from JSON."""

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def visualize_results():

    results = load_experiment_results()

    experiments = list(results.keys())

    accuracy = [
        results[name]["accuracy"]
        for name in experiments
    ]

    macro_f1 = [
        results[name]["macro_f1"] * 100
        for name in experiments
    ]

    latency = [
        results[name]["average_latency"]
        for name in experiments
    ]

    prompt_tokens = [
        results[name]["prompt_tokens"]
        for name in experiments
    ]

    completion_tokens = [
        results[name]["completion_tokens"]
        for name in experiments
    ]

    # Create dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        "CLAIMFAST - EXPERIMENT COMPARISON DASHBOARD",
        fontsize=16,
        fontweight="bold"
    )

    # 1. Classification Accuracy
    axes[0, 0].bar(experiments, accuracy)

    axes[0, 0].set_title("Classification Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0, 110)

    for i, value in enumerate(accuracy):
        axes[0, 0].text(i, value + 2, f"{value:.1f}%",
                        ha="center")

    # 2. Macro F1 Score
    axes[0, 1].bar(experiments, macro_f1)

    axes[0, 1].set_title("Macro F1-Score")
    axes[0, 1].set_ylabel("F1-Score (%)")
    axes[0, 1].set_ylim(0, 110)

    for i, value in enumerate(macro_f1):
        axes[0, 1].text(i, value + 2, f"{value:.2f}%",
                        ha="center")

    # 3. Average Latency
    axes[1, 0].bar(experiments, latency)

    axes[1, 0].set_title("Average Processing Latency")
    axes[1, 0].set_ylabel("Seconds")

    for i, value in enumerate(latency):
        axes[1, 0].text(i, value + 0.03, f"{value:.3f}s",
                        ha="center")

    # 4. Token Usage
    x = range(len(experiments))
    width = 0.35

    axes[1, 1].bar(
        [i - width / 2 for i in x],
        prompt_tokens,
        width,
        label="Prompt Tokens"
    )

    axes[1, 1].bar(
        [i + width / 2 for i in x],
        completion_tokens,
        width,
        label="Completion Tokens"
    )

    axes[1, 1].set_title("Token Usage Comparison")
    axes[1, 1].set_ylabel("Number of Tokens")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(experiments)
    axes[1, 1].legend()

    # Improve dashboard layout
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save dashboard
    output_file = OUTPUT_DIR / "experiment_comparison.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print("\n====================================")
    print(" CLAIMFAST VISUALIZATION COMPLETED")
    print("====================================")

    print("Dashboard saved at:", output_file)

    # Display best experiment
    best = max(results, key=lambda name: results[name]["accuracy"])

    print("\nBest Performing Experiment:", best)
    print("Accuracy:", results[best]["accuracy"], "%")
    print("Macro F1-Score:",
          round(results[best]["macro_f1"], 4))

    plt.show()


if __name__ == "__main__":
    visualize_results()

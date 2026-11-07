
import json
from pathlib import Path
from datetime import date

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "comparison_report.json"
REPORT_DIR = BASE_DIR / "report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "claimfast_experiment_report.md"


def load_results():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_report():

    results = load_results()

    # Find best experiment
    best_accuracy = max(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    best_f1 = max(
        results,
        key=lambda name: results[name]["macro_f1"]
    )

    fastest = min(
        results,
        key=lambda name: results[name]["average_latency"]
    )

    # Report content
    report = []

    report.append("# CLAIMFAST")
    report.append("## Experimental Results and Performance Analysis")
    report.append("")
    report.append(f"**Report Date:** {date.today()}")
    report.append("**Project:** Insurance Claim Classification using LLMs")
    report.append("")

    report.append("---")
    report.append("")

    # 1. Introduction
    report.append("# 1. Introduction")
    report.append(
        "ClaimFast is an experimental project designed to explore "
        "the use of Large Language Models (LLMs) for classifying "
        "insurance claim messages and extracting structured information."
    )
    report.append(
        "This report compares four prompting approaches: "
        "Zero-Shot, Few-Shot 3, Few-Shot 5 and Few-Shot 8."
    )
    report.append("")

    # 2. Experimental methodology
    report.append("# 2. Experimental Methodology")
    report.append("")
    report.append(
        "The project evaluates four prompting configurations "
        "using recorded model predictions and evaluation results."
    )
    report.append("")
    report.append("The experiments are:")
    report.append("- Zero-Shot Prompting")
    report.append("- Few-Shot Prompting with 3 examples")
    report.append("- Few-Shot Prompting with 5 examples")
    report.append("- Few-Shot Prompting with 8 examples")
    report.append("")
    report.append(
        "Each experiment contains 20 evaluated records according "
        "to the comparison report."
    )
    report.append("")

    # 3. Evaluation metrics
    report.append("# 3. Evaluation Metrics")
    report.append("")
    report.append("The following metrics were used:")
    report.append("")
    report.append(
        "- **JSON Validity:** Percentage of responses that "
        "produced valid JSON."
    )
    report.append(
        "- **Classification Accuracy:** Percentage of correctly "
        "classified claim messages."
    )
    report.append(
        "- **Macro F1-Score:** Average F1-score across claim "
        "categories, giving equal weight to each category."
    )
    report.append(
        "- **Average Latency:** Average processing time in seconds."
    )
    report.append(
        "- **Prompt Tokens:** Number of tokens used in the input prompts."
    )
    report.append(
        "- **Completion Tokens:** Number of tokens generated in responses."
    )
    report.append("")

    # 4. Results table
    report.append("# 4. Experimental Results")
    report.append("")
    report.append(
        "| Experiment | Accuracy | Macro F1 | JSON Validity "
        "| Latency (s) | Prompt Tokens | Completion Tokens |"
    )
    report.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for name, data in results.items():

        report.append(
            f"| {name} "
            f"| {data['accuracy']:.2f}% "
            f"| {data['macro_f1']:.4f} "
            f"| {data['json_validity']:.2f}% "
            f"| {data['average_latency']:.3f} "
            f"| {data['prompt_tokens']} "
            f"| {data['completion_tokens']} |"
        )

    report.append("")

    # 5. Accuracy analysis
    report.append("# 5. Classification Accuracy Analysis")
    report.append("")

    for name, data in results.items():
        report.append(
            f"- **{name}:** {data['accuracy']:.2f}% accuracy."
        )

    report.append("")
    report.append(
        f"The highest classification accuracy was achieved by "
        f"**{best_accuracy}**, with an accuracy of "
        f"{results[best_accuracy]['accuracy']:.2f}%."
    )
    report.append("")

    # 6. F1 analysis
    report.append("# 6. Macro F1-Score Analysis")
    report.append("")

    report.append(
        f"The highest Macro F1-score was achieved by "
        f"**{best_f1}**, with a score of "
        f"{results[best_f1]['macro_f1']:.4f}."
    )
    report.append("")

    # 7. Latency analysis
    report.append("# 7. Processing Latency Analysis")
    report.append("")

    for name, data in results.items():
        report.append(
            f"- **{name}:** {data['average_latency']:.3f} seconds."
        )

    report.append("")
    report.append(
        f"**{fastest}** recorded the lowest average latency "
        f"at {results[fastest]['average_latency']:.3f} seconds."
    )
    report.append("")

    # 8. Token usage
    report.append("# 8. Token Usage Analysis")
    report.append("")

    for name, data in results.items():
        total = data["prompt_tokens"] + data["completion_tokens"]

        report.append(
            f"- **{name}:** {total:,} total tokens "
            f"({data['prompt_tokens']:,} prompt + "
            f"{data['completion_tokens']:,} completion)."
        )

    report.append("")

    # 9. Field-level performance
    report.append("# 9. Field-Level Accuracy Analysis")
    report.append("")
    report.append(
        "The following table compares the accuracy of individual "
        "information extraction fields."
    )
    report.append("")

    fields = list(
        next(iter(results.values()))["field_accuracy"].keys()
    )

    report.append(
        "| Field | " + " | ".join(results.keys()) + " |"
    )

    report.append(
        "|---|" + "---:|" * len(results)
    )

    for field in fields:

        values = [
            f"{results[name]['field_accuracy'][field]:.2f}%"
            for name in results
        ]

        report.append(
            "| " + field + " | " + " | ".join(values) + " |"
        )

    report.append("")

    # 10. Visualization
    report.append("# 10. Experimental Visualization")
    report.append("")
    report.append(
        "The following dashboard presents the graphical "
        "comparison of the four experiments."
    )
    report.append("")
    report.append(
        "![ClaimFast Experiment Dashboard]"
        "(../outputs/experiment_comparison.png)"
    )
    report.append("")

    # 11. Discussion
    report.append("# 11. Discussion")
    report.append("")

    report.append(
        "The experimental results show that prompting strategy "
        "has an impact on classification performance."
    )

    report.append(
        f"{best_accuracy} achieved the highest classification "
        f"accuracy, while {best_f1} achieved the highest Macro "
        f"F1-score."
    )

    report.append(
        f"{fastest} provided the lowest processing latency."
    )

    report.append(
        "The Few-Shot 8 approach achieved better classification "
        "performance but also consumed more prompt tokens and "
        "recorded higher latency than the other approaches."
    )

    report.append("")

    # 12. Limitations
    report.append("# 12. Limitations")
    report.append("")
    report.append(
        "- Each experiment contains only 20 evaluated records."
    )
    report.append(
        "- Results may vary with different datasets and models."
    )
    report.append(
        "- Processing latency depends on the execution environment."
    )
    report.append(
        "- Higher token usage can increase computational cost."
    )
    report.append(
        "- Further testing with larger datasets is recommended."
    )
    report.append("")

    # 13. Conclusion
    report.append("# 13. Conclusion")
    report.append("")

    report.append(
        f"Based on the recorded experimental results, "
        f"{best_accuracy} is the best-performing approach "
        f"in terms of classification accuracy."
    )

    report.append(
        f"It achieved {results[best_accuracy]['accuracy']:.2f}% "
        f"accuracy and a Macro F1-score of "
        f"{results[best_accuracy]['macro_f1']:.4f}."
    )

    report.append(
        "However, the choice of prompting strategy should also "
        "consider processing time and token consumption."
    )

    report.append(
        "Future work should evaluate the system using a larger "
        "dataset and additional testing scenarios."
    )

    # Save report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(report))

    print("\n====================================")
    print(" CLAIMFAST REPORT GENERATED")
    print("====================================")
    print("Best Accuracy:", best_accuracy)
    print("Best Macro F1:", best_f1)
    print("Fastest Experiment:", fastest)
    print("\nReport saved at:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    generate_report()

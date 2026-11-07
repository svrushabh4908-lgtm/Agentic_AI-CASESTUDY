
import json
import csv
from pathlib import Path
from collections import Counter

# ==========================================
# CLAIMFAST EXPERIMENT COMPARISON
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

EXPERIMENTS = {
    "Zero-Shot": "zero_shot_results.jsonl",
    "Few-Shot 3": "few_shot_3_results.jsonl",
    "Few-Shot 5": "few_shot_5_results.jsonl",
    "Few-Shot 8": "few_shot_8_results.jsonl",
}

CLAIM_TYPES = [
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim",
]

FIELDS = [
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
]


# ==========================================
# LOAD EXPERIMENT RESULTS
# ==========================================

def load_results(filename):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    records = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as error:
                print(
                    f"Invalid JSONL record at line "
                    f"{line_number}: {error}"
                )

    return records


# ==========================================
# NORMALIZE VALUES FOR COMPARISON
# ==========================================

def normalize(value):

    if isinstance(value, str):

        value = value.strip()

        if value.lower() == "null":
            return None

        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

    return value


# ==========================================
# JSON VALIDITY
# ==========================================

def calculate_json_validity(records):

    valid = 0

    for record in records:

        raw = record.get("raw_response")
        parse_error = record.get("parse_error")

        if not isinstance(raw, str) or parse_error is not None:
            continue

        try:
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                valid += 1

        except (json.JSONDecodeError, TypeError):
            pass

    if not records:
        return 0.0

    return (valid / len(records)) * 100


# ==========================================
# CLASSIFICATION ACCURACY
# ==========================================

def calculate_accuracy(records):

    correct = 0
    total = 0

    for record in records:

        gold = record.get("gold", {})
        prediction = record.get("prediction", {})

        if not isinstance(gold, dict):
            continue

        if not isinstance(prediction, dict):
            prediction = {}

        actual = gold.get("claim_type")
        predicted = prediction.get("claim_type")

        total += 1

        if normalize(actual) == normalize(predicted):
            correct += 1

    if total == 0:
        return 0.0

    return (correct / total) * 100


# ==========================================
# MACRO F1 SCORE
# ==========================================

def calculate_macro_f1(records):

    actual_labels = []
    predicted_labels = []

    for record in records:

        gold = record.get("gold", {})
        prediction = record.get("prediction", {})

        if not isinstance(gold, dict):
            continue

        if not isinstance(prediction, dict):
            prediction = {}

        actual = gold.get("claim_type")
        predicted = prediction.get("claim_type")

        if actual not in CLAIM_TYPES:
            continue

        actual_labels.append(actual)

        if predicted not in CLAIM_TYPES:
            predicted = None

        predicted_labels.append(predicted)

    if not actual_labels:
        return 0.0

    f1_scores = []

    for label in CLAIM_TYPES:

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for actual, predicted in zip(
            actual_labels, predicted_labels
        ):

            if actual == label and predicted == label:
                true_positive += 1

            elif actual != label and predicted == label:
                false_positive += 1

            elif actual == label and predicted != label:
                false_negative += 1

        denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )

        if denominator == 0:
            f1 = 0.0

        else:
            f1 = (2 * true_positive) / denominator

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


# ==========================================
# FIELD LEVEL ACCURACY
# ==========================================

def calculate_field_accuracy(records):

    results = {}

    for field in FIELDS:

        correct = 0
        total = 0

        for record in records:

            gold = record.get("gold", {})
            prediction = record.get("prediction", {})

            if not isinstance(gold, dict):
                continue

            if not isinstance(prediction, dict):
                prediction = {}

            total += 1

            actual = normalize(gold.get(field))
            predicted = normalize(prediction.get(field))

            if actual == predicted:
                correct += 1

        results[field] = (
            (correct / total) * 100 if total else 0.0
        )

    return results


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def calculate_performance(records):

    latencies = [
        record["latency_seconds"]
        for record in records
        if isinstance(record.get("latency_seconds"), (int, float))
    ]

    prompt_tokens = sum(
        record.get("prompt_tokens", 0) or 0
        for record in records
        if isinstance(record.get("prompt_tokens", 0), (int, float))
    )

    completion_tokens = sum(
        record.get("completion_tokens", 0) or 0
        for record in records
        if isinstance(record.get("completion_tokens", 0), (int, float))
    )

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies else 0.0
    )

    return {
        "average_latency": average_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ==========================================
# MAIN COMPARISON
# ==========================================

def main():

    print("\n======================================")
    print(" CLAIMFAST EXPERIMENT COMPARISON")
    print("======================================")

    comparison = {}

    for name, filename in EXPERIMENTS.items():

        print(f"\nEvaluating: {name}")

        records = load_results(filename)

        if not records:
            raise ValueError(
                f"No records found in {filename}"
            )

        performance = calculate_performance(records)

        comparison[name] = {
            "records": len(records),

            "json_validity": calculate_json_validity(
                records
            ),

            "accuracy": calculate_accuracy(records),

            "macro_f1": calculate_macro_f1(records),

            "field_accuracy": calculate_field_accuracy(
                records
            ),

            **performance
        }

        print("Records:", len(records))
        print(
            "Accuracy:",
            f"{comparison[name]['accuracy']:.2f}%"
        )
        print(
            "Macro F1:",
            f"{comparison[name]['macro_f1']:.4f}"
        )

    # ======================================
    # FINAL COMPARISON TABLE
    # ======================================

    print("\n======================================")
    print(" FINAL EXPERIMENT COMPARISON")
    print("======================================")

    print(
        f"{'Method':<15}"
        f"{'Accuracy':<13}"
        f"{'Macro F1':<13}"
        f"{'JSON Valid':<13}"
        f"{'Latency':<12}"
    )

    print("-" * 66)

    for name, result in comparison.items():

        print(
            f"{name:<15}"
            f"{result['accuracy']:<13.2f}"
            f"{result['macro_f1']:<13.4f}"
            f"{result['json_validity']:<13.2f}"
            f"{result['average_latency']:<12.3f}"
        )

    # ======================================
    # FIELD LEVEL COMPARISON
    # ======================================

    print("\n======================================")
    print(" FIELD LEVEL ACCURACY")
    print("======================================")

    for name, result in comparison.items():

        print(f"\n{name}:")

        for field, score in result["field_accuracy"].items():

            print(f"{field}: {score:.2f}%")

    # ======================================
    # TOKEN USAGE
    # ======================================

    print("\n======================================")
    print(" TOKEN USAGE")
    print("======================================")

    for name, result in comparison.items():

        print(f"\n{name}")

        print(
            "Prompt tokens:",
            result["prompt_tokens"]
        )

        print(
            "Completion tokens:",
            result["completion_tokens"]
        )

    # ======================================
    # BEST PERFORMING METHOD
    # ======================================

    best_method = max(
        comparison,
        key=lambda name: comparison[name]["accuracy"]
    )

    print("\n======================================")
    print(" BEST PERFORMING EXPERIMENT")
    print("======================================")

    print("Method:", best_method)

    print(
        "Accuracy:",
        f"{comparison[best_method]['accuracy']:.2f}%"
    )

    print(
        "Macro F1:",
        f"{comparison[best_method]['macro_f1']:.4f}"
    )

    # ======================================
    # SAVE COMPARISON REPORT
    # ======================================

    report_file = OUTPUT_DIR / "comparison_report.json"

    with open(report_file, "w", encoding="utf-8") as file:

        json.dump(
            comparison,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save CSV report
    csv_file = OUTPUT_DIR / "comparison_report.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "Records",
            "Accuracy",
            "Macro F1",
            "JSON Validity",
            "Average Latency",
            "Prompt Tokens",
            "Completion Tokens"
        ])

        for name, result in comparison.items():

            writer.writerow([
                name,
                result["records"],
                result["accuracy"],
                result["macro_f1"],
                result["json_validity"],
                result["average_latency"],
                result["prompt_tokens"],
                result["completion_tokens"]
            ])

    print("\n======================================")
    print(" COMPARISON COMPLETED")
    print("======================================")

    print("JSON report:", report_file)
    print("CSV report:", csv_file)


if __name__ == "__main__":
    main()

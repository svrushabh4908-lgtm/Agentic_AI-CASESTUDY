
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "few_shot_8_results.jsonl"

CLAIM_TYPES = [
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
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
    "police_report_filed"
]


def load_results():
    results = []

    with open(OUTPUT_FILE, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    return results


def calculate_macro_f1(y_true, y_pred):

    scores = []

    for label in CLAIM_TYPES:

        tp = sum(
            1 for t, p in zip(y_true, y_pred)
            if t == label and p == label
        )

        fp = sum(
            1 for t, p in zip(y_true, y_pred)
            if t != label and p == label
        )

        fn = sum(
            1 for t, p in zip(y_true, y_pred)
            if t == label and p != label
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        scores.append(f1)

    return sum(scores) / len(scores)


def main():

    results = load_results()

    total = len(results)

    if total == 0:
        print("No results available.")
        return

    valid_json = sum(
        1 for r in results
        if isinstance(r.get("prediction"), dict)
    )

    correct_claims = 0
    y_true = []
    y_pred = []

    field_matches = Counter()
    field_totals = Counter()

    for result in results:

        gold = result["gold"]
        prediction = result.get("prediction")

        if not isinstance(prediction, dict):
            continue

        actual = gold["claim_type"]
        predicted = prediction.get("claim_type")

        y_true.append(actual)
        y_pred.append(predicted)

        if actual == predicted:
            correct_claims += 1

        for field in FIELDS:

            field_totals[field] += 1

            if gold.get(field) == prediction.get(field):
                field_matches[field] += 1

    accuracy = correct_claims / total

    macro_f1 = calculate_macro_f1(y_true, y_pred)

    latencies = [
        r["latency_seconds"]
        for r in results
        if isinstance(r.get("latency_seconds"), (int, float))
    ]

    average_latency = (
        sum(latencies) / len(latencies) if latencies else 0
    )

    prompt_tokens = sum(
        r.get("prompt_tokens") or 0 for r in results
    )

    completion_tokens = sum(
        r.get("completion_tokens") or 0 for r in results
    )

    print("\n================================")
    print(" CLAIMFAST FEW-SHOT 8 EVALUATION ")
    print("================================")

    print(f"Total messages: {total}")
    print(f"Valid JSON responses: {valid_json}")
    print(f"JSON validity rate: {valid_json / total:.2%}")

    print(f"\nClassification accuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nField-level exact match:")

    for field in FIELDS:
        score = field_matches[field] / total
        print(f"{field}: {score:.2%}")

    print(f"\nAverage latency: {average_latency:.3f} seconds")
    print(f"Total prompt tokens: {prompt_tokens}")
    print(f"Total completion tokens: {completion_tokens}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

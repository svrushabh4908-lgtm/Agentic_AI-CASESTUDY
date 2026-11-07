
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

VALID_TYPES = {
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
}

VALID_URGENCY = {"high", "medium", "low"}

REQUIRED_FIELDS = {
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed"
}

def load_data(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

dev = load_data("dev.jsonl")
test = load_data("test.jsonl")
all_records = dev + test

messages = [record["message"].strip().lower()
            for record in all_records]

print("=== ClaimFast Dataset Validation ===")
print("Development records:", len(dev))
print("Test records:", len(test))
print("Total records:", len(all_records))
print("Unique messages:", len(set(messages)))
print("Duplicate messages:", len(messages) - len(set(messages)))

errors = []

if len(dev) != 20 or len(test) != 60:
    errors.append("Incorrect development/test split")

if len(set(messages)) != len(messages):
    errors.append("Duplicate messages found")

for record in all_records:
    gold = record["gold"]
    record_id = record["id"]

    if set(gold) != REQUIRED_FIELDS:
        errors.append(f"{record_id}: Incorrect JSON fields")

    if gold["claim_type"] not in VALID_TYPES:
        errors.append(f"{record_id}: Invalid claim type")

    if gold["urgency"] not in VALID_URGENCY:
        errors.append(f"{record_id}: Invalid urgency")

    for field in ["injury_reported", "police_report_filed"]:
        if gold.get(field) not in (True, False, None):
            errors.append(f"{record_id}: Invalid {field}")

print("\nClaim type distribution:")
for name, count in Counter(
    record["gold"]["claim_type"] for record in all_records
).items():
    print(f"{name}: {count}")

print("\n=== Validation Result ===")

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print("-", error)
else:
    print("VALIDATION PASSED")

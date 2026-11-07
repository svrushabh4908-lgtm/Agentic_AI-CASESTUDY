
import json
import random
import re
from pathlib import Path
from collections import Counter
from datetime import datetime

# ==========================================
# CLAIMFAST - INSURANCE CLAIM DATASET
# 80 UNIQUE SYNTHETIC MESSAGES
# ==========================================

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLAIM_TYPES = {
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim",
}

URGENCY_LEVELS = {"high", "medium", "low"}

# Format:
# (message, claim_type, urgency, policy_number,
#  incident_date, location, vehicle_reg_no,
#  estimated_amount_inr, injury_reported,
#  police_report_filed)

CASES = [

    # ======================================
    # 1. OWN DAMAGE - 18 MESSAGES
    # ======================================

    (
        "My car MH02AB1234 hit a divider near Andheri on 21 Oct 2026. Repair estimate is Rs 45000.",
        "own_damage", "medium", None,
        "2026-10-21", "Andheri",
        "MH02AB1234", 45000, None, None
    ),
    (
        "My bumper got scratched while parking in Thane. Repair may cost 3000.",
        "own_damage", "low", None,
        None, "Thane", None, 3000, None, None
    ),
    (
        "Meri gaadi ka bumper toot gaya hai kal parking mein.",
        "own_damage", "low", None,
        None, None, None, None, None, None
    ),
    (
        "My car was damaged in heavy rainfall near Dadar. Policy MOT-1001.",
        "own_damage", "medium", "MOT-1001",
        None, "Dadar", None, None, None, None
    ),
    (
        "Accident happened on 15 October 2026 near Bandra. My car MH 01 CD 2345 needs repair.",
        "own_damage", "medium", None,
        "2026-10-15", "Bandra",
        "MH01CD2345", None, None, None
    ),
    (
        "My vehicle hit a pothole and the suspension broke. Repair cost 18000.",
        "own_damage", "medium", None,
        None, None, None, 18000, None, None
    ),
    (
        "Car ka front bumper dent ho gaya, koi injured nahi hua.",
        "own_damage", "low", None,
        None, None, None, None, False, None
    ),
    (
        "A tree branch fell on my parked car in Navi Mumbai.",
        "own_damage", "medium", None,
        None, "Navi Mumbai", None, None, None, None
    ),
    (
        "My car slipped on the road and is now unsafe to drive.",
        "own_damage", "high", None,
        None, None, None, None, None, None
    ),
    (
        "Polcy MOT-1002. My car door damaged yesterday at Powai. Estimate 12000.",
        "own_damage", "low", "MOT-1002",
        None, "Powai", None, 12000, None, None
    ),
    (
        "My car was hit by a falling signboard near Kurla. Damage around 25000.",
        "own_damage", "medium", None,
        None, "Kurla", None, 25000, None, None
    ),
    (
        "Gaadi ka tyre aur axle damage hua, ab gaadi chal nahi sakti aur unsafe hai.",
        "own_damage", "high", None,
        None, None, None, None, None, None
    ),
    (
        "I crashed my vehicle into a wall. No one was injured. Repair estimate 40000.",
        "own_damage", "medium", None,
        None, None, None, 40000, False, None
    ),
    (
        "My vehicle MH12EF3456 was damaged during flooding in Pune.",
        "own_damage", "medium", None,
        None, "Pune", "MH12EF3456", None,
        None, None
    ),
    (
        "Small dent on the left door after parking at Vashi.",
        "own_damage", "low", None,
        None, "Vashi", None, None, None, None
    ),
    (
        "Forwarded message: My car skidded in rain near Nerul. Need claim assistance.",
        "own_damage", "medium", None,
        None, "Nerul", None, None, None, None
    ),
    (
        "On 2026-10-12 my car MH 04 GH 4567 hit a pole in Mulund. Damage Rs 22000.",
        "own_damage", "medium", None,
        "2026-10-12", "Mulund",
        "MH04GH4567", 22000, None, None
    ),
    (
        "Yesterday my mirror broke while reversing. Also last week my bumper was scratched.",
        "own_damage", "low", None,
        None, None, None, None, None, None
    ),

    # ======================================
    # 2. THIRD-PARTY LIABILITY - 14
    # ======================================

    (
        "My car hit a bike near Andheri and the rider was injured.",
        "third_party_liability", "high", None,
        None, "Andheri", None, None, True, None
    ),
    (
        "I accidentally damaged my neighbour's parked car. Repair cost Rs 9000.",
        "third_party_liability", "medium", None,
        None, None, None, 9000, None, None
    ),
    (
        "Meri car se pedestrian ko chot lagi hai near Dadar.",
        "third_party_liability", "high", None,
        None, "Dadar", None, None, True, None
    ),
    (
        "A cyclist was injured after a collision with my insured vehicle.",
        "third_party_liability", "high", None,
        None, None, None, None, True, None
    ),
    (
        "My vehicle MH02JK5678 hit another car in Bandra. Police report filed.",
        "third_party_liability", "medium", None,
        None, "Bandra", "MH02JK5678",
        None, None, True
    ),
    (
        "Accident near Thane. Other driver's car damaged. No injury reported.",
        "third_party_liability", "medium", None,
        None, "Thane", None, None, False, None
    ),
    (
        "Policy MOT-2001. I hit a motorbike on 10 Oct 2026 and the rider got hurt.",
        "third_party_liability", "high", "MOT-2001",
        "2026-10-10", None, None, None, True, None
    ),
    (
        "My car damaged a shop entrance while reversing. Estimated loss 30000.",
        "third_party_liability", "medium", None,
        None, None, None, 30000, None, None
    ),
    (
        "Gaadi se dusri car ko takkar lagi, driver injured hai.",
        "third_party_liability", "high", None,
        None, None, None, None, True, None
    ),
    (
        "I hit a parked scooter in Powai. Repair estimate is 6000.",
        "third_party_liability", "medium", None,
        None, "Powai", None, 6000, None, None
    ),
    (
        "My insured vehicle collided with a taxi. Taxi passenger injured. FIR filed.",
        "third_party_liability", "high", None,
        None, None, None, None, True, True
    ),
    (
        "While driving near Kurla I damaged another person's property.",
        "third_party_liability", "medium", None,
        None, "Kurla", None, None, None, None
    ),
    (
        "My car hit a delivery bike. Police have not been informed yet.",
        "third_party_liability", "medium", None,
        None, None, None, None, None, False
    ),
    (
        "On 2026-10-18 my vehicle hit a pedestrian at Vashi. Ambulance came. Police complaint filed.",
        "third_party_liability", "high", None,
        "2026-10-18", "Vashi",
        None, None, True, True
    ),

    # ======================================
    # 3. THEFT - 14 MESSAGES
    # ======================================

    (
        "My car was stolen from the parking area in Thane.",
        "theft", "medium", None,
        None, "Thane", None, None, None, None
    ),
    (
        "Meri gaadi parking se chori ho gayi near Dadar.",
        "theft", "medium", None,
        None, "Dadar", None, None, None, None
    ),
    (
        "Someone broke my car window and stole the stereo.",
        "theft", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "My vehicle MH12LM6789 was stolen in Pune on 2026-10-14.",
        "theft", "medium", None,
        "2026-10-14", "Pune",
        "MH12LM6789", None, None, None
    ),
    (
        "Policy MOT-3001. My car disappeared from outside my home.",
        "theft", "medium", "MOT-3001",
        None, None, None, None, None, None
    ),
    (
        "My parked car is missing. I filed a police complaint.",
        "theft", "medium", None,
        None, None, None, None, None, True
    ),
    (
        "Car ki battery kisi ne chura li parking mein.",
        "theft", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "My vehicle was stolen near Andheri station. FIR filed.",
        "theft", "medium", None,
        None, "Andheri station", None,
        None, None, True
    ),
    (
        "Yesterday someone stole both side mirrors from my car.",
        "theft", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "My windshield was broken and the car music system was stolen.",
        "theft", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "I parked my car near Vashi and later found it missing.",
        "theft", "medium", None,
        None, "Vashi", None, None, None, None
    ),
    (
        "Car chori hui hai. Police report abhi file nahi ki.",
        "theft", "medium", None,
        None, None, None, None, None, False
    ),
    (
        "My bike MH03NP7890 was stolen on 2026-10-05.",
        "theft", "medium", None,
        "2026-10-05", None,
        "MH03NP7890", None, None, None
    ),
    (
        "Forwarded email: Customer reports stolen car from Mulund parking.",
        "theft", "medium", None,
        None, "Mulund", None, None, None, None
    ),

    # ======================================
    # 4. TOTAL LOSS - 10 MESSAGES
    # ======================================

    (
        "My car was declared a total loss after the accident.",
        "total_loss", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "The garage says the vehicle is beyond economical repair.",
        "total_loss", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "Meri car puri tarah damage ho gayi, surveyor ne total loss declare kiya.",
        "total_loss", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "My vehicle MH04QR8901 is written off after a collision.",
        "total_loss", "medium", None,
        None, None, "MH04QR8901", None,
        None, None
    ),
    (
        "Policy MOT-4001. Surveyor confirmed constructive total loss.",
        "total_loss", "medium", "MOT-4001",
        None, None, None, None, None, None
    ),
    (
        "After flooding in Mumbai the insurer classified my car as total loss.",
        "total_loss", "medium", None,
        None, "Mumbai", None, None, None, None
    ),
    (
        "My car cannot be repaired economically after the crash on 2026-10-09.",
        "total_loss", "medium", None,
        "2026-10-09", None, None, None, None, None
    ),
    (
        "The survey report states my insured vehicle is a complete write-off.",
        "total_loss", "medium", None,
        None, None, None, None, None, None
    ),
    (
        "Car total loss declare hui, aur driver ko injury bhi hui.",
        "total_loss", "high", None,
        None, None, None, None, True, None
    ),
    (
        "My vehicle was declared total loss following an accident near Panvel. FIR filed.",
        "total_loss", "medium", None,
        None, "Panvel", None, None, None, True
    ),

    # ======================================
    # 5. GLASS ONLY - 12 MESSAGES
    # ======================================

    (
        "My windshield cracked due to a flying stone.",
        "glass_only", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Only my side window glass was broken in the parking area.",
        "glass_only", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Car ka front glass crack ho gaya near Bandra.",
        "glass_only", "low", None,
        None, "Bandra", None, None, None, None
    ),
    (
        "My rear windshield shattered while parked at Powai.",
        "glass_only", "low", None,
        None, "Powai", None, None, None, None
    ),
    (
        "A small crack appeared on my car windshield. Nothing else damaged.",
        "glass_only", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Policy MOT-5001. Only the driver-side glass needs replacement.",
        "glass_only", "low", "MOT-5001",
        None, None, None, None, None, None
    ),
    (
        "Windshild broken after stone hit it on 2026-10-11.",
        "glass_only", "low", None,
        "2026-10-11", None, None, None, None, None
    ),
    (
        "My car glass broke during parking. Estimated replacement 5000.",
        "glass_only", "low", None,
        None, None, None, 5000, None, None
    ),
    (
        "Only the back window glass of MH02ST9012 was damaged.",
        "glass_only", "low", None,
        None, None, "MH02ST9012",
        None, None, None
    ),
    (
        "Meri car ka sheesha toot gaya hai, aur koi damage nahi.",
        "glass_only", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Passenger-side car window cracked near Nerul. Repair estimate Rs 2500.",
        "glass_only", "low", None,
        None, "Nerul", None, 2500, None, None
    ),
    (
        "On 2026-10-16 a branch broke only the windshield at Thane.",
        "glass_only", "low", None,
        "2026-10-16", "Thane", None, None, None, None
    ),

    # ======================================
    # 6. NOT A CLAIM - 12 MESSAGES
    # ======================================

    (
        "Please send me my insurance renewal certificate.",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "What is my no claim bonus percentage?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Policy renewal kab hoga?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "I want to update my mobile number in the insurance policy.",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Can you explain the benefits of zero depreciation cover?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Please provide a copy of policy MOT-6001.",
        "not_a_claim", "low", "MOT-6001",
        None, None, None, None, None, None
    ),
    (
        "I need information about premium payment options.",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "What documents are required to renew car insurance?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "How do I change my registered address to Mumbai?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Can I transfer my insurance policy to another vehicle?",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "My policy expires next month. Please explain renewal charges.",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
    (
        "Please tell me whether roadside assistance is included in my plan.",
        "not_a_claim", "low", None,
        None, None, None, None, None, None
    ),
]


# ==========================================
# VALIDATE THE SOURCE DATA
# ==========================================

assert len(CASES) == 80, (
    f"Expected 80 messages, found {len(CASES)}"
)

messages = [case[0].strip().casefold() for case in CASES]

assert len(set(messages)) == 80, (
    "Duplicate source messages detected"
)

for index, case in enumerate(CASES, start=1):

    assert len(case) == 10, (
        f"Record {index} has incorrect number of fields"
    )

    (
        message,
        claim_type,
        urgency,
        policy_number,
        incident_date,
        location,
        vehicle_reg_no,
        amount,
        injury,
        police,
    ) = case

    assert claim_type in CLAIM_TYPES
    assert urgency in URGENCY_LEVELS
    assert isinstance(message, str) and message.strip()

    assert injury is None or type(injury) is bool
    assert police is None or type(police) is bool

    assert amount is None or (
        type(amount) is int and amount >= 0
    )

    if incident_date is not None:
        datetime.strptime(incident_date, "%Y-%m-%d")

    if vehicle_reg_no is not None:
        assert re.fullmatch(
            r"[A-Z0-9]+", vehicle_reg_no
        ), f"Invalid registration: {vehicle_reg_no}"


# ==========================================
# CONVERT INTO REQUIRED JSON STRUCTURE
# ==========================================

records = []

for index, case in enumerate(CASES, start=1):

    (
        message,
        claim_type,
        urgency,
        policy_number,
        incident_date,
        location,
        vehicle_reg_no,
        amount,
        injury,
        police,
    ) = case

    records.append({
        "id": f"CF{index:03d}",
        "message": message,
        "gold": {
            "claim_type": claim_type,
            "urgency": urgency,
            "policy_number": policy_number,
            "incident_date": incident_date,
            "location": location,
            "vehicle_reg_no": vehicle_reg_no,
            "estimated_amount_inr": amount,
            "injury_reported": injury,
            "police_report_filed": police,
        },
    })


# ==========================================
# SPLIT DEVELOPMENT AND TEST DATA
# ==========================================

# Stratified splitting keeps every claim
# category represented in both datasets.

grouped = {
    claim_type: [] for claim_type in CLAIM_TYPES
}

for record in records:
    claim_type = record["gold"]["claim_type"]
    grouped[claim_type].append(record)

# Development allocation: total 20
dev_counts = {
    "own_damage": 4,
    "third_party_liability": 4,
    "theft": 4,
    "total_loss": 2,
    "glass_only": 3,
    "not_a_claim": 3,
}

dev_data = []
test_data = []

for claim_type in sorted(CLAIM_TYPES):

    group = grouped[claim_type]
    random.shuffle(group)

    dev_count = dev_counts[claim_type]

    dev_data.extend(group[:dev_count])
    test_data.extend(group[dev_count:])

random.shuffle(dev_data)
random.shuffle(test_data)

assert len(dev_data) == 20
assert len(test_data) == 60


# ==========================================
# SAVE JSONL FILES
# ==========================================

def save_jsonl(filename, data):

    filepath = DATA_DIR / filename

    with open(filepath, "w", encoding="utf-8") as file:

        for record in data:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved: {filepath.name}")


save_jsonl("dev.jsonl", dev_data)
save_jsonl("test.jsonl", test_data)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n=== ClaimFast Dataset Generated ===")

print("Development messages:", len(dev_data))
print("Testing messages:", len(test_data))
print("Total messages:", len(records))
print("Unique messages:", len(set(messages)))

print("\nClaim Type Distribution:")

counts = Counter(
    record["gold"]["claim_type"]
    for record in records
)

for claim_type in sorted(CLAIM_TYPES):
    print(f"{claim_type}: {counts[claim_type]}")

print("\nDataset generation completed!")
print("Next: Run validate_dataset.py")

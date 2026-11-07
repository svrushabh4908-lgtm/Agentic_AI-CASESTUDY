
# ClaimFast – Labelling Guide

## 1. Claim Types
- own_damage: Damage to the insured vehicle.
- third_party_liability: A claim involving damage or injury to a third party.
- theft: A vehicle or vehicle-related item has been stolen.
- total_loss: The vehicle is explicitly described as a total loss or beyond economical repair.
- glass_only: Only the glass is damaged.
- not_a_claim: The message does not report a new insurance claim.

## 2. Urgency
- high: An injury is reported, or the vehicle is undrivable and unsafe.
- medium: A significant incident requiring attention but no high-urgency condition is reported.
- low: Minor incidents and general enquiries.

## 3. Extraction Rules
- policy_number: Extract the stated policy number. Otherwise null.
- incident_date: Use YYYY-MM-DD where an exact date can be determined. Otherwise null.
- location: Extract the incident location. Otherwise null.
- vehicle_reg_no: Convert registration numbers to uppercase and remove spaces.
- estimated_amount_inr: Extract the amount in rupees as an integer. Otherwise null.
- injury_reported: true if an injury is explicitly reported, false if explicitly denied, otherwise null.
- police_report_filed: true if a report is explicitly filed, false if explicitly not filed, otherwise null.

## 4. Ambiguity Rules
- If glass was broken and an item was stolen, classify as theft.
- If multiple incidents appear, use the incident requiring the highest urgency; if tied, use the first reported incident.
- Do not invent missing details.
- Use null for unknown extraction values.
- Use consistent rules for Hinglish messages and spelling mistakes.


import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# =====================================
# CLAIMFAST - FEW SHOT (3 EXAMPLES)
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "llama3.2:3b"
API_URL = "http://localhost:11434/api/chat"


# =====================================
# LOAD PROMPT
# =====================================

def load_prompt():
    path = PROMPT_DIR / "few_shot_3.txt"

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# =====================================
# LOAD DEVELOPMENT DATA
# =====================================

def load_dataset():
    records = []

    path = DATA_DIR / "dev.jsonl"

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


# =====================================
# CALL OLLAMA API
# =====================================

def get_prediction(message, prompt_template):

    prompt = prompt_template.replace(
        "{message}", message
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start_time = time.perf_counter()

    with urllib.request.urlopen(
        request, timeout=300
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    latency = time.perf_counter() - start_time

    raw_response = result["message"]["content"]

    try:
        prediction = json.loads(raw_response)
        parse_error = None

    except json.JSONDecodeError as error:
        prediction = None
        parse_error = str(error)

    return {
        "prediction": prediction,
        "raw_response": raw_response,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": result.get("prompt_eval_count"),
        "completion_tokens": result.get("eval_count"),
        "parse_error": parse_error
    }


# =====================================
# MAIN EXPERIMENT
# =====================================

def main():

    print("\n================================")
    print(" CLAIMFAST FEW-SHOT EXPERIMENT")
    print(" PROMPT EXAMPLES: 3")
    print("================================\n")

    prompt_template = load_prompt()
    records = load_dataset()

    results = []

    for index, record in enumerate(records, start=1):

        print(
            f"Processing {index}/{len(records)}: "
            f"{record['id']}"
        )

        try:

            response = get_prediction(
                record["message"],
                prompt_template
            )

            result = {
                "id": record["id"],
                "message": record["message"],
                "gold": record["gold"],
                **response
            }

            results.append(result)

            if response["prediction"] is None:

                print("  Invalid JSON response")

            else:

                print(
                    "  Predicted:",
                    response["prediction"].get("claim_type")
                )

            print(
                "  Latency:",
                response["latency_seconds"],
                "seconds"
            )

        except (
            urllib.error.URLError,
            TimeoutError
        ) as error:

            print("Connection error:", error)
            break

    output_file = (
        OUTPUT_DIR / "few_shot_3_results.jsonl"
    )

    with open(
        output_file, "w", encoding="utf-8"
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n================================")
    print(" FEW-SHOT EXPERIMENT COMPLETED")
    print("================================")

    print("Total records processed:", len(results))
    print("Results saved:", output_file)


if __name__ == "__main__":
    main()

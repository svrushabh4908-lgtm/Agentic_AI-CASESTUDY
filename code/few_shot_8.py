
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# ==========================================
# CLAIMFAST - FEW-SHOT EXPERIMENT (8 EXAMPLES)
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "llama3.2:3b"
API_URL = "http://localhost:11434/api/chat"

# ==========================================
# LOAD PROMPT
# ==========================================

def load_prompt():

    prompt_path = PROMPT_DIR / "few_shot_8.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


# ==========================================
# LOAD DEVELOPMENT DATASET
# ==========================================

def load_dataset():

    records = []
    dataset_path = DATA_DIR / "dev.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if line.strip():

                try:
                    records.append(json.loads(line))

                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    )

    return records


# ==========================================
# CALL OLLAMA API
# ==========================================

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
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start_time = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=300
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


# ==========================================
# SAVE EXPERIMENT RESULTS
# ==========================================

def save_results(results):

    output_file = (
        OUTPUT_DIR / "few_shot_8_results.jsonl"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    return output_file


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    print("\n======================================")
    print(" CLAIMFAST FEW-SHOT EXPERIMENT")
    print(" PROMPT EXAMPLES: 8")
    print(" MODEL: LLAMA 3.2 3B")
    print("======================================\n")

    try:

        prompt_template = load_prompt()
        records = load_dataset()

    except (FileNotFoundError, ValueError) as error:

        print("Setup error:", error)
        return

    print("Development records:", len(records))
    print("\nStarting classification...\n")

    results = []

    for index, record in enumerate(records, start=1):

        record_id = record.get("id", f"record_{index}")

        print(
            f"Processing {index}/{len(records)}: {record_id}"
        )

        try:

            response = get_prediction(
                record["message"],
                prompt_template
            )

            result = {
                "id": record_id,
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
            print("Experiment stopped.")
            break

        except (KeyError, TypeError) as error:

            print("Invalid dataset record:", error)
            print("Experiment stopped.")
            break

    output_file = save_results(results)

    print("\n======================================")
    print(" FEW-SHOT 8 EXPERIMENT COMPLETED")
    print("======================================")

    print("Total records processed:", len(results))
    print("Expected records:", len(records))
    print("Results saved:", output_file)

    if len(results) != len(records):
        print("WARNING: Not all records were processed.")
    else:
        print("All development records processed successfully.")


if __name__ == "__main__":
    main()

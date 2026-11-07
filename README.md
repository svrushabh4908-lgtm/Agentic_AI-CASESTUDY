# ClaimFast – AI-Powered Insurance Claim Classification

## 1. Project Overview

ClaimFast is an experimental AI-based insurance claim classification system that uses Large Language Models (LLMs) to classify vehicle insurance claim messages and extract structured information.

The project explores how different prompting techniques affect the accuracy, reliability, processing time, and token usage of an LLM.

The system uses Ollama with the Llama 3.2 (3B) model to process customer messages and generate structured JSON responses.

## 2. Project Objectives

- Classify vehicle insurance claims automatically.
- Extract important information from customer messages.
- Generate structured JSON outputs.
- Compare Zero-Shot and Few-Shot prompting techniques.
- Evaluate classification accuracy and Macro F1-score.
- Measure processing latency and token usage.
- Identify the best-performing prompting approach.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Ollama | Local LLM execution |
| Llama 3.2 (3B) | Language model |
| JSON | Structured output |
| Matplotlib | Data visualization |
| CSV | Experiment result storage |
| Markdown | Project documentation |

## 4. Project Structure

```text
ClaimFast/
│
├── code/
│   ├── zero_shot.py
│   ├── few_shot_3.py
│   ├── few_shot_5.py
│   ├── few_shot_8.py
│   ├── evaluate_zero_shot.py
│   ├── compare_experiments.py
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── visualize_results.py
│   ├── generate_report.py
│   └── README.md
│
├── data/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── labelling_guide.md
│
├── demo/
│
├── outputs/
│   ├── comparison_report.json
│   ├── comparison_report.csv
│   ├── experiment_comparison.png
│   ├── zero_shot_results.jsonl
│   ├── few_shot_3_results.jsonl
│   ├── few_shot_5_results.jsonl
│   └── few_shot_8_results.jsonl
│
├── prompts/
│   ├── zero_shot.txt
│   ├── few_shot_3.txt
│   ├── few_shot_5.txt
│   └── few_shot_8.txt
│
├── report/
│   └── claimfast_experiment_report.md
│
└── README.md
```

## 5. Experimental Approaches

The project evaluates four prompting techniques:

**Zero-Shot Prompting**

The model receives instructions without any examples.

**Few-Shot 3**

The model receives three examples before processing the customer message.

**Few-Shot 5**

The model receives five examples to improve its understanding of claim classification.

**Few-Shot 8**

The model receives eight examples to provide additional contextual guidance.

## 6. Experimental Results

Each experiment was evaluated using 20 test records.

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Zero-Shot | 75% | 0.7783 |
| Few-Shot 3 | 85% | 0.8434 |
| Few-Shot 5 | 80% | 0.8278 |
| Few-Shot 8 | 90% | 0.9101 |

### Performance Analysis

- Zero-Shot achieved 75% classification accuracy.
- Few-Shot 3 improved accuracy to 85%.
- Few-Shot 5 achieved 80% accuracy.
- Few-Shot 8 achieved the highest accuracy of 90%.

Few-Shot 8 was the best-performing approach in terms of classification accuracy and Macro F1-score.

Few-Shot 5 achieved the lowest average processing latency of approximately 1.207 seconds.

All four approaches achieved 100% JSON validity in the evaluated test records.

## 7. Performance Metrics

The project uses the following evaluation metrics:

- Classification Accuracy
- Macro F1-score
- JSON Validity Rate
- Field-Level Exact Match
- Average Processing Latency
- Prompt Token Usage
- Completion Token Usage

## 8. Installation and Setup

### Step 1: Install Python

Install Python 3.9 or later.

### Step 2: Install Ollama

Download Ollama from:

https://ollama.com/

### Step 3: Download the Model

Open the terminal and execute:

```bash
ollama pull llama3.2:3b
```

### Step 4: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ClaimFast
```

Replace the repository URL with your actual GitHub repository URL.

## 9. Running the Project

Run the following commands from the main ClaimFast directory.

### Run Zero-Shot Experiment

```bash
python3 code/zero_shot.py
```

### Run Few-Shot 3 Experiment

```bash
python3 code/few_shot_3.py
```

### Run Few-Shot 5 Experiment

```bash
python3 code/few_shot_5.py
```

### Run Few-Shot 8 Experiment

```bash
python3 code/few_shot_8.py
```

### Compare Experiments

```bash
python3 code/compare_experiments.py
```

### Generate Visualization

```bash
python3 code/visual
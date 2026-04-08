---
title: BugPilot
emoji: 🐛
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# AI Bug Report Triager Environment

This project provides a production-ready OpenEnv-compatible simulation environment where an Artificial Intelligence Agent acts as a primary software triage engineer. 

The environment tests the ability of an AI agent to read a bug report, classify its severity, identify the matching software component, cross-reference previous issues to flag duplicates, and assign it to the appropriate team engineer.

## Architecture

* **Environment (`environment.py`)**: Core state tracking using the standard Reinforcement Learning workflow style (`reset()`, `step()`, `state`).
* **Grader (`grader.py`)**: Generates continuous reward scores from 0.0 to 1.0 based on matching performance with partial credit.
* **REST API (`api/main.py`)**: Exposes the environment steps via a FastAPI router to match OpenEnv specifications.
* **Data Variants (`data/*.json`)**: Diverse complexity testing via mock issues (easy, medium, hard).
* **Agent Baseline (`inference.py`)**: A zero-dependency script demonstrating how to parse states and query actions over the environment.

## Installation

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

## Running the API Locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Visit http://127.0.0.1:8000/docs for Swagger interactive documentation.

## Running the Baseline Agent

Run the predefined heuristic script to see standard interaction:
```bash
python inference.py
```

## API Usage Example

**Reset Environmental State:**
```bash
curl -X POST http://127.0.0.1:8000/reset
```

**Step Example:**
```bash
curl -X POST http://127.0.0.1:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "set_severity", "value": "critical"}'
```

**Finish & Grade:**
```bash
curl -X POST http://127.0.0.1:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "submit"}'
```

## Deployment steps (Hugging Face Spaces)
1. Ensure your Hugging Face Space is configured as a **Docker** Space.
2. Provide this repository's contents unchanged. Space automatically detects the `Dockerfile` and builds the environment.
3. The Space will expose Port 8000 on its routing mechanism naturally as specified by the EXPOSE tag in the Docker image.

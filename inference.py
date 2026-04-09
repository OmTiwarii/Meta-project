from environment import BugTriagerEnv
import json
import re
import os
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# LLM PROXY CONFIGURATION (injected by hackathon validator)
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

AGENT_SYSTEM_PROMPT = """You are an expert software engineering AI specialized in bug report triage.
You are operating inside a Reinforcement Learning environment.
Your goal is to maximize your reward score by making the best triaging decisions.

You will receive a bug report state and must respond with ONLY a valid JSON action.

Available actions:
- {{ "action": "set_severity", "value": "critical|high|medium|low" }}
- {{ "action": "set_component", "value": "auth|payments|ui|backend|mobile" }}
- {{ "action": "flag_duplicate", "duplicate_of": "<issue-id>" }}
- {{ "action": "assign", "assignee": "<engineer-name>" }}
- {{ "action": "request_info", "fields": ["steps_to_reproduce", "device"] }}
- {{ "action": "submit" }}

Current Bug Report State:
{state_json}

Respond ONLY with a valid JSON action. No explanation.
"""


def build_agent_prompt(state):
    """Formats the current environment state into an LLM-ready prompt."""
    agent_visible_state = {
        "report": state.get("report", {}),
        "existing_issues": state.get("existing_issues", []),
        "team_roster": state.get("team_roster", []),
        "current_progress": state.get("current_progress", {}),
    }
    return AGENT_SYSTEM_PROMPT.format(
        state_json=json.dumps(agent_visible_state, indent=2)
    )


def call_llm(prompt):
    # type: (str) -> Optional[dict]
    """
    Call the LLM via the hackathon-provided proxy endpoint.
    Uses API_BASE_URL and API_KEY from environment variables.
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert bug triage AI. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception:
        return None


def is_duplicate(report_text, existing_issue_text):
    """Duplicate detection using Jaccard Similarity."""
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        stopwords = {"the", "is", "in", "and", "to", "on", "it", "my", "for", "of", "with"}
        return set(w for w in text.split() if w not in stopwords and len(w) > 2)

    r_tokens = normalize(report_text)
    i_tokens = normalize(existing_issue_text)
    if not r_tokens or not i_tokens:
        return False
    intersection = len(r_tokens & i_tokens)
    union = len(r_tokens | i_tokens)
    return (intersection / union) > 0.20 if union > 0 else False


def heuristic_action(state):
    """Fallback heuristic agent when LLM is unavailable."""
    progress = state.get("current_progress", {})
    report_data = state["report"]
    full_text = (report_data["title"] + " " + report_data["body"]).lower()

    if not progress.get("severity"):
        if "crash" in full_text or "timeout" in full_text or "504" in full_text:
            return {"action": "set_severity", "value": "critical"}
        elif "broken" in full_text or "500" in full_text or "unauthorized" in full_text:
            return {"action": "set_severity", "value": "high"}
        elif "typo" in full_text:
            return {"action": "set_severity", "value": "low"}
        return {"action": "set_severity", "value": "medium"}

    if not progress.get("component"):
        if "payment" in full_text or "checkout" in full_text:
            return {"action": "set_component", "value": "payments"}
        elif "login" in full_text or "password" in full_text:
            return {"action": "set_component", "value": "auth"}
        elif "mobile" in full_text or "android" in full_text:
            return {"action": "set_component", "value": "mobile"}
        elif "button" in full_text or "ui" in full_text:
            return {"action": "set_component", "value": "ui"}
        return {"action": "set_component", "value": "backend"}

    # Check duplicate
    for issue in state.get("existing_issues", []):
        issue_text = issue["title"] + " " + issue.get("body", "")
        if is_duplicate(full_text, issue_text):
            if not progress.get("duplicate_of"):
                return {"action": "flag_duplicate", "duplicate_of": issue["id"]}

    if not progress.get("assignee"):
        return {"action": "assign", "assignee": "alice"}

    return {"action": "submit"}


def run_episode(env, task_name):
    """Run a single triage episode and emit structured output blocks."""
    state = env.reset()
    step_count = 0
    total_reward = 0.0
    max_steps = state.get("meta", {}).get("max_steps", 10)

    print("[START] task={}".format(task_name), flush=True)

    done = False
    while not done and step_count < max_steps:
        prompt = build_agent_prompt(state)
        action = call_llm(prompt)
        if action is None:
            action = heuristic_action(state)

        step_count += 1
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        state = new_state

        print("[STEP] step={} reward={:.2f} action={}".format(
            step_count, reward, action.get("action", "?")
        ), flush=True)

    # Grade the result
    from grader import grade
    actual_gold = env.current_scenario["gold"]
    grader_output = grade(state["current_progress"], actual_gold, debug=True)
    # Score must be strictly between 0 and 1 (not 0.0 and not 1.0)
    final_score = max(0.01, min(0.99, grader_output["score"]))

    print("[END] task={} score={:.2f} steps={}".format(
        task_name, final_score, step_count
    ), flush=True)

    return total_reward


def run_baseline_agent():
    """
    Run 3 graded triage episodes to satisfy OpenEnv Phase 2 Task Validation.
    Each episode uses the LLM proxy (API_BASE_URL + API_KEY) with heuristic fallback.
    """
    env = BugTriagerEnv()

    tasks = ["bug_triage_1", "bug_triage_2", "bug_triage_3"]
    total = 0.0
    for task_name in tasks:
        total += run_episode(env, task_name)

    return total


if __name__ == "__main__":
    run_baseline_agent()

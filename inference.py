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
  → Use to classify how urgent/impactful the bug is

- {{ "action": "set_component", "value": "auth|payments|ui|backend|mobile" }}
  → Use to identify which part of the system the bug belongs to

- {{ "action": "flag_duplicate", "duplicate_of": "<issue-id>" }}
  → Use when this bug already exists in the existing_issues list

- {{ "action": "assign", "assignee": "<engineer-name>" }}
  → Use to assign to the correct engineer from the team roster

- {{ "action": "request_info", "fields": ["steps_to_reproduce", "device"] }}
  → Use when the report is vague or missing key context

- {{ "action": "submit" }}
  → Use when all fields are filled and triage is complete

Scoring rules you must optimize for:
- Correct severity classification: +0.30 (on submit)
- Correct component identification: +0.30 (on submit)
- Correct duplicate detection: +0.25 (on submit)
- Correct engineer assignment: +0.15 (on submit)
- Intermediate correct step: +0.05 to +0.10
- Wrong/invalid action: -0.05
- Repeated same action: -0.02
- Exceeding max steps (10): -0.20

Previous action reward: {previous_reward}
Current session score: {session_score}

Current Bug Report State:
{state_json}

Always respond in pure JSON only. No explanation outside the JSON.
"""


def build_agent_prompt(state: dict, previous_reward: float = 0.0, session_score: float = 0.0) -> str:
    """Formats the current environment state into an LLM-ready system prompt."""
    agent_visible_state = {
        "report": state.get("report", {}),
        "existing_issues": state.get("existing_issues", []),
        "team_roster": state.get("team_roster", []),
        "current_progress": state.get("current_progress", {}),
        "meta": state.get("meta", {})
    }
    return AGENT_SYSTEM_PROMPT.format(
        previous_reward=f"{previous_reward:+.2f}",
        session_score=f"{session_score:+.2f}",
        state_json=json.dumps(agent_visible_state, indent=2)
    )


def call_llm(prompt: str) -> Optional[dict]:
    """
    Call the LLM via the hackathon-provided proxy endpoint.
    Uses API_BASE_URL and API_KEY from environment variables.
    Returns parsed JSON action dict or None on failure.
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
                {"role": "system", "content": "You are an expert bug triage AI. Respond ONLY with a valid JSON action."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception:
        return None


def is_duplicate(report_text: str, existing_issue_text: str) -> bool:
    """Stronger duplicate detection using Jaccard Similarity on normalized text."""
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        stopwords = {"the", "is", "in", "and", "to", "on", "it", "my", "for", "of", "with"}
        tokens = set(word for word in text.split() if word not in stopwords and len(word) > 2)
        return tokens

    r_tokens = normalize(report_text)
    i_tokens = normalize(existing_issue_text)

    if not r_tokens or not i_tokens:
        return False

    intersection = len(r_tokens.intersection(i_tokens))
    union = len(r_tokens.union(i_tokens))
    similarity = intersection / union if union > 0 else 0
    return similarity > 0.20


def heuristic_action(state: dict) -> dict:
    """Fallback heuristic agent when LLM is unavailable."""
    progress = state.get("current_progress", {})
    report_data = state["report"]
    full_text = (report_data["title"] + " " + report_data["body"]).lower()

    if not progress.get("severity"):
        if "crash" in full_text or "timeout" in full_text or "504" in full_text:
            return {"action": "set_severity", "value": "critical"}
        elif "broken" in full_text or "500" in full_text:
            return {"action": "set_severity", "value": "high"}
        elif "typo" in full_text:
            return {"action": "set_severity", "value": "low"}
        else:
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
        else:
            return {"action": "set_component", "value": "backend"}

    if not progress.get("assignee"):
        return {"action": "assign", "assignee": "alice"}

    return {"action": "submit"}


def run_baseline_agent():
    """
    LLM-powered inference agent that calls the hackathon's LiteLLM proxy
    and emits required [START]/[STEP]/[END] structured output blocks.
    Falls back to heuristic if LLM is unavailable.
    """
    env = BugTriagerEnv()
    state = env.reset()
    total_reward = 0.0
    step_count = 0
    session_score = 0.0
    task_name = "bug_triage"

    # ── REQUIRED: Signal start of task ────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    done = False
    max_steps = state.get("meta", {}).get("max_steps", 10)

    while not done and step_count < max_steps:
        # Build LLM prompt from current state
        prompt = build_agent_prompt(state, previous_reward=0.0, session_score=session_score)

        # Try LLM first, fall back to heuristic
        action = call_llm(prompt)
        if action is None:
            action = heuristic_action(state)

        step_count += 1
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        session_score = total_reward
        state = new_state

        # ── REQUIRED: Emit a step line per action ─────────────────────────────
        print(f"[STEP] step={step_count} reward={reward:.2f} action={action.get('action', '?')}", flush=True)

    # Grade final result
    from grader import grade
    actual_gold = env.current_scenario["gold"]
    grader_output = grade(state["current_progress"], actual_gold, debug=True)
    final_score = grader_output["score"]

    # ── REQUIRED: Signal end of task ──────────────────────────────────────────
    print(f"[END] task={task_name} score={final_score:.2f} steps={step_count}", flush=True)

    return total_reward


if __name__ == "__main__":
    run_baseline_agent()

from environment import BugTriagerEnv
import json
import re

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED AGENT SYSTEM PROMPT
# Action names aligned to match BugTriagerEnv's supported action space:
#   set_severity, set_component, flag_duplicate, assign, request_info, submit
# ─────────────────────────────────────────────────────────────────────────────
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
    """
    Option A: Formats the current environment state into an LLM-ready system prompt.
    
    Usage:
        state = env.reset()
        prompt = build_agent_prompt(state, previous_reward=0.0, session_score=0.0)
        # → Send `prompt` to any LLM (GPT-4, Llama, Gemini, etc.)
        # → Parse the LLM's JSON response
        # → Call env.step(json_response)
    
    Args:
        state: Current observation dict from env.reset() or env.step()
        previous_reward: Reward received from the last action
        session_score: Cumulative score so far in this session
    
    Returns:
        A formatted string ready to be used as a system/user prompt for any LLM.
    """
    # Only expose the fields an agent needs to make a decision (not internal gold)
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


def is_duplicate(report_text: str, existing_issue_text: str) -> bool:
    """Stronger duplicate detection using Jaccard Similarity on normalized text."""
    def normalize(text):
        # lowercase and strip basic non-alphanumerics
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # ignore common short stopwords
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
    
    # Strong overlap ratio test
    return similarity > 0.20 

def run_baseline_agent():
    """
    Upgraded inference agent acting sequentially to maximize reward while
    emitting required [START]/[STEP]/[END] structured output blocks for
    OpenEnv Phase 2 validation.
    """
    env = BugTriagerEnv()
    state = env.reset()
    total_reward = 0.0
    step_count = 0

    task_name = "bug_triage"

    # ── REQUIRED: Signal start of task ────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    def take_action(action_dict):
        nonlocal state, total_reward, env, step_count
        step_count += 1
        new_state, reward, done, info = env.step(action_dict)
        total_reward += reward
        state = new_state
        # ── REQUIRED: Emit a step line per action ──────────────────────────────
        print(f"[STEP] step={step_count} reward={reward:.2f} action={action_dict.get('action','?')}", flush=True)
        return done

    # ── Triage logic ──────────────────────────────────────────────────────────
    body = state["report"]["body"].lower()
    if len(body.split()) < 30:
        take_action({"action": "request_info", "fields": ["steps_to_reproduce", "device"]})

    report_data = state["report"]
    full_text = report_data["title"] + " " + report_data["body"]
    if "steps_to_reproduce" in report_data:
        full_text += " " + report_data["steps_to_reproduce"]
    if "device" in report_data:
        full_text += " " + report_data["device"]
    full_text = full_text.lower()

    # Severity
    if "crash" in full_text or "timeout" in full_text or "504" in full_text or "401" in full_text:
        take_action({"action": "set_severity", "value": "critical"})
    elif "broken" in full_text or "500" in full_text or "unauthorized" in full_text:
        take_action({"action": "set_severity", "value": "high"})
    elif "typo" in full_text:
        take_action({"action": "set_severity", "value": "low"})
    else:
        take_action({"action": "set_severity", "value": "medium"})

    # Component
    if "payment" in full_text or "checkout" in full_text or "stripe" in full_text:
        take_action({"action": "set_component", "value": "payments"})
    elif "log in" in full_text or "login" in full_text or "password" in full_text:
        take_action({"action": "set_component", "value": "auth"})
    elif "android" in full_text or "iphone" in full_text or "mobile" in full_text or "device" in full_text:
        take_action({"action": "set_component", "value": "mobile"})
    elif "button" in full_text or "screen" in full_text or "image" in full_text or "avatar" in full_text or "ui" in full_text:
        take_action({"action": "set_component", "value": "ui"})
    else:
        take_action({"action": "set_component", "value": "backend"})

    # Duplicate detection
    for issue in state["existing_issues"]:
        issue_text = issue["title"] + " " + issue.get("body", "")
        if is_duplicate(full_text, issue_text):
            take_action({"action": "flag_duplicate", "duplicate_of": issue["id"]})
            break

    # Assign
    take_action({"action": "assign", "assignee": "alice"})

    # Submit
    take_action({"action": "submit"})

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


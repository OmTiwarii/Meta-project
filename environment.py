import json
import random
import os

class BugTriagerEnv:
    """
    Reinforcement-learning style environment for the AI Bug Report Triager.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
        self.team_roster = ["alice", "bob", "charlie", "diana"]
        self.existing_issues = [
            {"id": "issue_101", "title": "App crashes on start for Android 10", "body": "Just updated to Android 10 and app fails to open splash screen."},
            {"id": "issue_102", "title": "Stripe checkout fails", "body": "Payments returning unauthorized error on checkout."},
            {"id": "issue_103", "title": "Broken avatars on feed", "body": "Users are showing missing image icons instead of their profiles."}
        ]
        self.scenarios = []
        self._load_data()
        self.current_scenario = None
        self.current_progress = {}
        self.done = True
        
        # Step enforcement limits
        self.max_steps = 10
        self.step_count = 0
        self.action_history = []

    def _load_data(self):
        """Loads easy, medium, and hard scenarios from data directory."""
        for level in ["easy.json", "medium.json", "hard.json"]:
            path = os.path.join(self.data_dir, level)
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.scenarios.extend(json.load(f))

    def reset(self):
        """Starts a new bug report scenario."""
        if not self.scenarios:
            self._load_data()
            
        if not self.scenarios:
            raise RuntimeError("No scenarios loaded. Ensure data/*.json are present.")
            
        self.current_scenario = random.choice(self.scenarios)
        self.current_progress = {
            "severity": None,
            "component": None,
            "duplicate_of": None,
            "assignee": None,
            "extra_info": {}  # Stores dynamic context explicitly
        }
        self.done = False
        self.step_count = 0
        self.action_history = []
        return self.get_state()

    def get_state(self):
        """Returns the current observation state."""
        if self.current_scenario is None:
            return {}
            
        # Extract report data and safely inject extra dynamically gathered info
        report_data = {**self.current_scenario["report"]}
        if self.current_progress.get("extra_info"):
            report_data.update(self.current_progress["extra_info"])
            
        return {
            "report": report_data,
            "existing_issues": self.existing_issues,
            "team_roster": self.team_roster,
            "current_progress": {k: v for k, v in self.current_progress.items() if k != "extra_info"},
            "done": self.done,
            "meta": {
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "remaining_steps": self.max_steps - self.step_count
            }
        }

    def _intermediate_score(self):
        """Calculates step-wise correctness reward for incremental tracking."""
        score = 0.0
        gold = self.current_scenario["gold"]
        
        if self.current_progress.get("severity") == gold.get("severity") and gold.get("severity") is not None:
            score += 0.10
        if self.current_progress.get("component") == gold.get("component") and gold.get("component") is not None:
            score += 0.10
            
        gold_dup = gold.get("duplicate_of")
        pred_dup = self.current_progress.get("duplicate_of")
        if pred_dup:
            if pred_dup == gold_dup:
                score += 0.10
                
        if self.current_progress.get("assignee") == gold.get("assignee") and gold.get("assignee") is not None:
            score += 0.05
            
        return score

    def step(self, action: dict):
        """
        Processes agent action and updates the environment state with step limits and step-wise rewards.
        """
        if self.done:
            return self.get_state(), 0.0, self.done, {"error": "Environment is already done. Call reset()."}

        self.step_count += 1
        action_type = action.get("action")
        reward = 0.0
        info = {}

        # Action Validation: Penalize exact repeated actions
        action_str = json.dumps(action, sort_keys=True)
        if action_str in self.action_history:
            reward -= 0.02
            info["warning"] = "Repeated action penalty"
        self.action_history.append(action_str)

        old_score = self._intermediate_score()

        # Execute Actions
        valid_action = True
        if action_type == "set_severity":
            val = action.get("value")
            if val in ["critical", "high", "medium", "low"]:
                self.current_progress["severity"] = val
            else:
                valid_action = False
                info["error"] = "Invalid severity"
                
        elif action_type == "set_component":
            val = action.get("value")
            if val in ["auth", "payments", "ui", "backend", "mobile"]:
                self.current_progress["component"] = val
            else:
                valid_action = False
                info["error"] = "Invalid component"
                
        elif action_type == "flag_duplicate":
            if not action.get("duplicate_of"):
                valid_action = False
                info["error"] = "Missing duplicate_of field"
            else:
                self.current_progress["duplicate_of"] = action.get("duplicate_of")
            
        elif action_type == "assign":
            val = action.get("assignee")
            if val in self.team_roster:
                self.current_progress["assignee"] = val
            else:
                valid_action = False
                info["error"] = f"Invalid assignee. Options: {self.team_roster}"
                
        elif action_type == "request_info":
            # Upgrade: Useful Missing Context Injection
            already_requested = len(self.current_progress["extra_info"]) > 0
            if not already_requested:
                reward += 0.05  # Explicit reward for asking clarifying context
                self.current_progress["extra_info"] = {
                    "steps_to_reproduce": "User attempted to use dashboard functions without proper credentials mapped.",
                    "device": "Device Info Requested: Desktop Chrome & iPhone 13"
                }
                info["requested_info"] = "Provided missing context directly to state."
            else:
                reward -= 0.02  # Penalty for superfluous repeating info requests
                info["warning"] = "Info already requested previously."
            
        elif action_type == "submit":
            self.done = True
            from grader import grade
            final_grade = grade(self.current_progress, self.current_scenario["gold"])
            reward += final_grade
            info["message"] = "Task completed and graded successfully."
            
        else:
            valid_action = False
            info["error"] = f"Unknown action: {action_type}"

        # Penalize bad formatting or incorrect changes immediately via isolated diff scoring
        if not valid_action:
            reward -= 0.05
        else:
            new_score = self._intermediate_score()
            diff = new_score - old_score
            if diff < 0:
                reward -= 0.05 # Overwrote a correct property with an incorrect property
            elif diff > 0:
                reward += diff # Cleanly increment reward

        # Enforce maximum action limit
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            reward -= 0.20
            info["error"] = "max steps exceeded"
            
        return self.get_state(), reward, self.done, info

def duplicate_detected_correctly(predicted_dup: str, gold_dup: str) -> bool:
    """Helper to check if duplicate was correctly flagged."""
    if not gold_dup:
        return not predicted_dup
    return predicted_dup == gold_dup

def grade(predicted: dict, gold: dict, debug: bool = False):
    """
    Grades the agent's actions based on the gold standard.
    Returns a float score between 0 and 1, or optionally a dict with debug breakdown.
    
    Scoring Breakdown:
    - Severity correct -> +0.30
    - Component correct -> +0.30
    - Duplicate detection -> +0.25
    - Assignee correct -> +0.15
    """
    score = 0.0
    breakdown = {
        "severity": False,
        "component": False,
        "duplicate": False,
        "assignee": False
    }
    
    # Check severity
    if predicted.get("severity") == gold.get("severity"):
        score += 0.30
        breakdown["severity"] = True
        
    # Check component
    if predicted.get("component") == gold.get("component"):
        score += 0.30
        breakdown["component"] = True
        
    # Check duplicate detection
    if duplicate_detected_correctly(predicted.get("duplicate_of"), gold.get("duplicate_of")):
        score += 0.25
        breakdown["duplicate"] = True
        
    # Check assignee
    if predicted.get("assignee") == gold.get("assignee"):
        score += 0.15
        breakdown["assignee"] = True
        
    final_score = min(1.0, score)
    if debug:
        return {
            "score": final_score,
            "breakdown": breakdown
        }
    return final_score

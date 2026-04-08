from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import BugTriagerEnv

app = FastAPI(
    title="AI Bug Report Triager API",
    description="RL-style OpenAI Gym compatible web environment API for AI Bug Triage",
    version="1.0.0"
)

# Global Environment Instance
env = BugTriagerEnv()

class ActionRequest(BaseModel):
    action: str
    value: Optional[str] = None
    duplicate_of: Optional[str] = None
    assignee: Optional[str] = None
    fields: Optional[List[str]] = None

@app.get("/")
def root():
    """Root endpoint — welcome message and links."""
    return {
        "name": "AI Bug Report Triager API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health"
        }
    }

@app.post("/reset")
def reset_env():
    """Resets the environment and loads a new scenario."""
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step_env(action_req: ActionRequest):
    """
    Takes an action to update the environment state.
    Returns: new state, reward, done flag, and extra info.
    """
    # Compatibility with older Pydantic dict and v2 model_dump
    try:
        action_dict = action_req.model_dump(exclude_none=True)
    except AttributeError:
        action_dict = action_req.dict(exclude_none=True)
        
    state, reward, done, info = env.step(action_dict)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    """Returns the current observation of the environment."""
    return {"state": env.get_state()}

@app.get("/health")
def health_check():
    return {"status": "ok"}

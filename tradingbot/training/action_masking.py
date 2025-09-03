from typing import Dict, Any, List

ACTIONS = ["FLAT", "LONG_0.5", "LONG_1.0", "SHORT_0.5", "SHORT_1.0"]

def mask_actions(state: Dict[str, Any]) -> List[bool]:
    mask = [True] * len(ACTIONS)
    session_ok = bool(state.get("session_ok", True))
    allow_short = bool(state.get("allow_short", True))
    if not session_ok:
        for i in range(len(mask)):
            mask[i] = (ACTIONS[i] == "FLAT")
    if not allow_short:
        for i, a in enumerate(ACTIONS):
            if a.startswith("SHORT"):
                mask[i] = False
    return mask
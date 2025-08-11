from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Proposal:
    changes: List[Dict[str, Any]]
    rationale: List[str]


def clamp_positive_int(v: int, min_v: int = 1, max_v: int = 1_000_000) -> int:
    return max(min_v, min(max_v, int(v)))


def apply_action(cfg: Dict[str, Any], action: Dict[str, Any]) -> None:
    p = action.get("param")
    if p is None or p not in cfg:
        return
    if "delta" in action and isinstance(cfg[p], int):
        cfg[p] = clamp_positive_int(cfg[p] + int(action["delta"]))
    if "scale" in action and isinstance(cfg[p], (int, float)):
        cfg[p] = type(cfg[p])(cfg[p] * float(action["scale"]))


def plan_updates(cfg: Dict[str, Any], actions: List[Dict[str, Any]]) -> Proposal:
    # Filter and sort by confidence
    actions = sorted(actions, key=lambda a: a.get("confidence", 0.0), reverse=True)

    # Keep top-k modest changes to avoid destabilization
    topk = []
    seen = set()
    for a in actions:
        p = a.get("param")
        if not p or p in seen:
            continue
        seen.add(p)
        topk.append(a)
        if len(topk) >= 5:
            break

    rationale = [f"{a.get('param')}: {a.get('reason')} (conf={a.get('confidence', 0.0):.2f})" for a in topk]
    # Create a copy and apply
    new_cfg = dict(cfg)
    for a in topk:
        apply_action(new_cfg, a)

    return Proposal(changes=topk, rationale=rationale)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import math

_KEYWORDS = [
    ("chain-of-thought", ["cot", "chain of thought", "few-shot reasoning", "step-by-step"]),
    ("hierarchical", ["hierarchical", "multi-scale", "coarse-to-fine", "planner", "controller"]),
    ("ponder", ["ponder", "act", "adaptive computation", "halting", "computation time"]),
    ("world_model", ["world model", "latent dynamics", "recurrent state", "mbrl"]),
    ("attention", ["attention", "transformer", "kv cache", "heads"]),
    ("memory", ["memory", "external memory", "retrieval", "associative"]),
    ("modularity", ["modular", "mixture of experts", "moe", "router"]),
]

@dataclass
class Concept:
    name: str
    score: float
    evidence: List[str]


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def score_concepts(papers: List[Dict[str, Any]]) -> List[Concept]:
    # Simple frequency-based scoring over title+abstract
    counts: Dict[str, float] = {}
    evidences: Dict[str, List[str]] = {}
    for p in papers:
        blob = _normalize_text(p.get("title", "") + " \n " + p.get("summary", ""))
        for name, keys in _KEYWORDS:
            hit = 0
            ev_local = []
            for k in keys:
                if k in blob:
                    hit += 1
                    ev_local.append(k)
            if hit:
                counts[name] = counts.get(name, 0.0) + hit
                evidences.setdefault(name, []).append(p.get("id", "?"))
    # Log scaling to dampen popularity
    concepts = [Concept(name=k, score=math.log1p(v), evidence=evidences.get(k, [])) for k, v in counts.items()]
    concepts.sort(key=lambda c: c.score, reverse=True)
    return concepts


def map_concepts_to_actions(concepts: List[Concept]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    names = {c.name: c for c in concepts}

    # Heuristics mapping to HRM config changes
    if "hierarchical" in names:
        actions.append({
            "param": "H_layers",
            "delta": +1,
            "reason": "Recent work emphasizes deeper high-level planning modules.",
            "confidence": min(0.9, 0.4 + 0.1 * names["hierarchical"].score),
        })
        actions.append({
            "param": "L_layers",
            "delta": +1,
            "reason": "Finer low-level modeling complements deeper planners.",
            "confidence": 0.4,
        })
    if "ponder" in names:
        actions.append({
            "param": "halt_max_steps",
            "delta": +1,
            "reason": "Adaptive computation time benefits from more steps budget.",
            "confidence": min(0.9, 0.5 + 0.1 * names["ponder"].score),
        })
        actions.append({
            "param": "halt_exploration_prob",
            "scale": 1.1,
            "reason": "Encourage exploration of halting decisions early on.",
            "confidence": 0.35,
        })
    if "attention" in names:
        actions.append({
            "param": "num_heads",
            "delta": +1,
            "reason": "More heads can improve multi-facet attention reasoning.",
            "confidence": 0.3,
        })
    if "world_model" in names:
        actions.append({
            "param": "H_cycles",
            "delta": +1,
            "reason": "Iterative planning cycles mirror latent rollouts.",
            "confidence": 0.35,
        })
    if "memory" in names:
        actions.append({
            "param": "expansion",
            "scale": 1.15,
            "reason": "Larger MLP capacity to store/retrieve intermediate facts.",
            "confidence": 0.3,
        })
    if "modularity" in names:
        actions.append({
            "param": "L_cycles",
            "delta": +1,
            "reason": "More rapid sub-step iterations for specialized modules.",
            "confidence": 0.25,
        })

    return actions

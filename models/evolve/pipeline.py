from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import logging

from .literature import ArxivMiner
from .concepts import score_concepts, map_concepts_to_actions
from .introspect import model_config_dict
from .planner import plan_updates
from .apply import apply_config_diff


class SelfEvolvePipeline:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(__name__)

    def run(self, model: Any, *, query_terms=None, days_back: int = 14, max_results: int = 50, dry_run: bool = True) -> Dict[str, Any]:
        # 1) Mine papers
        miner = ArxivMiner(max_results=max_results, logger=self.log)
        papers = miner.fetch(query_terms=query_terms, days_back=days_back)
        self.log.info("Fetched %d papers from arXiv", len(papers))

        # 2) Concept scoring
        concepts = score_concepts(papers)
        # 3) Map to actions
        actions = map_concepts_to_actions(concepts)

        # 4) Introspect current config
        cfg = model_config_dict(model)

        # 5) Plan updates
        proposal = plan_updates(cfg, actions)

        report = {
            "papers": papers,
            "concepts": [c.__dict__ for c in concepts],
            "actions": actions,
            "rationale": proposal.rationale,
            "old_config": cfg,
        }

        # 6) Apply
        if not dry_run:
            new_cfg = dict(cfg)
            from .planner import apply_action
            for a in proposal.changes:
                apply_action(new_cfg, a)
            report["new_config"] = new_cfg
            # Rebuild model in caller; here we return config diff only
        return report

import json
import os
import time
import subprocess
import shutil
import datetime as dt
from types import SimpleNamespace
from typing import Dict, Any, List

# Optional resource monitoring
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# Optional torch import for checkpoint compatibility checks
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

# Import test harness functions
from scripts import test_self_evolve as tse

# Global configuration (no hardcoded literals inside logic)
RUN_CONFIG: Dict[str, Any] = {
    "ARTIFACT_ROOT": os.path.join("artifacts", "runs"),
    "RUN_ID": dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),

    # Timeboxing and schedule
    "MAX_RUNTIME_SEC": 24 * 60 * 60,
    "MAX_CYCLES": 10_000,
    "CYCLE_INTERVAL_SEC": 15 * 60,
    "CHECKPOINT_FILENAME": "model_latest.pt",

    # Training per cycle
    "TRAIN_STEPS_PER_CYCLE": 2,
    "BATCH_SIZE": 4,
    "SEQ_LEN": 32,
    "VOCAB_SIZE": 50257,
    "NUM_PUZZLE_IDS": 128,
    "UNTIL_HALT": False,

    # Self-evolution
    "APPLY_EVOLVE": True,
    "QUERY_TERMS": ["reasoning", "hierarchical", "transformer", "RL", "ACT"],
    "DAYS_BACK": 7,
    "MAX_RESULTS": 25,
    "QUERY_ROTATE_TERMS": [],  # Optional: additional terms to rotate across cycles
    "GLOBAL_SEEN_PAPERS_PATH": "",  # Optional: cross-run dedup file path

    # Code-evolution (LLM)
    "LLM_PROVIDER": "lmstudio",
    "LLM_MODEL": "openai/gpt-oss-20b",
    "LLM_HOST": "http://localhost:1234",
    "LLM_TIMEOUT": 180,
    "LLM_TEMPERATURE": 0.2,
    "LLM_MAX_TOKENS": 1200,

    # Summary tuning
    "SUMMARY_BASIC": True,
    "SUMMARY_BYTES": 4000,

    # Visualization
    "GENERATE_PLOTS": True,
    "PLOTS_FILENAME": "plots.png",

    # Resource thresholds (skip cycle if exceeded)
    "MIN_DISK_FREE_GB": 2.0,
    "MAX_MEMORY_PCT": 90.0,

    # Autopatch (apply code-evolve patches)
    "AUTO_APPLY_PATCHES": False,
    "AUTOPATCH_MAX_FILES": 10,
    "AUTOPATCH_STRICT": True,
    "SMOKETEST_CMD": "python -m scripts.test_self_evolve --no-evolve --no-train",

    # Config chaining
    "CHAIN_EVOLVED_CONFIG": True,
    # Only reuse a previous checkpoint if its saved config matches the config we will use this cycle
    "CHECKPOINT_REQUIRE_MATCH": True,
}

# Architecture keys used to determine checkpoint/config compatibility
ARCH_MATCH_KEYS: List[str] = [
    "puzzle_emb_ndim",
    "num_puzzle_identifiers",
    "vocab_size",
    "H_cycles",
    "L_cycles",
    "H_layers",
    "L_layers",
    "hidden_size",
    "expansion",
    "num_heads",
    "pos_encodings",
    "halt_max_steps",
]


def _project_root() -> str:
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _load_dotenv_if_present() -> None:
    # Try python-dotenv first
    env_loaded = False
    try:
        from dotenv import load_dotenv  # type: ignore
        # Prefer project root .env, then CWD
        pr = _project_root()
        env_path = os.path.join(pr, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            env_loaded = True
        else:
            load_dotenv(override=False)
            env_loaded = True
    except Exception:
        env_loaded = False

    if env_loaded:
        return
    # Fallback: very small .env parser at project root
    try:
        pr = _project_root()
        env_path = os.path.join(pr, ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        pass


def _run_dir(cfg: Dict[str, Any]) -> str:
    return os.path.join(cfg["ARTIFACT_ROOT"], cfg["RUN_ID"])  # artifacts/runs/<run_id>


def _cycle_dir(cfg: Dict[str, Any], cycle: int) -> str:
    return os.path.join(_run_dir(cfg), f"cycle_{cycle:05d}")


def _ensure_dirs(cfg: Dict[str, Any], cycle: int) -> Dict[str, str]:
    rdir = _run_dir(cfg)
    cdir = _cycle_dir(cfg, cycle)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    return {
        "run_dir": rdir,
        "cycle_dir": cdir,
        "metrics_path": os.path.join(rdir, "metrics.jsonl"),
        "metrics_csv_path": os.path.join(rdir, "metrics.csv"),
        "state_path": os.path.join(rdir, "run_state.json"),
        "seen_papers_path": os.path.join(rdir, "seen_papers.jsonl"),
        "report_path": os.path.join(cdir, "evolve_report.json"),
        "proposal_path": os.path.join(cdir, "code_evolve_proposal.json"),
        "config_export_path": os.path.join(cdir, "hrm_evolved_config.json"),
        "train_log_path": os.path.join(cdir, "train.json"),
        "backup_dir": os.path.join(cdir, "autopatch_backup"),
        "patches_dir": os.path.join(cdir, "autopatch_patches"),
    }


def _resource_ok(cfg: Dict[str, Any]) -> bool:
    try:
        # Disk
        stat = os.statvfs(_run_dir(cfg)) if hasattr(os, "statvfs") else None
    except Exception:
        stat = None
    free_gb = None
    if stat:
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    # On Windows, fallback to psutil if available
    if free_gb is None and psutil is not None:
        free_gb = psutil.disk_usage(_run_dir(cfg)).free / (1024 ** 3)

    mem_ok = True
    if psutil is not None:
        mem_ok = psutil.virtual_memory().percent < cfg["MAX_MEMORY_PCT"]
    disk_ok = True if free_gb is None else (free_gb >= cfg["MIN_DISK_FREE_GB"])
    return mem_ok and disk_ok


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _append_csv(path: str, rec: Dict[str, Any]) -> None:
    import csv
    exists = os.path.exists(path)
    # Flatten a few fields for CSV
    row = {
        "cycle": rec.get("cycle"),
        "ts": rec.get("ts"),
        "steps": rec.get("train", {}).get("steps"),
        "loss": rec.get("train", {}).get("loss"),
        "papers": rec.get("papers"),
        "new_papers_logged": rec.get("new_papers_logged"),
        "actions": rec.get("actions"),
        "proposal_ok": rec.get("proposal_ok"),
        # Autopatch fields to align with JSONL
        "autopatch_attempted": rec.get("autopatch_attempted"),
        "autopatch_applied": rec.get("autopatch_applied"),
        "autopatch_smoke_ok": rec.get("autopatch_smoke_ok"),
        "autopatch_rolled_back": rec.get("autopatch_rolled_back"),
        "autopatch_error": rec.get("autopatch_error"),
    }
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _load_seen_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and "id" in obj:
                        ids.add(obj["id"])
                except Exception:
                    continue
    return ids


def _update_seen_ids(path: str, papers: Any) -> int:
    new = 0
    with open(path, "a", encoding="utf-8") as f:
        for p in papers or []:
            pid = p.get("id")
            if pid:
                f.write(json.dumps({"id": pid}) + "\n")
                new += 1
    return new


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


def _rotated_query_terms(cfg: Dict[str, Any], cycle: int) -> List[str]:
    base = list(cfg.get("QUERY_TERMS", []) or [])
    extra = list(cfg.get("QUERY_ROTATE_TERMS", []) or [])
    if not extra:
        # Ensure unique, stable order
        seen = set()
        out: List[str] = []
        for t in base:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    idx = cycle % len(extra)
    pick = extra[idx]
    out: List[str] = []
    seen = set()
    for t in base + [pick]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _apply_patches(cfg: Dict[str, Any], proposal: Dict[str, Any], paths: Dict[str, str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "attempted": False,
        "applied": False,
        "smoke_ok": False,
        "rolled_back": False,
        "error": "",
    }
    if not cfg.get("AUTO_APPLY_PATCHES", False):
        return res

    patches = proposal.get("patches", []) or []
    if not patches:
        return res

    res["attempted"] = True
    max_files = int(cfg.get("AUTOPATCH_MAX_FILES", 10))
    strict = bool(cfg.get("AUTOPATCH_STRICT", True))

    os.makedirs(paths["patches_dir"], exist_ok=True)

    written: List[str] = []
    for i, p in enumerate(patches[:max_files]):
        # Accept either dict with 'diff' or raw string
        if isinstance(p, dict):
            diff_txt = p.get("diff") or p.get("patch") or ""
        else:
            diff_txt = str(p)
        if not diff_txt.strip():
            continue
        pth = os.path.join(paths["patches_dir"], f"patch_{i:03d}.diff")
        with open(pth, "w", encoding="utf-8") as f:
            f.write(diff_txt)
        written.append(pth)

    if not written:
        return res

    def _git(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=_project_root(), capture_output=True, text=True)

    # Try to apply all patches
    for pth in written:
        cmd = ["git", "apply", pth]
        r = _git(cmd)
        if r.returncode != 0 and not strict:
            # Retry with 3-way merge if non-strict
            r = _git(["git", "apply", "--3way", pth])
        if r.returncode != 0:
            res["error"] = (r.stderr or r.stdout or "git apply failed").strip()
            # Roll back any partial application
            _git(["git", "reset", "--hard", "HEAD"])
            res["rolled_back"] = True
            return res

    res["applied"] = True

    # Smoke test
    smoke_cmd = str(cfg.get("SMOKETEST_CMD", "")).strip()
    if smoke_cmd:
        # Run as shell command to allow flags in string
        smoke = subprocess.run(smoke_cmd, cwd=_project_root(), shell=True)
        if smoke.returncode != 0:
            res["error"] = f"smoke test failed: {smoke.returncode}"
            _git(["git", "reset", "--hard", "HEAD"])
            res["rolled_back"] = True
            return res
        res["smoke_ok"] = True

    return res


def run_cycle(cfg: Dict[str, Any], cycle: int) -> Dict[str, Any]:
    paths = _ensure_dirs(cfg, cycle)

    # Basic resource guard
    if not _resource_ok(cfg):
        return {"status": "skipped_resource", "cycle": cycle, "ts": _now_iso()}

    # Ensure literature miner can filter already-seen papers
    # Prefer global dedup file if provided, else per-run file
    global_seen = str(cfg.get("GLOBAL_SEEN_PAPERS_PATH", "")).strip()
    os.environ["HRM_SEEN_PAPERS_PATH"] = global_seen or paths["seen_papers_path"]

    # Determine config chaining input for this cycle
    chain_enabled = bool(cfg.get("CHAIN_EVOLVED_CONFIG", True))
    prev_cfg_path = _prev_evolved_config_path(cfg, cycle) if chain_enabled else ""
    use_cfg_path = prev_cfg_path if prev_cfg_path else ""
    chained_config_used = bool(use_cfg_path)

    # Train short burst (skip entirely if steps<=0, but emit stub)
    checkpoint_path = os.path.join(_run_dir(cfg), cfg["CHECKPOINT_FILENAME"])
    # Check checkpoint compatibility with the config we will use this cycle
    load_ckpt_path = checkpoint_path
    if chained_config_used and bool(cfg.get("CHECKPOINT_REQUIRE_MATCH", True)):
        if not _checkpoint_cfg_compatible(checkpoint_path, use_cfg_path, ARCH_MATCH_KEYS):
            load_ckpt_path = ""

    if int(cfg.get("TRAIN_STEPS_PER_CYCLE", 0)) > 0:
        args_train = SimpleNamespace(
            use_config=use_cfg_path,
            batch_size=cfg["BATCH_SIZE"],
            seq_len=cfg["SEQ_LEN"],
            vocab_size=cfg["VOCAB_SIZE"],
            num_puzzle_ids=cfg["NUM_PUZZLE_IDS"],
            until_halt=cfg["UNTIL_HALT"],
            act_steps=cfg["TRAIN_STEPS_PER_CYCLE"],
            load_checkpoint_path=load_ckpt_path,
            save_checkpoint_path=checkpoint_path,
        )
        t_steps, t_loss, t_metrics, t_all_halted = tse.run_forward_backward(args_train)
    else:
        t_steps, t_loss, t_metrics, t_all_halted = 0, None, {}, False
    with open(paths["train_log_path"], "w", encoding="utf-8") as f:
        json.dump({
            "cycle": cycle,
            "steps": t_steps,
            "loss": t_loss,
            "metrics": t_metrics,
            "all_halted": t_all_halted,
            "ts": _now_iso(),
        }, f, indent=2)

    # Self evolve
    args_evolve = SimpleNamespace(
        use_config=use_cfg_path,
        batch_size=cfg["BATCH_SIZE"],
        seq_len=cfg["SEQ_LEN"],
        vocab_size=cfg["VOCAB_SIZE"],
        num_puzzle_ids=cfg["NUM_PUZZLE_IDS"],
        apply_evolve=cfg["APPLY_EVOLVE"],
        save_report=paths["report_path"],
        export_config=paths["config_export_path"],
        query_terms=_rotated_query_terms(cfg, cycle),
        days_back=cfg["DAYS_BACK"],
        max_results=cfg["MAX_RESULTS"],
    )
    report = tse.run_self_evolve(args_evolve)

    # Update seen papers list (best-effort)
    seen_before = _load_seen_ids(paths["seen_papers_path"]) if os.path.exists(paths["seen_papers_path"]) else set()
    papers = (report or {}).get("papers", [])
    new_seen_count = _update_seen_ids(paths["seen_papers_path"], papers)

    # Code evolve (using the report)
    args_code = SimpleNamespace(
        save_proposal=paths["proposal_path"],
        use_report=paths["report_path"],
        llm_provider=cfg["LLM_PROVIDER"],
        llm_model=cfg["LLM_MODEL"],
        llm_host=cfg["LLM_HOST"],
        llm_timeout=cfg["LLM_TIMEOUT"],
        llm_temperature=cfg["LLM_TEMPERATURE"],
        llm_max_tokens=cfg["LLM_MAX_TOKENS"],
        summary_basic=cfg["SUMMARY_BASIC"],
        summary_bytes=cfg["SUMMARY_BYTES"],
    )
    proposal = tse.run_code_evolve(args_code, report)
    proposal_is_dict = isinstance(proposal, dict)
    proposal_auto = bool(proposal.get("auto_synthesized", False)) if proposal_is_dict else False
    patches_n = len(proposal.get("patches", [])) if proposal_is_dict else 0

    # Autopatch
    autopatch = _apply_patches(cfg, proposal if proposal_is_dict else {}, paths)

    # Metrics aggregation
    rec = {
        "cycle": cycle,
        "ts": _now_iso(),
        "train": {"steps": t_steps, "loss": t_loss, **t_metrics},
        "papers": len(papers),
        "new_papers_logged": new_seen_count,
        "actions": len((report or {}).get("actions", [])),
        "proposal_ok": bool(proposal),
        "proposal_source": ("synthesized" if proposal_auto else ("llm" if proposal else "none")),
        "patches": patches_n,
        "autopatch_attempted": autopatch.get("attempted", False),
        "autopatch_applied": autopatch.get("applied", False),
        "autopatch_smoke_ok": autopatch.get("smoke_ok", False),
        "autopatch_rolled_back": autopatch.get("rolled_back", False),
        "autopatch_error": autopatch.get("error", ""),
        "chained_config_used": chained_config_used,
        "chained_config_path": use_cfg_path,
        "checkpoint_used": bool(load_ckpt_path and os.path.exists(load_ckpt_path)),
    }
    _append_jsonl(paths["metrics_path"], rec)
    _append_csv(paths["metrics_csv_path"], rec)
    return {"status": "ok", **rec}


def main() -> None:
    # Load .env before reading overlay
    _load_dotenv_if_present()
    cfg = dict(RUN_CONFIG)

    # Overlay from JSON config and environment variables
    cfg = _overlay_env(cfg)

    # Create run dir and save command/config snapshot
    paths = _ensure_dirs(cfg, 0)
    with open(os.path.join(paths["run_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    start = time.time()
    done_cycles = 0

    # Resume support
    state_path = paths["state_path"]
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
                done_cycles = int(st.get("last_cycle", 0)) + 1
        except Exception:
            done_cycles = 0

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= cfg["MAX_RUNTIME_SEC"]:
            break
        if done_cycles >= cfg["MAX_CYCLES"]:
            break

        try:
            res = run_cycle(cfg, done_cycles)
        except Exception as e:  # Early stop on unexpected error
            res = {"status": "error", "error": repr(e), "cycle": done_cycles, "ts": _now_iso()}
            # Persist state and break
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_cycle": done_cycles,
                    "last_status": res.get("status"),
                    "error": res.get("error"),
                    "ts": _now_iso(),
                }, f, indent=2)
            break
        # Persist state
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_cycle": done_cycles,
                "last_status": res.get("status"),
                "ts": _now_iso(),
            }, f, indent=2)

        done_cycles += 1
        # Sleep between cycles
        time.sleep(cfg["CYCLE_INTERVAL_SEC"])

    # Final summary
    with open(os.path.join(paths["run_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_id": cfg["RUN_ID"],
            "cycles": done_cycles,
            "elapsed_sec": int(time.time() - start),
            "ended": _now_iso(),
        }, f, indent=2)

    # Optional visualization (loss vs cycle)
    if cfg.get("GENERATE_PLOTS", False):
        try:
            import csv
            import matplotlib.pyplot as plt  # type: ignore
            xs, ys = [], []
            with open(paths["metrics_csv_path"], "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        xs.append(int(row.get("cycle", "0")))
                        ys.append(float(row.get("loss", "nan")))
                    except Exception:
                        continue
            if xs and ys:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(xs, ys, marker='o', linewidth=1)
                ax.set_title("Training loss per cycle")
                ax.set_xlabel("cycle")
                ax.set_ylabel("loss")
                fig.tight_layout()
                out_png = os.path.join(paths["run_dir"], cfg.get("PLOTS_FILENAME", "plots.png"))
                fig.savefig(out_png)
                plt.close(fig)
        except Exception:
            pass


# ---- Environment overlay helpers ----
def _coerce_type(raw: str, prev_value: Any) -> Any:
    t = type(prev_value)
    try:
        if t is bool:
            return raw.lower() in ("1", "true", "yes", "y", "on")
        if t is int:
            return int(raw)
        if t is float:
            return float(raw)
        if t is list:
            # Try JSON first, else comma-split to strings
            try:
                val = json.loads(raw)
                return val if isinstance(val, list) else [val]
            except Exception:
                return [s for s in (x.strip() for x in raw.split(",")) if s]
        if t is dict:
            try:
                val = json.loads(raw)
                return val if isinstance(val, dict) else prev_value
            except Exception:
                return prev_value
        # default: string
        return raw
    except Exception:
        return raw


def _overlay_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    # Load JSON config if provided
    json_path = os.environ.get("HRM_RUN_CONFIG_JSON", "").strip()
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                if isinstance(user_cfg, dict):
                    out.update(user_cfg)
        except Exception:
            pass
    # Overlay simple env vars: HRM_<KEY>
    for k, v in list(out.items()):
        env_key = f"HRM_{k}"
        if env_key in os.environ:
            raw = os.environ[env_key]
            out[k] = _coerce_type(raw, v)
    return out
 
 
if __name__ == "__main__":
    main()

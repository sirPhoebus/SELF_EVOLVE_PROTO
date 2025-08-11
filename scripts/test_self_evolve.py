import argparse
import json
import re
import os
import sys
from typing import List, Optional

import torch

# Local imports
from hrm import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from models.evolve.code_summary import summarize_files, render_summary_text, ROOT_FILES_DEFAULT
from models.evolve.llm_client import LLMClient


# Global knobs for code-evolution behavior (env-overridable; no magic literals in logic)
CODE_EVOLVE_MAX_RETRIES: int = int(os.environ.get("HRM_CODE_EVOLVE_MAX_RETRIES", "2"))
CODE_EVOLVE_SILENT: bool = os.environ.get("HRM_CODE_EVOLVE_SILENT", "true").lower() in ("1", "true", "yes", "on")


# Global knobs for synthetic data generation via LLM (env-overridable)
SYNTHETIC_DATA_ENABLE: bool = os.environ.get("HRM_SYNTHETIC_DATA_ENABLE", "false").lower() in ("1", "true", "yes", "on")
SYNTHETIC_DATA_PROVIDER: str = os.environ.get("HRM_SYNTHETIC_DATA_PROVIDER", os.environ.get("HRM_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio")))
SYNTHETIC_DATA_MODEL: str = os.environ.get("HRM_SYNTHETIC_DATA_MODEL", os.environ.get("HRM_LLM_MODEL", os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")))
SYNTHETIC_DATA_HOST: str = os.environ.get("HRM_SYNTHETIC_DATA_HOST", os.environ.get("HRM_LLM_HOST", os.environ.get("LLM_HOST", "http://localhost:1234")))
SYNTHETIC_DATA_TIMEOUT: int = int(os.environ.get("HRM_SYNTHETIC_DATA_TIMEOUT", os.environ.get("HRM_LLM_TIMEOUT", os.environ.get("LLM_TIMEOUT", "180"))))
SYNTHETIC_DATA_TEMPERATURE: float = float(os.environ.get("HRM_SYNTHETIC_DATA_TEMPERATURE", os.environ.get("HRM_LLM_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.2"))))
SYNTHETIC_DATA_MAX_TOKENS: int = int(os.environ.get("HRM_SYNTHETIC_DATA_MAX_TOKENS", os.environ.get("HRM_LLM_MAX_TOKENS", os.environ.get("LLM_MAX_TOKENS", "512"))))

# Prompt knobs
SYNTHETIC_DATA_SYSTEM: str = os.environ.get(
    "HRM_SYNTHETIC_DATA_SYSTEM",
    "You generate strictly valid JSON containing integer token sequences suitable for toy model training."
)
SYNTHETIC_DATA_PROMPT_TMPL: str = os.environ.get(
    "HRM_SYNTHETIC_DATA_PROMPT",
    (
        "Produce only a single JSON object with keys: inputs, puzzle_identifiers, labels.\n"
        "- inputs: a 2D array of shape [B, T] with integers in [0, VOCAB-1].\n"
        "- puzzle_identifiers: length-B array with integers in [0, NUM_PUZZLE_IDS-1].\n"
        "- labels: a 2D array of shape [B, T] with integers in [0, VOCAB-1] or -100 for ignore.\n"
        "Use mildly structured patterns (e.g., small runs, repetitions) so the task is learnable.\n"
        "Parameters: B={B}, T={T}, VOCAB={VOCAB}, NUM_PUZZLE_IDS={NUM_PUZZLE_IDS}."
    )
)


def _llm_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, num_puzzle_identifiers: int, ignore_frac: float = 0.2):
    # Read env at call-time so .env loaded later is respected
    provider = os.environ.get("HRM_SYNTHETIC_DATA_PROVIDER", os.environ.get("HRM_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio")))
    model = os.environ.get("HRM_SYNTHETIC_DATA_MODEL", os.environ.get("HRM_LLM_MODEL", os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")))
    host = os.environ.get("HRM_SYNTHETIC_DATA_HOST", os.environ.get("HRM_LLM_HOST", os.environ.get("LLM_HOST", "http://localhost:1234")))
    timeout_s = int(os.environ.get("HRM_SYNTHETIC_DATA_TIMEOUT", os.environ.get("HRM_LLM_TIMEOUT", os.environ.get("LLM_TIMEOUT", "180"))))
    temperature = float(os.environ.get("HRM_SYNTHETIC_DATA_TEMPERATURE", os.environ.get("HRM_LLM_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.2"))))
    max_tokens = int(os.environ.get("HRM_SYNTHETIC_DATA_MAX_TOKENS", os.environ.get("HRM_LLM_MAX_TOKENS", os.environ.get("LLM_MAX_TOKENS", "512"))))
    system_prompt = os.environ.get(
        "HRM_SYNTHETIC_DATA_SYSTEM",
        "You generate strictly valid JSON containing integer token sequences suitable for toy model training."
    )
    user_prompt_tmpl = os.environ.get(
        "HRM_SYNTHETIC_DATA_PROMPT",
        (
            "Produce only a single JSON object with keys: inputs, puzzle_identifiers, labels.\n"
            "- inputs: a 2D array of shape [B, T] with integers in [0, VOCAB-1].\n"
            "- puzzle_identifiers: length-B array with integers in [0, NUM_PUZZLE_IDS-1].\n"
            "- labels: a 2D array of shape [B, T] with integers in [0, VOCAB-1] or -100 for ignore.\n"
            "Use mildly structured patterns (e.g., small runs, repetitions) so the task is learnable.\n"
            "Parameters: B={B}, T={T}, VOCAB={VOCAB}, NUM_PUZZLE_IDS={NUM_PUZZLE_IDS}."
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = LLMClient(
        provider=provider,
        model=model,
        host=host,
        timeout_s=timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prompt = user_prompt_tmpl.format(B=batch_size, T=seq_len, VOCAB=vocab_size, NUM_PUZZLE_IDS=num_puzzle_identifiers)
    try:
        print("[TRAIN] Requesting synthetic batch from LLM (provider=%s, model=%s)" % (provider, model))
        txt = client.complete(prompt, system=system_prompt or None)
        # Extract robustly if model wraps JSON
        json_str = _extract_json_from_text(txt) or txt
        obj = json.loads(json_str)
        # Validate and coerce shapes
        def _as_int_clamped(x, lo, hi):
            try:
                v = int(x)
            except Exception:
                v = lo
            return max(lo, min(hi, v))

        raw_inputs = obj.get("inputs", [])
        raw_labels = obj.get("labels", [])
        raw_pids = obj.get("puzzle_identifiers", [])

        # Truncate/pad to batch_size, seq_len
        def _fix_2d(arr2d, default_val=0):
            arr = list(arr2d) if isinstance(arr2d, list) else []
            arr = arr[:batch_size] + [[default_val] * seq_len for _ in range(max(0, batch_size - len(arr)))]
            out = []
            for row in arr:
                r = list(row) if isinstance(row, list) else []
                r = r[:seq_len] + [default_val] * max(0, seq_len - len(r))
                out.append([_as_int_clamped(v, 0, vocab_size - 1) for v in r])
            return out

        def _fix_1d(arr1d, default_val=0):
            arr = list(arr1d) if isinstance(arr1d, list) else []
            arr = arr[:batch_size] + [default_val] * max(0, batch_size - len(arr))
            return [_as_int_clamped(v, 0, num_puzzle_identifiers - 1) for v in arr]

        inputs_2d = _fix_2d(raw_inputs, default_val=0)
        labels_2d = _fix_2d(raw_labels, default_val=0)
        # Allow -100 ignore id in labels
        labels_2d = [[v if isinstance(v, int) and (v == IGNORE_LABEL_ID or (0 <= v < vocab_size)) else 0 for v in row] for row in labels_2d]
        pids_1d = _fix_1d(raw_pids, default_val=0)

        inputs = torch.tensor(inputs_2d, dtype=torch.int32, device=device)
        labels = torch.tensor(labels_2d, dtype=torch.int64, device=device)
        puzzle_identifiers = torch.tensor(pids_1d, dtype=torch.int32, device=device)

        # Optional: randomly add ignore labels if none present and ignore_frac>0
        if ignore_frac > 0 and not ((labels == IGNORE_LABEL_ID).any().item()):
            mask = torch.rand(batch_size, seq_len, device=device) < ignore_frac
            labels = torch.where(mask, torch.full_like(labels, IGNORE_LABEL_ID), labels)

        return {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers,
            "labels": labels,
        }
    except Exception as e:
        print("[TRAIN] Synthetic batch generation failed; falling back to random. Reason:", repr(e))
        # Fallback to random batch on any failure
        pass

    # Fallback: random dummy
    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int32, device=device)
    puzzle_identifiers = torch.randint(low=0, high=num_puzzle_identifiers, size=(batch_size,), dtype=torch.int32, device=device)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int64, device=device)
    if ignore_frac > 0:
        mask = torch.rand(batch_size, seq_len, device=device) < ignore_frac
        labels = torch.where(mask, torch.full_like(labels, IGNORE_LABEL_ID), labels)
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": labels,
    }


def make_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, num_puzzle_identifiers: int, ignore_frac: float = 0.2):
    enabled = os.environ.get("HRM_SYNTHETIC_DATA_ENABLE", "false").lower() in ("1", "true", "yes", "on")
    if enabled:
        return _llm_synthetic_batch(batch_size, seq_len, vocab_size, num_puzzle_identifiers, ignore_frac=ignore_frac)
    # Default: random dummy batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int32, device=device)
    puzzle_identifiers = torch.randint(low=0, high=num_puzzle_identifiers, size=(batch_size,), dtype=torch.int32, device=device)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int64, device=device)
    if ignore_frac > 0:
        mask = torch.rand(batch_size, seq_len, device=device) < ignore_frac
        labels = torch.where(mask, torch.full_like(labels, IGNORE_LABEL_ID), labels)
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": labels,
    }


def build_small_config(batch_size: int, seq_len: int, vocab_size: int, num_puzzle_identifiers: int):
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": 0,  # keep simple for test; can set >0 to exercise sparse embeddings
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "vocab_size": vocab_size,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,
        # Use float32 for CPU compatibility in tests
        "forward_dtype": "float32",
    }


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extract the largest balanced JSON object from arbitrary text.

    Handles cases where the model wraps JSON in prose or fenced code blocks.
    Returns the JSON substring if found, otherwise None.
    """
    if not text:
        return None
    # Remove code fences like ```json ... ``` to avoid confusing brace scan
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    txt = "\n".join(lines)

    start_idx = -1
    depth = 0
    best = None
    best_len = 0
    for i, ch in enumerate(txt):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    cand = txt[start_idx : i + 1]
                    if len(cand) > best_len:
                        best = cand
                        best_len = len(cand)
    if best:
        return best
    # Fallback: try to parse the text after removing fences
    s = txt.strip()
    if not s:
        return None
    try:
        json.loads(s)
        return s
    except Exception:
        return None


def run_forward_backward(args):
    torch.manual_seed(42)

    if args.use_config:
        cfg = _load_json(args.use_config)
    else:
        cfg = build_small_config(args.batch_size, args.seq_len, args.vocab_size, args.num_puzzle_ids)
    model = HierarchicalReasoningModel_ACTV1(cfg)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optional: load checkpoint
    load_ckpt = getattr(args, "load_checkpoint_path", None)
    if load_ckpt and isinstance(load_ckpt, str) and os.path.exists(load_ckpt):
        try:
            state = torch.load(load_ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
                print(f"[TRAIN] Loaded checkpoint from {load_ckpt}")
        except Exception as e:
            print("[TRAIN] Failed to load checkpoint:", repr(e))

    batch = make_dummy_batch(args.batch_size, args.seq_len, args.vocab_size, args.num_puzzle_ids, ignore_frac=0.2)

    # Loss head wraps ACT loop and returns loss/metrics
    loss_head = ACTLossHead(model, loss_type="softmax_cross_entropy")
    carry = loss_head.initial_carry(batch=batch)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    steps_done = 0
    all_halted = torch.tensor(False)
    while True:
        opt.zero_grad(set_to_none=True)
        carry, loss, metrics, outputs, all_halted = loss_head(return_keys=["logits", "q_halt_logits"], carry=carry, batch=batch)
        loss.backward()
        opt.step()
        steps_done += 1
        if args.until_halt and bool(all_halted.item()):
            break
        if not args.until_halt and steps_done >= max(1, int(args.act_steps)):
            break

    # Print concise metrics summary
    m = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
    print("[TRAIN] steps=%d, loss=%.4f, metrics=%s, all_halted=%s" % (steps_done, loss.item(), json.dumps(m), bool(all_halted.item())))
    # Optional: save checkpoint
    save_ckpt = getattr(args, "save_checkpoint_path", None)
    if save_ckpt and isinstance(save_ckpt, str):
        d = os.path.dirname(save_ckpt)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        try:
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, save_ckpt)
            print(f"[TRAIN] Saved checkpoint to {save_ckpt}")
        except Exception as e:
            print("[TRAIN] Failed to save checkpoint:", repr(e))
    return steps_done, float(loss.item()), m, bool(all_halted.item())


def _config_diff(old_cfg: dict, new_cfg: dict):
    diffs = []
    keys = sorted(set(list(old_cfg.keys()) + list(new_cfg.keys())))
    for k in keys:
        ov = old_cfg.get(k, None)
        nv = new_cfg.get(k, None)
        if ov != nv:
            diffs.append((k, ov, nv))
    return diffs


def _save_report(path: str, report: dict):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[EVOLVE] Report saved to {path}")


def run_self_evolve(args):
    if args.use_config:
        cfg = _load_json(args.use_config)
    else:
        cfg = build_small_config(args.batch_size, args.seq_len, args.vocab_size, args.num_puzzle_ids)
    model = HierarchicalReasoningModel_ACTV1(cfg)
    model.eval()

    try:
        report = model.self_evolve(
            query_terms=args.query_terms if args.query_terms else None,
            days_back=args.days_back,
            max_results=args.max_results,
            dry_run=not args.apply_evolve,
        )
    except Exception as e:
        print("[EVOLVE] Failed to fetch/process arXiv content:", repr(e))
        return None

    # Print a compact summary
    print("[EVOLVE] papers=%d" % len(report.get("papers", [])))
    print("[EVOLVE] rationale:\n%s" % report.get("rationale", "<none>"))
    actions = report.get("actions", [])
    print("[EVOLVE] actions (top %d):" % min(10, len(actions)))
    for a in actions[:10]:
        print(" -", a)

    if args.apply_evolve and "new_config" in report:
        old_cfg = report.get("old_config", {})
        new_cfg = report["new_config"]
        diffs = _config_diff(old_cfg, new_cfg)
        if diffs:
            print("[EVOLVE] Applied config diff:")
            for k, ov, nv in diffs:
                print(f" - {k}: {ov} -> {nv}")
        else:
            print("[EVOLVE] No differences between old and new config.")
        # Optionally instantiate the evolved model to validate construction
        _ = HierarchicalReasoningModel_ACTV1(new_cfg)
        print("[EVOLVE] Evolved model instantiated successfully.")
        if args.export_config:
            d = os.path.dirname(args.export_config)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            with open(args.export_config, "w", encoding="utf-8") as f:
                json.dump(new_cfg, f, indent=2)
            print(f"[EVOLVE] Exported new_config to {args.export_config}")
    elif args.export_config:
        print("[EVOLVE] --export-config requested but no new_config available. Use --apply-evolve to generate it.")

    if args.save_report:
        _save_report(args.save_report, report)
    return report


def run_code_evolve(args, report: Optional[dict]):
    # Load research report if not provided
    if report is None:
        if not args.use_report:
            print("[CODE-EVOLVE] No report available. Provide --use-report or run self-evolve first in this session.")
            return None
        try:
            report = _load_json(args.use_report)
        except Exception as e:
            print("[CODE-EVOLVE] Failed to load report:", repr(e))
            return None

    # Summarize codebase
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    include = None
    if args.summary_basic:
        include = [
            "hrm.py",
            os.path.join("models", "layers.py"),
            os.path.join("models", "evolve", "pipeline.py"),
        ]
    summary = summarize_files(project_root, include=include, max_bytes_per_file=int(args.summary_bytes))
    summary_text = render_summary_text(summary)

    # Compose prompt
    sys_prompt = (
        "You are an expert ML systems engineer.\n"
        "Given the current HRM codebase summary and a research/evolution report,\n"
        "propose safe, incremental code changes to improve reasoning and self-evolution.\n"
        "Output STRICT JSON with fields: title, high_level_changes (list of strings),\n"
        "patches (list of {file, explanation, diff}), and optional config_changes (list).\n"
        "Diffs should be contextual unified diffs (not applied here)."
    )
    user_prompt = (
        "[RESEARCH_REPORT_JSON]\n" + json.dumps(report, indent=2) + "\n\n" +
        "[CODEBASE_SUMMARY]\n" + summary_text + "\n\n" +
        "Propose the JSON as specified. Ensure validity (no trailing commas)."
    )

    client = LLMClient(
        provider=(args.llm_provider or "ollama"),
        model=(args.llm_model or "llama3.1:8b"),
        host=(args.llm_host or None),
        timeout_s=int(args.llm_timeout),
        temperature=float(args.llm_temperature),
        max_tokens=int(args.llm_max_tokens),
    )

    # Directory for saving artifacts
    save_dir = os.path.dirname(args.save_proposal) if args.save_proposal else ""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Try initial request + up to CODE_EVOLVE_MAX_RETRIES repair attempts
    proposal = None
    last_raw = ""
    for attempt in range(CODE_EVOLVE_MAX_RETRIES + 1):
        try:
            if attempt == 0:
                raw = client.complete(prompt=user_prompt, system=sys_prompt)
            else:
                repair_sys = (
                    sys_prompt
                    + "\nReturn STRICT JSON only. Do not include backticks or prose."
                    + " Fix your previous response to be valid JSON with required keys."
                )
                repair_prompt = (
                    "The previous response failed JSON parsing. Here is your last output:\n"
                    + (last_raw[:8000] if last_raw else "<empty>")
                    + "\n\nRe-emit the corrected JSON only."
                )
                raw = client.complete(prompt=repair_prompt, system=repair_sys)
        except Exception as e:
            if not CODE_EVOLVE_SILENT:
                print("[CODE-EVOLVE] LLM request error; retrying...", repr(e))
            raw = ""

        last_raw = (raw or "").strip()
        if not last_raw:
            # try next attempt
            continue

        extracted = _extract_json_from_text(last_raw)
        try:
            candidate = json.loads(extracted if extracted is not None else last_raw)
            # Basic schema presence
            if isinstance(candidate, dict) and "title" in candidate and "patches" in candidate:
                proposal = candidate
                break
        except Exception:
            # try next attempt
            continue

    # Final fallback: synthesize a minimal valid proposal from the report (ensures evolution proceeds)
    if proposal is None:
        hl_changes = []
        if isinstance(report, dict):
            for act in report.get("actions", [])[:10]:
                if isinstance(act, dict):
                    p = act.get("param")
                    if "delta" in act:
                        hl_changes.append(f"Increment {p} by {act.get('delta')}")
                    elif "scale" in act:
                        hl_changes.append(f"Scale {p} by {act.get('scale')}")
        proposal = {
            "title": "Synthesized code-evolution from self-evolution actions",
            "high_level_changes": hl_changes,
            "patches": [],  # No code patches when synthesized
            "config_changes": report.get("actions", []) if isinstance(report, dict) else [],
            "auto_synthesized": True,
        }

    # Persist JSON proposal (always)
    if args.save_proposal:
        with open(args.save_proposal, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
    # Optionally persist raw text quietly for debugging
    if save_dir and last_raw:
        try:
            with open(os.path.join(save_dir, "code_evolve_raw.txt"), "w", encoding="utf-8") as f:
                f.write(last_raw)
        except Exception:
            pass

    # Print a concise summary
    title = proposal.get("title", "<no title>")
    changes = proposal.get("high_level_changes", [])
    patches = proposal.get("patches", [])
    print(f"[CODE-EVOLVE] Title: {title}")
    print("[CODE-EVOLVE] High-level changes:")
    for c in changes[:10]:
        print(" -", c)
    print(f"[CODE-EVOLVE] Patch suggestions: {len(patches)} file(s)")

    if args.save_proposal:
        d = os.path.dirname(args.save_proposal)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(args.save_proposal, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
        print(f"[CODE-EVOLVE] Proposal saved to {args.save_proposal}")
    return proposal


def main():
    parser = argparse.ArgumentParser(description="Minimal HRM self-evolve and training test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--num-puzzle-ids", type=int, default=128)

    parser.add_argument("--no-train", action="store_true", help="Skip forward/backward test")
    parser.add_argument("--no-evolve", action="store_true", help="Skip self-evolve step")

    parser.add_argument("--apply-evolve", action="store_true", help="Apply proposed config (dry_run=False) and print diff")
    parser.add_argument("--save-report", type=str, default="", help="Path to write full evolve report JSON (optional)")
    parser.add_argument("--export-config", type=str, default="", help="Export new_config to JSON (requires --apply-evolve)")
    parser.add_argument("--use-config", type=str, default="", help="Use a JSON config for training/evolution instead of the built-in test config")
    parser.add_argument("--act-steps", type=int, default=1, help="Number of ACTLossHead calls for training when not using --until-halt")
    parser.add_argument("--until-halt", action="store_true", help="Iterate training calls until all sequences halt (may take multiple steps)")

    # Code evolution flags
    parser.add_argument("--code-evolve", action="store_true", help="Run LLM-driven code evolution to produce a proposal JSON")
    parser.add_argument("--use-report", type=str, default="", help="Path to an existing evolve report JSON to feed into code evolution")
    parser.add_argument("--save-proposal", type=str, default="", help="Path to write the LLM code-evolution proposal (JSON or raw text on parse failure)")
    parser.add_argument("--llm-provider", type=str, default="ollama", help="LLM provider: ollama | openai")
    parser.add_argument("--llm-model", type=str, default="llama3.1:8b", help="LLM model name for the provider")
    parser.add_argument("--llm-host", type=str, default="", help="LLM host base URL (for Ollama; default http://localhost:11434)")
    parser.add_argument("--llm-timeout", type=int, default=120, help="Timeout (seconds) for LLM requests")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM sampling temperature")
    parser.add_argument("--llm-max-tokens", type=int, default=256, help="Max tokens to generate (smaller is faster, avoids timeouts)")
    parser.add_argument("--summary-bytes", type=int, default=8000, help="Max bytes per file to parse for summarization (smaller reduces prompt size)")
    parser.add_argument("--summary-basic", action="store_true", help="Summarize only key files (hrm.py, models/layers.py, evolve/pipeline.py)")

    parser.add_argument("--query-terms", nargs="*", default=["reasoning", "hierarchical", "transformer", "RL", "ACT"]) 
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--max-results", type=int, default=25)

    args = parser.parse_args()

    report = None
    if not args.no_train:
        run_forward_backward(args)

    if not args.no_evolve:
        report = run_self_evolve(args)

    if args.code_evolve:
        run_code_evolve(args, report)


if __name__ == "__main__":
    main()

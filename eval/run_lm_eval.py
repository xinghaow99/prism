import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add root directory and evaluation harness to sys.path
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(EVAL_DIR, "lm-evaluation-harness"))

import torch

from prism.utils.patch import _SUPPORTED_MODELS, apply_patch

PATCH_TYPE = os.environ.get("PATCH_TYPE", "prism").lower()


def _load_patch_context(patch_type: str):
    if patch_type == "none":
        return None, None

    if patch_type == "prism":
        from prism.prism import STAT_COLLECTOR, prism_attention_forward

        return prism_attention_forward, STAT_COLLECTOR

    if patch_type == "xattn":
        import baselines.XAttention as xattn

        return xattn.xattn_attention_forward, xattn.STAT_COLLECTOR

    if patch_type == "flexprefill":
        import baselines.FlexPrefill as flexprefill

        return flexprefill.flexprefill_attention_forward, flexprefill.STAT_COLLECTOR

    if patch_type == "minference":
        import baselines.Minference as minference

        return minference.minference_attention_forward, minference.STAT_COLLECTOR

    raise ValueError(
        f"Unsupported PATCH_TYPE: {patch_type}. Expected one of: none, prism, xattn, flexprefill, minference"
    )


def apply_attention_patch() -> None:
    if PATCH_TYPE == "none":
        print("Running without any attention monkey patch (Baseline).")
        return

    model_id = os.environ.get("MODEL_ID")

    try:
        patch_forward, stat_collector = _load_patch_context(PATCH_TYPE)
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import dependencies for PATCH_TYPE='{PATCH_TYPE}'. "
            f"Please install required optional dependencies. Original error: {e}"
        ) from e

    stat_collector.reset()

    patched_count = apply_patch(
        forward_fn=patch_forward,
        model_id=model_id,
        supported_models=_SUPPORTED_MODELS,
    )
    print(f"Successfully applied {PATCH_TYPE} monkey patch to {patched_count} attention classes.")


try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    print("Error: Could not import lm_eval. Make sure eval/lm-evaluation-harness is correctly placed and installed.")
    sys.exit(1)


def _parse_output_path(argv):
    output_path = None
    for idx, arg in enumerate(argv):
        if arg in ("--output_path", "-o") and idx + 1 < len(argv):
            output_path = argv[idx + 1]
            break
        if arg.startswith("--output_path="):
            output_path = arg.split("=", 1)[1]
            break
        if arg.startswith("-o="):
            output_path = arg.split("=", 1)[1]
            break
    return output_path


def _resolve_output_dir(output_path):
    if not output_path:
        return None
    path = Path(output_path)
    if path.suffix == ".json":
        return path.parent
    return path


def _is_rank0():
    return os.environ.get("RANK", "0") == "0"


def _reduce_sums(values, count):
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.tensor(values + [float(count)], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            reduced = tensor.cpu().tolist()
            return reduced[:-1], int(round(reduced[-1]))
    except Exception:
        pass
    return values, count


def _reduce_density_summary(summary):
    if not summary:
        return None
    sum_total = summary["total"] * summary["count"]
    count = summary["count"]
    (sum_total,), count = _reduce_sums([sum_total], count)
    if count == 0:
        return None
    return {
        "method": summary.get("method", PATCH_TYPE),
        "total": sum_total / count,
        "count": count,
    }


def _collect_density_summary():
    if PATCH_TYPE == "none":
        return None
    _, stat_collector = _load_patch_context(PATCH_TYPE)
    return _reduce_density_summary(stat_collector.summary())


def _write_density_summary(argv):
    if os.environ.get("COLLECT_DENSITY", "false").lower() not in ("1", "true", "yes"):
        return

    output_path = _parse_output_path(argv)
    output_dir = _resolve_output_dir(output_path)

    if output_dir is None:
        output_dir = Path("results")

    summary = _collect_density_summary()
    if not summary:
        return

    if not _is_rank0():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{PATCH_TYPE}_density_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Density summary saved to {output_file}")


if __name__ == "__main__":
    try:
        apply_attention_patch()
    except Exception as e:
        print(f"Error applying patch: {e}")
        sys.exit(1)

    cli_evaluate()
    _write_density_summary(sys.argv)

import importlib
from functools import lru_cache
from typing import Callable


# Centralized model support
_SUPPORTED_MODELS = ("qwen3_vl", "qwen3", "llama")


def get_attention_classes(model_id: str, supported_models: tuple[str, ...] = _SUPPORTED_MODELS):
    model_id = (model_id or "").lower()
    for model_name in supported_models:
        if model_name in model_id.replace("-", "_"):
            mod_name = f"transformers.models.{model_name}.modeling_{model_name}"
            mod = importlib.import_module(mod_name)

            # Determine class names (standard, FlashAttention2, SdpaAttention)
            classes = []
            for suffix in ["Attention", "FlashAttention2", "SdpaAttention"]:
                if model_name == "qwen3_vl":
                    cls_name = "Qwen3VLTextAttention" if suffix == "Attention" else f"Qwen3VLText{suffix}"
                else:
                    cls_name = f"{model_name.capitalize()}{suffix}"
                
                cls = getattr(mod, cls_name, None)
                if cls is not None and cls not in classes:
                    classes.append(cls)

            if not classes:
                raise RuntimeError(f"Could not find any attention classes for model '{model_name}'.")
            return classes
    raise RuntimeError(f"Unsupported model '{model_id}'. Supported: {supported_models}")


@lru_cache(maxsize=None)
def get_rotary_fn(module_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, "apply_rotary_pos_emb", None)


def apply_patch(
    forward_fn: Callable,
    model_id: str | None,
    supported_models: tuple[str, ...] = _SUPPORTED_MODELS,
) -> int:
    attn_classes = get_attention_classes(model_id=model_id, supported_models=supported_models)
    for cls in attn_classes:
        cls.forward = forward_fn
    return len(attn_classes)

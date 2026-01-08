"""
Core merging utilities for linear model merging.

Provides functions for:
1. Basic linear merging (weighted average of parameters)
2. Selective layer-wise merging (merge only specified layers)
"""

import torch
from typing import Dict, Optional, Set
from .layer_utils import extract_layer_index, is_embedding_or_head, is_qkv_parameter


def linear_merge_state_dicts(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    alpha: float,
    layers_to_merge: Optional[Set[int]] = None,
    layer_pattern: Optional[str] = None,
    qkv_only: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform linear merging of two state dictionaries.

    Merged parameter = alpha * param_a + (1 - alpha) * param_b

    Args:
        state_dict_a: State dict from model A (base/anchor model)
        state_dict_b: State dict from model B (model to merge in)
        alpha: Weight for model A (typically 0.5-0.9)
               alpha=1.0 means keep only model A
               alpha=0.0 means keep only model B
        layers_to_merge: Optional set of layer indices to merge.
                        If None, merge all parameters.
                        If specified, only merge these layers; other layers use model A.
        layer_pattern: Regex pattern for identifying layer parameters.
                      Required if layers_to_merge is specified.
        qkv_only: If True, only merge q_proj, k_proj, v_proj parameters in selected layers.
                 Other parameters (o_proj, mlp, layernorm, etc.) keep from model A.
                 Only effective when layers_to_merge is specified.

    Returns:
        Merged state dictionary

    Raises:
        ValueError: If layer_pattern is not provided when layers_to_merge is specified
        KeyError: If state dicts have mismatched keys

    Examples:
        # Full model merge
        >>> merged = linear_merge_state_dicts(sd_a, sd_b, alpha=0.7)

        # Selective layer merge (only layers 1, 2, 20)
        >>> merged = linear_merge_state_dicts(
        ...     sd_a, sd_b, alpha=0.7,
        ...     layers_to_merge={1, 2, 20},
        ...     layer_pattern=r'model\.layers\.(\d+)'
        ... )

        # QKV-only merge for specific layers
        >>> merged = linear_merge_state_dicts(
        ...     sd_a, sd_b, alpha=0.7,
        ...     layers_to_merge={1, 2, 20},
        ...     layer_pattern=r'model\.layers\.(\d+)',
        ...     qkv_only=True
        ... )
    """
    # Validate inputs
    if layers_to_merge is not None and layer_pattern is None:
        raise ValueError("layer_pattern must be provided when layers_to_merge is specified")

    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        raise KeyError(
            "State dictionaries have mismatched keys. "
            "Models must have identical architecture."
        )

    merged_state_dict = {}
    beta = 1.0 - alpha  # Weight for model B

    # Print merge mode
    if qkv_only and layers_to_merge is not None:
        print(f"Merging with alpha={alpha:.3f} (model_a={alpha:.3f}, model_b={beta:.3f}) [QKV-only mode]")
    else:
        print(f"Merging with alpha={alpha:.3f} (model_a={alpha:.3f}, model_b={beta:.3f})")

    # Track merge statistics
    num_merged = 0
    num_kept_from_a = 0
    num_qkv_merged = 0  # Track QKV merges separately

    for key in state_dict_a.keys():
        param_a = state_dict_a[key]
        param_b = state_dict_b[key]

        # Determine if this parameter should be merged
        should_merge = False

        if layers_to_merge is None:
            # No layer selection: merge everything (ignore qkv_only)
            should_merge = True
        else:
            # Layer selection active: check if this is a layer parameter
            layer_idx = extract_layer_index(key, layer_pattern)

            if layer_idx is not None:
                # This is a layer parameter
                if layer_idx in layers_to_merge:
                    if qkv_only:
                        # QKV-only mode: only merge Q/K/V parameters
                        if is_qkv_parameter(key):
                            should_merge = True
                            num_qkv_merged += 1
                        else:
                            should_merge = False
                    else:
                        # Normal mode: merge all parameters in this layer
                        should_merge = True
                else:
                    should_merge = False
            else:
                # Non-layer parameter (embedding, norm, head, etc.)
                # Keep from model A by default
                should_merge = False

        # Perform merge or keep from model A
        if should_merge:
            merged_state_dict[key] = alpha * param_a + beta * param_b
            num_merged += 1
        else:
            merged_state_dict[key] = param_a.clone()
            num_kept_from_a += 1

    # Print merge statistics
    total_params = len(state_dict_a)
    print(f"Merge statistics:")
    print(f"  Total parameters: {total_params}")
    print(f"  Merged parameters: {num_merged}")
    if qkv_only and layers_to_merge is not None:
        print(f"    (QKV parameters: {num_qkv_merged})")
    print(f"  Kept from model A: {num_kept_from_a}")

    return merged_state_dict


def apply_special_token_fixes(
    merged_state_dict: Dict[str, torch.Tensor],
    state_dict_base: Dict[str, torch.Tensor],
    special_token_ids: Optional[list] = None
) -> None:
    """
    Apply special fixes to specific tokens in lm_head (in-place operation).

    This preserves certain special tokens from the base model, which is important
    for maintaining compatibility with the base model's tokenizer.

    Args:
        merged_state_dict: Merged state dict (will be modified in-place)
        state_dict_base: Base model's state dict
        special_token_ids: List of token IDs to preserve from base model.
                          If None, uses default [151643, 151645] for Qwen models.

    Note:
        This function modifies merged_state_dict in-place.
        The default token IDs are specific to Qwen models.
    """
    if special_token_ids is None:
        # Default special token IDs for Qwen models
        special_token_ids = [151643, 151645]

    for key in merged_state_dict.keys():
        if "lm_head" in key and len(merged_state_dict[key].shape) >= 2:
            # Apply special token fixes
            for token_id in special_token_ids:
                try:
                    merged_state_dict[key][token_id, :] = state_dict_base[key][token_id, :]
                except IndexError:
                    # Token ID out of range for this model, skip
                    continue

            print(f"Applied special token fixes to {key} for token IDs: {special_token_ids}")


def merge_models_selective(
    model_a,  # AutoModelForCausalLM instance
    model_b,  # AutoModelForCausalLM instance
    alpha: float,
    layers_to_merge: Optional[Set[int]] = None,
    layer_pattern: Optional[str] = None,
    qkv_only: bool = False,
    apply_special_fixes: bool = True,
    special_token_ids: Optional[list] = None
) -> Dict[str, torch.Tensor]:
    """
    High-level function to merge two models with optional layer selection.

    Args:
        model_a: Base model (AutoModelForCausalLM instance)
        model_b: Model to merge in (AutoModelForCausalLM instance)
        alpha: Merging weight for model A
        layers_to_merge: Optional set of layer indices to merge
        layer_pattern: Regex pattern for layer identification
        qkv_only: If True, only merge Q/K/V parameters in selected layers
        apply_special_fixes: Whether to apply special token fixes (default: True)
        special_token_ids: Special token IDs to preserve (Qwen-specific by default)

    Returns:
        Merged state dictionary ready to be loaded into a model

    Examples:
        # Full model merge
        >>> merged_sd = merge_models_selective(model_a, model_b, alpha=0.7)

        # Merge only specific layers
        >>> merged_sd = merge_models_selective(
        ...     model_a, model_b, alpha=0.7,
        ...     layers_to_merge={1, 2, 20, 25},
        ...     layer_pattern=r'model\.layers\.(\d+)'
        ... )

        # Merge only QKV in specific layers
        >>> merged_sd = merge_models_selective(
        ...     model_a, model_b, alpha=0.7,
        ...     layers_to_merge={1, 2, 20, 25},
        ...     layer_pattern=r'model\.layers\.(\d+)',
        ...     qkv_only=True
        ... )
    """
    print("Extracting state dictionaries...")
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    # Perform linear merge
    merged_state_dict = linear_merge_state_dicts(
        state_dict_a=state_dict_a,
        state_dict_b=state_dict_b,
        alpha=alpha,
        layers_to_merge=layers_to_merge,
        layer_pattern=layer_pattern,
        qkv_only=qkv_only
    )

    # Apply special token fixes if requested
    if apply_special_fixes:
        apply_special_token_fixes(
            merged_state_dict=merged_state_dict,
            state_dict_base=state_dict_a,
            special_token_ids=special_token_ids
        )

    return merged_state_dict

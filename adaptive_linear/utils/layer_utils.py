"""
Layer utilities for parsing and validating layer indices in transformer models.

Supports common architectures including:
- LLaMA/LLaMA-2/LLaMA-3 (model.layers[i])
- Qwen/Qwen2 (model.layers[i])
- GPT-2/GPT-J (transformer.h[i])
- Mistral/Mixtral (model.layers[i])
"""

import re
from typing import List, Optional, Set


# Common layer key patterns for different model architectures
LAYER_PATTERNS = {
    'llama': r'model\.layers\.(\d+)',      # LLaMA, LLaMA-2, LLaMA-3
    'qwen': r'model\.layers\.(\d+)',       # Qwen, Qwen2
    'mistral': r'model\.layers\.(\d+)',    # Mistral, Mixtral
    'gpt2': r'transformer\.h\.(\d+)',      # GPT-2, GPT-J
    'gpt_neox': r'gpt_neox\.layers\.(\d+)',  # GPT-NeoX
}


def parse_layer_indices(layer_args: Optional[List[int]]) -> Optional[Set[int]]:
    """
    Parse layer indices from command line arguments.

    Args:
        layer_args: List of layer indices from argparse (e.g., [1, 2, 20, 25])
                   None means merge all layers

    Returns:
        Set of layer indices to merge, or None if all layers should be merged

    Examples:
        >>> parse_layer_indices([1, 2, 20])
        {1, 2, 20}
        >>> parse_layer_indices(None)
        None
    """
    if layer_args is None:
        return None

    # Remove duplicates and sort
    layer_set = set(layer_args)

    # Validate that all indices are non-negative
    if any(idx < 0 for idx in layer_set):
        raise ValueError(f"Layer indices must be non-negative. Got: {layer_args}")

    return layer_set


def validate_layer_indices(layer_indices: Set[int], max_layer_idx: int) -> None:
    """
    Validate that layer indices are within valid range.

    Args:
        layer_indices: Set of layer indices to validate
        max_layer_idx: Maximum valid layer index (exclusive)

    Raises:
        ValueError: If any layer index is out of range

    Examples:
        >>> validate_layer_indices({1, 2, 20}, 28)  # Valid
        >>> validate_layer_indices({1, 2, 30}, 28)  # Raises ValueError
    """
    invalid_indices = [idx for idx in layer_indices if idx >= max_layer_idx]

    if invalid_indices:
        raise ValueError(
            f"Layer indices out of range. Model has {max_layer_idx} layers (0-{max_layer_idx-1}), "
            f"but got indices: {sorted(invalid_indices)}"
        )


def get_layer_key_pattern(state_dict: dict) -> str:
    """
    Automatically detect the layer key pattern from a model's state_dict.

    Args:
        state_dict: Model's state dictionary

    Returns:
        Regex pattern string for matching layer keys

    Raises:
        ValueError: If no recognized layer pattern is found

    Examples:
        For a Qwen model with keys like "model.layers.0.self_attn.q_proj.weight":
        >>> pattern = get_layer_key_pattern(state_dict)
        >>> pattern
        'model\\.layers\\.(\\d+)'
    """
    # Try to find a matching pattern by checking a few keys
    sample_keys = list(state_dict.keys())[:100]  # Check first 100 keys

    for architecture, pattern in LAYER_PATTERNS.items():
        for key in sample_keys:
            if re.search(pattern, key):
                return pattern

    # If no pattern found, raise error
    raise ValueError(
        "Could not detect layer pattern from model state_dict. "
        f"Supported architectures: {list(LAYER_PATTERNS.keys())}"
    )


def extract_layer_index(param_name: str, pattern: str) -> Optional[int]:
    """
    Extract layer index from a parameter name using the given pattern.

    Args:
        param_name: Parameter name (e.g., "model.layers.5.self_attn.q_proj.weight")
        pattern: Regex pattern to match (e.g., r'model\.layers\.(\d+)')

    Returns:
        Layer index if found, None otherwise

    Examples:
        >>> extract_layer_index("model.layers.5.self_attn.q_proj.weight", r'model\.layers\.(\d+)')
        5
        >>> extract_layer_index("model.embed_tokens.weight", r'model\.layers\.(\d+)')
        None
    """
    match = re.search(pattern, param_name)
    if match:
        return int(match.group(1))
    return None


def get_num_layers(state_dict: dict, pattern: str) -> int:
    """
    Get the total number of transformer layers in the model.

    Args:
        state_dict: Model's state dictionary
        pattern: Regex pattern for matching layer keys

    Returns:
        Total number of layers

    Examples:
        >>> get_num_layers(state_dict, r'model\.layers\.(\d+)')
        28
    """
    layer_indices = set()

    for key in state_dict.keys():
        layer_idx = extract_layer_index(key, pattern)
        if layer_idx is not None:
            layer_indices.add(layer_idx)

    if not layer_indices:
        raise ValueError(f"No layers found in state_dict using pattern: {pattern}")

    # Return max index + 1 (assuming layers are 0-indexed and contiguous)
    return max(layer_indices) + 1


def is_embedding_or_head(param_name: str, pattern: str) -> bool:
    """
    Check if a parameter belongs to embeddings or language model head.

    These parameters are typically NOT merged layer-wise and should come from base model.

    Args:
        param_name: Parameter name
        pattern: Layer pattern (to exclude layer-specific params)

    Returns:
        True if parameter is embedding/head, False if it's a layer parameter

    Examples:
        >>> is_embedding_or_head("model.embed_tokens.weight", r'model\.layers\.(\d+)')
        True
        >>> is_embedding_or_head("model.layers.5.mlp.gate_proj.weight", r'model\.layers\.(\d+)')
        False
        >>> is_embedding_or_head("lm_head.weight", r'model\.layers\.(\d+)')
        True
    """
    # If it matches the layer pattern, it's a layer parameter
    if extract_layer_index(param_name, pattern) is not None:
        return False

    # Common embedding and head keywords
    embedding_head_keywords = [
        'embed_tokens',
        'embed_positions',
        'wte',  # word token embeddings (GPT-2)
        'wpe',  # word position embeddings (GPT-2)
        'lm_head',
        'embed_out',
    ]

    # Check if any keyword is in the parameter name
    return any(keyword in param_name for keyword in embedding_head_keywords)


def is_qkv_parameter(param_name: str) -> bool:
    """
    Check if a parameter is a query, key, or value projection matrix.

    Common naming patterns across different architectures:
    - LLaMA/Qwen/Mistral: q_proj, k_proj, v_proj
    - GPT-2/GPT-J: c_attn (combined QKV)
    - Some models: query, key, value
    - Attention: attn.q, attn.k, attn.v

    Args:
        param_name: Parameter name to check

    Returns:
        True if parameter is Q/K/V projection, False otherwise

    Examples:
        >>> is_qkv_parameter("model.layers.5.self_attn.q_proj.weight")
        True
        >>> is_qkv_parameter("model.layers.5.self_attn.k_proj.weight")
        True
        >>> is_qkv_parameter("model.layers.5.self_attn.v_proj.weight")
        True
        >>> is_qkv_parameter("model.layers.5.self_attn.o_proj.weight")
        False
        >>> is_qkv_parameter("model.layers.5.mlp.gate_proj.weight")
        False
    """
    # Keywords that identify Q/K/V parameters
    qkv_keywords = [
        'q_proj',      # Query projection (LLaMA, Qwen, Mistral)
        'k_proj',      # Key projection
        'v_proj',      # Value projection
        'c_attn',      # Combined QKV (GPT-2, GPT-J)
        'query.',      # Alternative naming
        'key.',
        'value.',
        'attn.q',      # Another variant
        'attn.k',
        'attn.v',
        'Wq',          # Some models use capital letters
        'Wk',
        'Wv',
    ]

    # Check if any QKV keyword is in the parameter name
    param_lower = param_name.lower()

    # Check exact matches and patterns
    for keyword in qkv_keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in param_lower:
            # Additional check: make sure it's not output projection (o_proj, out_proj, etc.)
            # Output projection often contains 'o' but we want to exclude it
            if 'o_proj' not in param_lower and 'out_proj' not in param_lower:
                return True

    return False

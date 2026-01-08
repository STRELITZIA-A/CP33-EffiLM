"""
Adaptive Linear Model Merging Script

This script merges two language models using linear interpolation,
with support for selective layer-wise merging.

Usage:
    # Full model merge
    python merge.py --model_a /path/to/model_a --model_b /path/to/model_b --alpha 0.7

    # Selective layer merge
    python merge.py --model_a /path/to/model_a --model_b /path/to/model_b --alpha 0.7 --layers 1 2 20 25

Author: Adaptive Linear Merging Project
"""

import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

from utils.layer_utils import (
    parse_layer_indices,
    validate_layer_indices,
    get_layer_key_pattern,
    get_num_layers
)
from utils.merge_utils import merge_models_selective

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive Linear Model Merging with Layer Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge entire models with alpha=0.7
  python merge.py --model_a /path/to/base_model --model_b /path/to/other_model --alpha 0.7

  # Merge only specific layers (1, 2, 20, 25, 27)
  python merge.py --model_a /path/to/base_model --model_b /path/to/other_model --alpha 0.7 --layers 1 2 20 25 27

  # Merge only Q/K/V matrices in specific layers
  python merge.py --model_a /path/to/base_model --model_b /path/to/other_model --alpha 0.7 --layers 1 2 20 --qkv_only

  # Specify custom output directory
  python merge.py --model_a /path/to/base_model --model_b /path/to/other_model --alpha 0.7 --output_dir ./my_merged_model
        """
    )

    parser.add_argument(
        '--model_a',
        type=str,
        required=True,
        help='Path to base/anchor model (model A). This model provides the foundation.'
    )

    parser.add_argument(
        '--model_b',
        type=str,
        required=True,
        help='Path to model B to merge in.'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        required=True,
        help='Merging weight for model A. Range: [0, 1]. '
             'alpha=1.0 keeps only model A, alpha=0.0 keeps only model B. '
             'Typical values: 0.5-0.9'
    )

    parser.add_argument(
        '--layers',
        type=int,
        nargs='*',
        default=None,
        help='Optional: Specify layer indices to merge (space-separated). '
             'If not provided, all layers will be merged. '
             'Example: --layers 1 2 20 25 27'
    )

    parser.add_argument(
        '--qkv_only',
        action='store_true',
        help='Optional: Only merge Q/K/V projection matrices in selected layers. '
             'Other parameters (output projection, MLP, layernorm) will be kept from model A. '
             'Only effective when --layers is specified. '
             'Example: --layers 1 2 20 --qkv_only'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional: Output directory for merged model. '
             'If not provided, will auto-generate based on model names and parameters.'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for loading models (default: cpu)'
    )

    parser.add_argument(
        '--no_special_fixes',
        action='store_true',
        help='Disable special token fixes for lm_head (default: enabled)'
    )

    args = parser.parse_args()

    # Validate alpha
    if not (0.0 <= args.alpha <= 1.0):
        parser.error(f"--alpha must be in range [0, 1], got {args.alpha}")

    return args


def generate_output_dir(model_a_path: str, model_b_path: str, alpha: float, layers: list = None, qkv_only: bool = False) -> str:
    """
    Generate a descriptive output directory name.

    Args:
        model_a_path: Path to model A
        model_b_path: Path to model B
        alpha: Merging weight
        layers: Optional list of layer indices
        qkv_only: Whether only QKV matrices are merged

    Returns:
        Path to output directory

    Examples:
        >>> generate_output_dir("/path/to/model_a", "/path/to/model_b", 0.7)
        'merged_models/model_a_model_b_alpha0.70'

        >>> generate_output_dir("/path/to/model_a", "/path/to/model_b", 0.7, [1, 2, 20])
        'merged_models/model_a_model_b_alpha0.70_layers1-2-20'

        >>> generate_output_dir("/path/to/model_a", "/path/to/model_b", 0.7, [1, 2, 20], True)
        'merged_models/model_a_model_b_alpha0.70_layers1-2-20_qkv'
    """
    # Extract model names from paths
    model_a_name = os.path.basename(model_a_path.rstrip('/'))
    model_b_name = os.path.basename(model_b_path.rstrip('/'))

    # Build directory name
    dir_name = f"{model_a_name}_{model_b_name}_alpha{alpha:.2f}"

    if layers is not None and len(layers) > 0:
        # Add layer information
        sorted_layers = sorted(layers)
        if len(sorted_layers) <= 10:
            # If few layers, list them all
            layer_str = '-'.join(map(str, sorted_layers))
        else:
            # If many layers, show count and range
            layer_str = f"count{len(sorted_layers)}_range{min(sorted_layers)}-{max(sorted_layers)}"
        dir_name += f"_layers{layer_str}"

        # Add qkv_only suffix if applicable
        if qkv_only:
            dir_name += "_qkv"

    # Create full path under merged_models/
    output_path = os.path.join("merged_models", dir_name)

    return output_path


def get_all_stop_tokens(tokenizer):
    """
    Extract all potential stop tokens from a tokenizer.
    (Copied from original model_merge.py)
    """
    stop_tokens = set()

    # Get eos token
    if tokenizer.eos_token_id is not None:
        stop_tokens.add(tokenizer.eos_token_id)

    return list(stop_tokens)


def save_merged_model(model, tokenizer_a, tokenizer_b, save_path: str):
    """
    Save merged model and handle tokenizer/generation config.

    Args:
        model: Merged model instance
        tokenizer_a: Tokenizer from model A
        tokenizer_b: Tokenizer from model B
        save_path: Path to save the merged model
    """
    print(f"\nSaving merged model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save model
    model.save_pretrained(save_path)
    print("  ✓ Model saved")

    # Save tokenizer (from model A)
    tokenizer_a.save_pretrained(save_path)
    print("  ✓ Tokenizer saved")

    # Merge stop tokens from both tokenizers
    print("  Merging stop tokens...")
    stop_tokens_a = get_all_stop_tokens(tokenizer_a)
    stop_tokens_b = get_all_stop_tokens(tokenizer_b)
    merged_stop_tokens = list(set(stop_tokens_a + stop_tokens_b))

    # Update generation config
    generation_config_path = os.path.join(save_path, "generation_config.json")
    if os.path.exists(generation_config_path):
        with open(generation_config_path, 'r', encoding='utf-8') as f:
            generation_config = json.load(f)

        generation_config['eos_token_id'] = merged_stop_tokens

        with open(generation_config_path, 'w', encoding='utf-8') as f:
            json.dump(generation_config, f, indent=2)
        print("  ✓ Generation config updated")

    print(f"\n{'='*60}")
    print(f"Merged model saved successfully to: {save_path}")
    print(f"{'='*60}")


def main():
    """Main execution function."""
    args = parse_args()

    print("="*60)
    print("Adaptive Linear Model Merging")
    print("="*60)
    print(f"Model A (base):  {args.model_a}")
    print(f"Model B:         {args.model_b}")
    print(f"Alpha:           {args.alpha}")
    print(f"Layers to merge: {'All layers' if args.layers is None else args.layers}")
    if args.qkv_only and args.layers is not None:
        print(f"QKV-only mode:   Enabled (only merge Q/K/V matrices)")
    print(f"Device:          {args.device}")
    print("="*60)

    # Parse layer indices
    layer_indices = parse_layer_indices(args.layers)

    # Load model A
    print("\n[1/5] Loading model A (base model)...")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a,
        device_map="auto",
        torch_dtype="auto"
    )
    print(f"  Model A loaded: {model_a.config._name_or_path}")
    print(f"  Architecture: {model_a.config.model_type}")

    # Detect layer pattern and validate layer indices
    print("\n[2/5] Analyzing model structure...")
    state_dict_a = model_a.state_dict()
    layer_pattern = get_layer_key_pattern(state_dict_a)
    num_layers = get_num_layers(state_dict_a, layer_pattern)
    print(f"  Layer pattern: {layer_pattern}")
    print(f"  Total layers: {num_layers}")

    if layer_indices is not None:
        validate_layer_indices(layer_indices, num_layers)
        print(f"  Validated layer indices: {sorted(layer_indices)}")

    # Load model B
    print("\n[3/5] Loading model B...")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b,
        device_map="auto",
        torch_dtype="auto"
    )
    print(f"  Model B loaded: {model_b.config._name_or_path}")

    # Perform merge 合并的话应该是指定层+qkv合并，其余保留
    print("\n[4/5] Performing linear merge...")
    merged_state_dict = merge_models_selective(
        model_a=model_a,
        model_b=model_b,
        alpha=args.alpha,
        layers_to_merge=layer_indices,
        layer_pattern=layer_pattern if layer_indices is not None else None,
        qkv_only=args.qkv_only,
        apply_special_fixes=not args.no_special_fixes
    )

    # Load merged weights into model_a
    print("\n  Loading merged weights into model...")
    model_a.load_state_dict(merged_state_dict)
    print("  ✓ Merged weights loaded")

    # Clean up model B
    del model_b, merged_state_dict
    torch.cuda.empty_cache()

    # Determine output directory
    if args.output_dir is None:
        output_dir = generate_output_dir(args.model_a, args.model_b, args.alpha, args.layers, args.qkv_only)
    else:
        output_dir = args.output_dir

    # Load tokenizers and save
    print("\n[5/5] Saving merged model...")
    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a)
    tokenizer_b = AutoTokenizer.from_pretrained(args.model_b)

    save_merged_model(model_a, tokenizer_a, tokenizer_b, output_dir)

    # Clean up
    del model_a, tokenizer_a, tokenizer_b
    torch.cuda.empty_cache()

    print("\n✨ Merge completed successfully!")


if __name__ == "__main__":
    main()

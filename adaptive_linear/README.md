# Adaptive Linear Model Merging

A flexible and extensible framework for merging language models using linear interpolation, with support for selective layer-wise merging.

## Overview

This project extends the proven **linear merging** approach (baseline method) with adaptive layer selection capabilities. It allows you to:

- **Full model merge**: Merge all parameters of two models
- **Selective layer merge**: Merge only specific transformer layers, keeping other layers from the base model
- **Automatic structure detection**: Works with LLaMA, Qwen, Mistral, GPT-2, and other architectures
- **Research-friendly**: Clean, well-documented code for experimentation

## Project Structure

```
adaptive_linear/
├── merge.py                  # Main merging script (CLI interface)
├── model_merge.py            # Original linear merging script (for reference)
├── utils/
│   ├── __init__.py
│   ├── layer_utils.py        # Layer indexing and name parsing
│   └── merge_utils.py        # Core linear merge logic
├── merged_models/            # Output directory (auto-created)
│   └── <merged_model_name>/
├── instructions.txt          # Project requirements
└── README.md                 # This file
```

## Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers
```

## Usage

### Basic Usage: Full Model Merge

Merge all parameters of two models:

```bash
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/other_model \
  --alpha 0.7
```

**What this does:**
- Merged parameters = `0.7 * model_a + 0.3 * model_b`
- All layers and parameters are merged
- Equivalent to original `model_merge.py` behavior

### Advanced Usage: Selective Layer Merge

Merge only specific transformer layers:

```bash
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/other_model \
  --alpha 0.7 \
  --layers 1 2 20 25 27
```

**What this does:**
- Layers 1, 2, 20, 25, 27: Merged using linear interpolation
- All other layers: Kept from `model_a` (base model)
- Embeddings and `lm_head`: Always kept from `model_a`

### Advanced Usage: QKV-Only Merge

Merge **only Q/K/V projection matrices** in specific layers, keeping all other parameters from the base model:

```bash
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/other_model \
  --alpha 0.7 \
  --layers 1 2 20 25 27 \
  --qkv_only
```

**What this does:**
- Layers 1, 2, 20, 25, 27:
  - **Q/K/V projections** (`q_proj`, `k_proj`, `v_proj`): Merged using linear interpolation
  - **Other parameters** (output projection, MLP, layernorm): Kept from `model_a`
- All other layers: Completely kept from `model_a`
- Embeddings and `lm_head`: Always kept from `model_a`

**Use case:** This is useful for transferring attention patterns while preserving the model's overall structure and reasoning capabilities.

### Custom Output Directory

```bash
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/other_model \
  --alpha 0.7 \
  --layers 1 2 20 \
  --output_dir ./my_custom_output
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_a` | Yes | Path to base/anchor model (Model A) |
| `--model_b` | Yes | Path to model to merge in (Model B) |
| `--alpha` | Yes | Merging weight for model A (range: 0.0-1.0) |
| `--layers` | No | Space-separated layer indices to merge (default: all layers) |
| `--qkv_only` | No | Only merge Q/K/V matrices in selected layers (requires `--layers`) |
| `--output_dir` | No | Custom output directory (default: auto-generated) |
| `--device` | No | Device for loading models: `cpu` or `cuda` (default: `cpu`) |
| `--no_special_fixes` | No | Disable special token fixes for `lm_head` |

## Output Directory Naming

If `--output_dir` is not specified, the script auto-generates a descriptive name:

**Examples:**

- Full merge: `merged_models/Qwen2.5-7B_DeepSeek-R1-7B_alpha0.70/`
- Layer merge: `merged_models/Qwen2.5-7B_DeepSeek-R1-7B_alpha0.70_layers1-2-20-25/`
- QKV-only merge: `merged_models/Qwen2.5-7B_DeepSeek-R1-7B_alpha0.70_layers1-2-20-25_qkv/`

## How It Works

### 1. Layer Selection Logic

```python
# Pseudocode
for each parameter in model:
    if layers_to_merge is None:
        # Full model merge
        merged_param = alpha * param_a + (1 - alpha) * param_b
    else:
        layer_idx = extract_layer_index(param_name)
        if layer_idx in layers_to_merge:
            if qkv_only:
                # QKV-only mode: only merge Q/K/V parameters
                if is_qkv_parameter(param_name):
                    merged_param = alpha * param_a + (1 - alpha) * param_b
                else:
                    merged_param = param_a  # Keep from base model
            else:
                # Normal mode: merge all parameters in this layer
                merged_param = alpha * param_a + (1 - alpha) * param_b
        else:
            # Keep from base model
            merged_param = param_a
```

### 2. Special Handling

- **Embeddings** (`embed_tokens`, `wte`, etc.): Always from `model_a`
- **Language model head** (`lm_head`): Always from `model_a`, with special token fixes
- **Layer normalization**: Follows layer selection rules

### 3. Supported Architectures

The code automatically detects layer patterns for:

- **LLaMA/LLaMA-2/LLaMA-3**: `model.layers[i]`
- **Qwen/Qwen2**: `model.layers[i]`
- **Mistral/Mixtral**: `model.layers[i]`
- **GPT-2/GPT-J**: `transformer.h[i]`
- **GPT-NeoX**: `gpt_neox.layers[i]`

## Examples

### Example 1: Merge Math and Reasoning Models

```bash
python merge.py \
  --model_a /path/to/Qwen2.5-Math-7B \
  --model_b /path/to/DeepSeek-R1-7B \
  --alpha 0.8 \
  --layers 5 10 15 20 25
```

Merges reasoning capabilities from DeepSeek-R1 into specific layers of Qwen Math model.

### Example 2: Full Model Merge (Baseline)

```bash
python merge.py \
  --model_a /path/to/Qwen2.5-Math-7B \
  --model_b /path/to/Qwen2.5-Instruct-7B \
  --alpha 0.6
```

Traditional linear merging of entire models.

### Example 3: QKV-Only Merge for Attention Transfer

```bash
# Transfer attention patterns from a specialized model
python merge.py \
  --model_a /path/to/Qwen2.5-Math-7B \
  --model_b /path/to/DeepSeek-R1-7B \
  --alpha 0.7 \
  --layers 10 15 20 25 \
  --qkv_only
```

This merges only the attention query/key/value matrices from DeepSeek-R1 into selected layers, while keeping the MLP and output projections from Qwen Math. Useful for transferring attention patterns without disrupting the model's reasoning structure.

### Example 4: Research Experiment

```bash
# Experiment: Merge only early layers
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/specialized_model \
  --alpha 0.7 \
  --layers 0 1 2 3 4 \
  --output_dir experiments/early_layers_merge

# Experiment: Merge only late layers
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/specialized_model \
  --alpha 0.7 \
  --layers 20 21 22 23 24 25 26 27 \
  --output_dir experiments/late_layers_merge

# Experiment: QKV-only merge in middle layers
python merge.py \
  --model_a /path/to/base_model \
  --model_b /path/to/specialized_model \
  --alpha 0.7 \
  --layers 10 11 12 13 14 15 \
  --qkv_only \
  --output_dir experiments/middle_layers_qkv_merge
```

## Code Quality Features

- **Type hints**: Clear function signatures
- **Docstrings**: Comprehensive documentation for all functions
- **Error handling**: Validation for layer indices, alpha values, and model compatibility
- **Logging**: Detailed progress output and merge statistics
- **Modular design**: Easily extensible for future research

## Future Extensions

This codebase is designed to support:

- **Layer-wise alpha**: Different alpha values per layer
- **Sensitivity-driven merging**: Automatic layer selection based on importance scores
- **Performance tracking**: Token usage and inference latency statistics
- **Advanced merge strategies**: TIES, DARE, SLERP, etc.

## Troubleshooting

### Layer Index Out of Range

```
ValueError: Layer indices out of range. Model has 28 layers (0-27), but got indices: [30]
```

**Solution**: Check your model's layer count and adjust `--layers` accordingly.

### Mismatched Model Architectures

```
KeyError: State dictionaries have mismatched keys.
```

**Solution**: Ensure both models have the same architecture (e.g., both Qwen-7B, not Qwen-7B and LLaMA-7B).

### Memory Issues

**Solution**: Use `--device cpu` and ensure sufficient RAM. For large models (>13B), consider using quantization or gradient checkpointing.
```

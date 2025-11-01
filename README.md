# Qwen3-8B GGUF Model Launcher

Local deployment scripts for Qwen3-8B quantized model using llama-cpp-python with CUDA acceleration.

## Files

- `qwen3-script.py` - Full version with diagnostics, GPU monitoring, and benchmarking
- `qwen3-bare-script.py` - Minimal version (71 lines) with core functionality only
- `environment.yml` - Conda environment configuration

## Requirements

- Python 3.10+ (3.13 recommended)
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.0+ (13.0 recommended)
- NVIDIA Driver 525+ (580+ for CUDA 13)
- llama-cpp-python compiled with CUDA support

## Model

- **File**: qwen3-8b-q5_k_m.gguf (5.85 GB)
- **Download**: https://huggingface.co/Qwen/Qwen3-8B-GGUF
- **Path**: `E:\ai\models\qwen3-8b-gguf\qwen3-8b-q5_k_m.gguf`

## Installation

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate qwen3launcher
```

2. Install llama-cpp-python with CUDA:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

3. Download model to specified path

4. Run script:
```bash
python qwen3-script.py        # Full version
python qwen3-bare-script.py   # Minimal version
```

## Configuration

Key parameters (modify in script):
- `MODEL_PATH`: Model file location
- `N_CTX`: Context window (8192 tokens)
- `N_GPU_LAYERS`: GPU offloading (-1 for all)
- `N_BATCH`: Batch size (512)
- `MAX_TOKENS`: Maximum generation (2048)
- `TEMPERATURE`: Sampling temperature (0.7)

## Functions

Both scripts provide:
- `generate(prompt, **kwargs)` - Single response generation
- `stream_generate(prompt, **kwargs)` - Streaming generation
- `chat(messages, stream=True, **kwargs)` - Chat completion
- `chat_loop()` - Interactive chat interface

Full version additionally includes:
- `benchmark()` - Performance testing
- `model_info()` - Display model/GPU details
- `gpu_monitor()` - Real-time GPU monitoring

## Usage

### Interactive Chat
Scripts auto-start chat loop when run. Commands:
- Type message and press Enter to chat
- `quit` - Exit
- `clear` - Reset conversation

### Python Import
```python
from qwen3_bare_script import llm, generate, chat

# Simple generation
response = generate("Explain quantum computing")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
]
response = chat(messages)
```

## Memory Requirements

- Model file: 5.85 GB
- Runtime VRAM: 7-8 GB
- Recommended: 8GB+ free VRAM
- System RAM: 16GB+ recommended

## Performance

Expected on RTX 3090 Ti (24GB VRAM):
- ~30-50 tokens/second
- 20-35ms per token latency
- Full 8K context support
- 1008 GB/s memory bandwidth

## Troubleshooting

If model fails to load:
- Check VRAM availability: `nvidia-smi`
- Reduce context: Lower `N_CTX`
- Reduce batch size: Lower `N_BATCH`
- Force CPU mode: Set `USE_GPU = False` (slow)

If CUDA not detected:
- Verify driver: `nvidia-smi`
- Reinstall llama-cpp-python with CUDA
- Check CUDA toolkit installation

## License

Model: Qwen3 license (check Hugging Face page)
Scripts: Provided as-is for local use

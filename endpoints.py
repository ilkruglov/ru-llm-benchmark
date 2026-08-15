import os
from pathlib import Path


def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()


def get_endpoints():
    return [
        {
            "tag": "27b-v2",
            "url": os.environ.get("QWEN27B_URL", ""),
            "key": os.environ.get("QWEN27B_API_KEY", ""),
            "model": "Qwen3.6-27B-FP8",
            "type": "qwen",
            "reasoning_field": "reasoning",
        },
        {
            "tag": "35b-a3b",
            "url": os.environ.get("QWEN35B_URL", ""),
            "key": os.environ.get("QWEN35B_API_KEY", ""),
            "model": "Qwen3.6-35B-A3B-FP8",
            "type": "qwen",
            "reasoning_field": "reasoning",
        },
        {
            "tag": "122b",
            "url": os.environ.get("QWEN122B_URL", ""),
            "key": os.environ.get("QWEN122B_API_KEY", ""),
            "model": "Qwen3.5-122B-A10B-FP8",
            "type": "qwen",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "gpt-oss-120b",
            "url": os.environ.get("GROQ_PROXY_URL", "https://api.groq.com/openai/v1"),
            "key": os.environ.get("GROQ_API_KEY", ""),
            "model": "openai/gpt-oss-120b",
            "type": "groq",
            "reasoning_field": "reasoning",
        },
        {
            "tag": "llama-scout",
            "url": "https://api.groq.com/openai/v1",   # direct: proxy hangs on VL; direct works & is fast
            "key": os.environ.get("GROQ_API_KEY", ""),
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "type": "groq",
            "reasoning_field": "reasoning",
        },
        {
            "tag": "deepseek-flash",
            "url": "https://api.deepseek.com/v1",
            "key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-v4-flash",
            "type": "deepseek",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "deepseek-flash-0731",
            "url": "https://api.deepseek.com/v1",
            "key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-v4-flash",   # alias now serves DeepSeek-V4-Flash-0731; the deepseek-flash rows above are the earlier snapshot
            "type": "deepseek",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "deepseek-pro",
            "url": "https://api.deepseek.com/v1",
            "key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-v4-pro",
            "type": "deepseek",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "deepseek-pro-new",
            "url": "https://api.deepseek.com/v1",
            "key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-v4-pro",   # alias now serves the newer pro snapshot; the deepseek-pro rows above are the earlier one
            "type": "deepseek",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "nemotron-super",
            "url": "https://api.deepinfra.com/v1/openai",
            "key": os.environ.get("DEEPINFRA_API_KEY", ""),
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            "type": "deepinfra",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "minimax-m3",
            "url": "https://api.deepinfra.com/v1/openai",
            "key": os.environ.get("DEEPINFRA_API_KEY", ""),
            "model": "MiniMaxAI/MiniMax-M3",
            "type": "deepinfra",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "nemotron-omni",
            "url": "https://api.deepinfra.com/v1/openai",
            "key": os.environ.get("DEEPINFRA_API_KEY", ""),
            "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
            "type": "deepinfra",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "glm-5.2",
            "url": "https://api.deepinfra.com/v1/openai",
            "key": os.environ.get("DEEPINFRA_API_KEY", ""),
            "model": "zai-org/GLM-5.2",
            "type": "deepinfra",
            "reasoning_field": "reasoning_content",
        },
    ]

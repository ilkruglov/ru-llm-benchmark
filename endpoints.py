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
            "tag": "deepseek-flash-vl",
            "url": "https://api.deepseek.com/v1",
            "key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-v4-flash-vision-exp",   # DeepSeek's experimental vision head on V4-Flash (only DS model that accepts images)
            "type": "deepseek",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "empirio-flash",
            "url": "https://api.empiriolabs.ai/v1",   # same EmpirioLabs host as empirio-qwen38, different model id
            "key": os.environ.get("EMPIRIOLABS_API_KEY", ""),
            "model": "qwen3-8-flash",   # Qwen3.8-Flash (smaller/faster 3.8 variant)
            "type": "empirio",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "empirio-max",
            "url": "https://api.empiriolabs.ai/v1",
            "key": os.environ.get("EMPIRIOLABS_API_KEY", ""),
            "model": "qwen3-8-max",   # Qwen3.8-Max (top 3.8 tier)
            "type": "empirio",
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "nvfp4-qwen38",
            "url": os.environ.get("NVFP4_URL", ""),   # self-hosted Qwen3.8-27B in NVFP4 quant (URL from env)
            "key": os.environ.get("NVFP4_API_KEY", ""),
            "model": "qwen3.8-27b",
            "type": "deepseek",   # generic OpenAI path (base + /chat/completions)
            "reasoning_field": "reasoning",
        },
        {
            "tag": "zai-glm53",
            "url": "https://api.z.ai/api/paas/v4",   # Zhipu z.ai international, OpenAI-compatible
            "key": os.environ.get("ZAI_API_KEY", ""),
            "model": "glm-5.3",
            "type": "deepseek",   # generic OpenAI path (base + /chat/completions)
            "reasoning_field": "reasoning_content",
        },
        {
            "tag": "zai-glm53-flash",
            "url": "https://api.z.ai/api/paas/v4",
            "key": os.environ.get("ZAI_API_KEY", ""),
            "model": "glm-5.3-flash",
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
        {
            "tag": "openrouter-muse-spark-13c",
            "url": "https://openrouter.ai/api/v1",   # OpenRouter, OpenAI-compatible
            "key": os.environ.get("OPENROUTER_API_KEY", ""),
            "model": "meta/muse-spark-1.3-contributor",   # Meta: Muse Spark 1.3 Contributor, 1M ctx, multimodal
            "type": "deepseek",   # generic OpenAI path (base + /chat/completions)
            "reasoning_field": "reasoning",
        },
        {
            "tag": "empirio-qwen38",
            "url": "https://api.empiriolabs.ai/v1",   # direct host of Qwen3.8-27B; parallel-capable (unlike neuraldeep's 1-req limit), 32K output cap
            "key": os.environ.get("EMPIRIOLABS_API_KEY", ""),
            "model": "qwen3-8-27b",
            "type": "empirio",
            "reasoning_field": "reasoning_content",
        },
    ]

import os

QUEUE_API_KEY = os.getenv("QUEUE_API_KEY", "super-cool-api-key")
QUEUE_URL = os.getenv("QUEUE_URL", "http://100.98.79.5:4949/exchange-batch")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))

# Text classifier: HF / LoRA
AIGT_REPO_ID = os.getenv("AIGT_REPO_ID", "DeepSee-io/qwen_adapters_aigt")
AIGT_SUBDIR_BY_LANG_JSON = os.getenv(
    "AIGT_SUBDIR_BY_LANG_JSON",
    '{"en":"EN/best","es":"ES/best","de":"DE/best","fr":"FR/best","it":"IT/best","ja":"JA/best"}'
)
AIGT_REVISION = os.getenv("AIGT_REVISION", "") or None
AIGT_HF_TOKEN = os.getenv("AIGT_HF_TOKEN", "") or None

# Runtime knobs
AIGT_DEVICE = os.getenv("AIGT_DEVICE", "cuda")
AIGT_CACHE_POLICY = os.getenv("AIGT_CACHE_POLICY", "unload_after_call")
AIGT_MAX_LEN = int(os.getenv("AIGT_MAX_LEN", "500"))
AIGT_BATCH_SIZE = int(os.getenv("AIGT_BATCH_SIZE", "16"))
AIGT_WINDOW_AI_THRESHOLD = float(os.getenv("AIGT_WINDOW_AI_THRESHOLD", "0.5"))
AIGT_PREFER_BF16 = os.getenv("AIGT_PREFER_BF16", "1") not in ("0", "false", "False")

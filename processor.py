import time
import logging
from typing import Any, Dict, List

from processor_utils import pop, push
from processor_config import (
    BATCH_SIZE,
    EMPTY_QUEUE_SLEEP_SECONDS,
    AIGT_REPO_ID,
    AIGT_SUBDIR_BY_LANG_JSON,
    AIGT_REVISION,
    AIGT_HF_TOKEN,
    AIGT_DEVICE,
    AIGT_CACHE_POLICY,
    AIGT_MAX_LEN,
    AIGT_BATCH_SIZE,
    AIGT_WINDOW_AI_THRESHOLD,
    AIGT_PREFER_BF16,
)

from core import TextClassifier, TextInferConfig

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

classifier = None


def _extract_text_items(data: Dict[str, Any]) -> List[str]:
    """
    Adapt this to your queue schema.
    Supported shapes:
      - data["texts"] = [str, ...]
      - data["text"] = str
      - data["chunks"] = [{"text": ...}, ...]
    """
    if isinstance(data.get("texts"), list):
        return [str(x or "").strip() for x in data["texts"]]

    if "text" in data:
        return [str(data["text"] or "").strip()]

    if isinstance(data.get("chunks"), list):
        return [str(c.get("text") or "").strip() for c in data["chunks"]]

    return []


def _extract_lang_for_items(data: Dict[str, Any], n: int) -> List[str]:
    lang = data.get("language_name") or data.get("lang") or "en"
    return [str(lang or "en")] * n


def process_batch(batch_size: int = 1):
    global classifier

    if classifier is None:
        logger.info("Initializing text classifier...")

        cfg = TextInferConfig(
            repo_id = AIGT_REPO_ID,
            subdir_by_lang_json = AIGT_SUBDIR_BY_LANG_JSON,
            revision = AIGT_REVISION,
            hf_token = AIGT_HF_TOKEN,
            device = AIGT_DEVICE,
            cache_policy = AIGT_CACHE_POLICY,
            max_len = AIGT_MAX_LEN,
            batch_size = AIGT_BATCH_SIZE,
            window_ai_threshold = AIGT_WINDOW_AI_THRESHOLD,
            prefer_bf16 = AIGT_PREFER_BF16,
        )

        classifier = TextClassifier(cfg = cfg)
        classifier.load_models()
        logger.info("Text classifier initialized successfully.")

    jobs = pop(batch_size = batch_size)

    if len(jobs) == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    logger.info(f"Processing {len(jobs)} jobs")

    session_ids = [j["data"]["session_id"] for j in jobs]
    target_urls = [j["data"]["target_url"] for j in jobs]
    timestamps = [j["data"]["timestamp"] for j in jobs]

    all_texts: List[str] = []
    all_langs: List[str] = []
    job_text_mapping = []

    for i, job in enumerate(jobs):
        data = job["data"]

        texts = _extract_text_items(data)
        langs = _extract_lang_for_items(data, len(texts))

        job_text_mapping.append({
            "job_index": i,
            "start_index": len(all_texts),
            "end_index": len(all_texts) + len(texts),
        })

        all_texts.extend(texts)
        all_langs.extend(langs)

    logger.info(f"Total texts to classify: {len(all_texts)}")

    prediction_ids = [str(i) for i in range(len(all_texts))]
    preds = classifier.classify_texts_batch(
        all_texts,
        langs = all_langs,
        prediction_ids = prediction_ids,
    )

    results = []

    for mapping in job_text_mapping:
        job_idx = mapping["job_index"]
        start = mapping["start_index"]
        end = mapping["end_index"]

        sid = session_ids[job_idx]
        turl = target_urls[job_idx]
        ts = timestamps[job_idx]

        for pr in preds[start:end]:
            results.append({
                "session_id": sid,
                "target_url": turl,
                "timestamp": ts,

                # Kept for compatibility with existing worker shapes
                "file_key": "",
                "orig_image_url": "",

                # Text outputs
                "lang": pr.get("lang"),
                "prediction_short": pr.get("prediction_short"),
                "prediction_long": pr.get("prediction_long"),
                "ai_probability": pr.get("ai_probability"),
                "human_probability": pr.get("human_probability"),
                "fraction_ai": pr.get("fraction_ai"),
                "n_windows": pr.get("n_windows"),
                "n_ai_segments": pr.get("n_ai_segments"),
                "n_human_segments": pr.get("n_human_segments"),
                "n_tokens": pr.get("n_tokens"),
                "status": pr.get("status"),
                "error": pr.get("error"),
            })

    processed_jobs = [{
        "jobs": [{"id": j["id"], "token": j["token"]} for j in jobs],
        "filename": f"text_results_{int(time.time())}.json",
        "results": results,
    }]

    push(processed_jobs)
    logger.info(f"Pushed {len(results)} results back to queue")


def main():
    logger.info("Starting text classification processor...")

    while True:
        try:
            process_batch(batch_size = BATCH_SIZE)
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
import time
import asyncio
import logging
from typing import Any, Dict, List

import aiohttp

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
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

classifier = None


def _default_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _build_download_link(session_id: str) -> str:
    return f"https://s3.eu-central-2.wasabisys.com/crawl-debug-media/{session_id}/response.txt"


def _extract_lang(data: Dict[str, Any]) -> str:
    lang = data.get("language_code") or "en"
    return str(lang or "en")


async def _fetch_text(
    session: aiohttp.ClientSession,
    session_id: str,
) -> Dict[str, Any]:
    url = _build_download_link(session_id)

    out = {
        "session_id": session_id,
        "download_link": url,
        "text": "",
        "http_status": None,
        "final_url": None,
        "fetch_error": "",
    }

    if not session_id or not isinstance(session_id, str):
        out["fetch_error"] = "Missing or invalid session_id"
        return out

    try:
        timeout = aiohttp.ClientTimeout(total = 15)

        async with session.get(url, timeout = timeout) as resp:
            out["http_status"] = resp.status
            out["final_url"] = str(resp.url)

            if resp.status == 200:
                out["text"] = await resp.text()
            else:
                out["fetch_error"] = f"HTTP {resp.status}"

    except Exception as e:
        out["fetch_error"] = f"{type(e).__name__}: {e!r}"

    return out


async def _fetch_all_texts(
    session_ids: List[str],
    concurrency: int = 50,
) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit = concurrency, ssl = False)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(
        connector = connector,
        headers = _default_headers(),
    ) as session:

        async def sem_fetch(sid: str) -> Dict[str, Any]:
            async with sem:
                return await _fetch_text(session, sid)

        tasks = [sem_fetch(sid) for sid in session_ids]
        return await asyncio.gather(*tasks)


def _init_classifier_if_needed():
    global classifier

    if classifier is not None:
        return

    logger.info("Initializing text classifier.")

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


def process_batch(batch_size: int = 1):
    global classifier

    _init_classifier_if_needed()

    jobs = pop(batch_size = batch_size)

    if len(jobs) == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    logger.info(f"Processing {len(jobs)} jobs")

    session_ids = [str(j["data"].get("session_id") or "") for j in jobs]
    target_urls = [str(j["data"].get("target_url") or "") for j in jobs]
    timestamps = [j["data"].get("timestamp") for j in jobs]
    langs = [_extract_lang(j["data"]) for j in jobs]

    fetched = asyncio.run(_fetch_all_texts(session_ids, concurrency = 50))

    all_texts = [row.get("text") or "" for row in fetched]
    prediction_ids = [str(i) for i in range(len(all_texts))]

    n_fetch_errors = sum(1 for row in fetched if row.get("fetch_error"))
    n_empty_texts = sum(1 for txt in all_texts if not str(txt).strip())

    logger.info(f"Fetched {len(all_texts)} texts")
    logger.info(f"Fetch errors: {n_fetch_errors}; empty texts: {n_empty_texts}")

    preds = classifier.classify_texts_batch(
        all_texts,
        langs = langs,
        prediction_ids = prediction_ids,
    )

    results = []

    for i, pr in enumerate(preds):
        fx = fetched[i]

        results.append({
            "session_id": session_ids[i],
            "target_url": target_urls[i],
            "timestamp": timestamps[i],

            # Legacy compatibility field
            "file_key": "",

            # Relevant input reference for text worker
            "download_link": fx.get("download_link"),
            "http_status": fx.get("http_status"),
            "final_url": fx.get("final_url"),
            "fetch_error": fx.get("fetch_error"),

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
    logger.info("Starting text classification processor.")

    while True:
        try:
            process_batch(batch_size = BATCH_SIZE)
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
import logging
import requests

from processor_config import QUEUE_API_KEY, QUEUE_URL

logger = logging.getLogger(__name__)

QUEUE_KEY = "AI_TEXT"


def pop(batch_size: int = 1) -> list:
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {"key": QUEUE_KEY, "get": batch_size}

    resp = requests.post(url = QUEUE_URL, json = data, headers = headers)
    resp.raise_for_status()

    payload = resp.json()
    return payload["data"]["jobs"]


def push(processed_jobs: list):
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {"key": QUEUE_KEY, "put": processed_jobs}

    resp = requests.post(url = QUEUE_URL, json = data, headers = headers)
    resp.raise_for_status()
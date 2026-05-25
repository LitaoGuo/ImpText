import base64
import concurrent.futures
import io
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _normalize_usage(usage: Any) -> Dict[str, int]:
    if usage is None:
        return _empty_usage()
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _add_usage(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))


class APIEngine:
    """Small OpenAI-compatible multimodal API runner.

    The client reads credentials from API_KEY or OPENAI_API_KEY and an optional
    API_BASE_URL or OPENAI_BASE_URL.
    """

    def __init__(
        self,
        model: str,
        concurrency: int = 4,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("Set API_KEY or OPENAI_API_KEY before running evaluation.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        suffix = image_path.lower()
        mime_type = "image/jpeg"
        if suffix.endswith(".png"):
            mime_type = "image/png"
        elif suffix.endswith(".webp"):
            mime_type = "image/webp"

        # Some providers reject very large inline images. Re-encode only when
        # necessary, preserving normal inputs byte-for-byte.
        if os.path.getsize(image_path) > 3.5 * 1024 * 1024:
            try:
                with Image.open(image_path) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    quality = 85
                    img.save(buf, format="JPEG", quality=quality)
                    while buf.tell() > 3.0 * 1024 * 1024:
                        buf.seek(0)
                        buf.truncate(0)
                        width, height = img.size
                        if width < 100 or height < 100:
                            break
                        img = img.resize((int(width * 0.9), int(height * 0.9)), Image.Resampling.LANCZOS)
                        img.save(buf, format="JPEG", quality=quality)
                    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
            except Exception:
                pass

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type

    def _image_part(self, image_path: str) -> Dict[str, Any]:
        encoded, mime_type = self._encode_image(image_path)
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}

    def _call_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        usage = _empty_usage()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": item["prompt"]},
                                self._image_part(item["image_path"]),
                            ],
                        }
                    ],
                }
                max_tokens = int(item.get("max_tokens", 1024))
                if "gpt-5" in self.model.lower():
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**kwargs)
                _add_usage(usage, _normalize_usage(getattr(response, "usage", None)))
                content = response.choices[0].message.content or ""
                if content.strip():
                    return {
                        "id": item["id"],
                        "text": content,
                        "usage": usage,
                        "success": True,
                        "attempts": attempt + 1,
                        "last_error": None,
                    }
                last_error = "empty response"
            except Exception as exc:
                last_error = str(exc)

            if attempt < self.max_retries - 1:
                time.sleep(2 * (2**attempt))

        return {
            "id": item["id"],
            "text": "",
            "usage": usage,
            "success": False,
            "attempts": self.max_retries,
            "last_error": last_error,
        }

    def generate(self, items: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._call_one, item): item for item in items}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="API inference"):
                item = futures[future]
                yield item, future.result()


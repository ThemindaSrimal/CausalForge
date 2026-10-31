from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
import time
import urllib.error
import urllib.request

from .world import Intervention


class MetaAPIError(RuntimeError):
    def __init__(self, message: str, estimated_cost_usd: float = 0.0):
        super().__init__(message)
        self.estimated_cost_usd = estimated_cost_usd


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class Decision:
    intervention: Intervention
    rationale: str
    response_id: str
    model: str
    latency_seconds: float
    usage: Usage
    api_calls: int


class MuseSparkClient:
    BASE_URL = "https://api.meta.ai/v1"
    MODEL = "muse-spark-1.3"
    INPUT_PER_MILLION = 1.25
    CACHED_INPUT_PER_MILLION = 0.15
    OUTPUT_PER_MILLION = 4.25

    def __init__(self, api_key: str | None = None, timeout: float = 60.0, base_url: str | None = None, transport_retries: int = 2):
        self.api_key = api_key or os.environ.get("MODEL_API_KEY")
        if not self.api_key:
            raise MetaAPIError("MODEL_API_KEY is required for real Muse Spark decisions")
        self.timeout = timeout
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.transport_retries = transport_retries

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        for retry in range(self.transport_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code <= 504
                if not retryable or retry >= self.transport_retries:
                    detail = error.read().decode("utf-8", errors="replace")
                    raise MetaAPIError(f"Meta API HTTP {error.code}: {detail}") from error
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 0.5 * (2**retry) + random.uniform(0, 0.2)
                time.sleep(min(delay, 5.0))
            except (urllib.error.URLError, TimeoutError) as error:
                if retry >= self.transport_retries:
                    reason = getattr(error, "reason", str(error))
                    raise MetaAPIError(f"Meta API request failed after {retry + 1} attempts: {reason}") from error
                time.sleep(0.5 * (2**retry) + random.uniform(0, 0.2))
        raise AssertionError("unreachable")

    def choose_experiment(self, context: dict) -> Decision:
        schema = {
            "type": "function",
            "function": {
                "name": "run_experiment",
                "description": "Choose the next intervention that best reduces uncertainty about the hidden causal rule.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        **{name: {"type": "integer", "enum": [-1, 0, 1]} for name in ("temperature", "pressure", "catalyst", "field")},
                        "rationale": {"type": "string"},
                    },
                    "required": ["temperature", "pressure", "catalyst", "field", "rationale"],
                    "additionalProperties": False,
                },
            },
        }
        messages = [
            {"role": "system", "content": "You are the scientist controlling a real simulator. Infer a sparse linear hidden rule. You must act by making exactly one run_experiment function call; a prose-only answer cannot run the experiment. Use the learner estimates and candidate uncertainties, and avoid duplicate experiments."},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ]
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "tools": [schema],
            "tool_choice": "auto",
            "temperature": 0,
            "reasoning_effort": "low",
            "max_tokens": 8192,
        }
        started = time.monotonic()
        input_tokens = output_tokens = cached = 0
        response_ids: list[str] = []
        body: dict = {}
        last_problem = "no response"
        for attempt in range(1, 4):
            payload["messages"] = messages
            body = self._post(payload)

            raw_usage = body.get("usage", {})
            input_tokens += int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)))
            output_tokens += int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)))
            cached += int(raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            response_ids.append(str(body.get("id", "")))

            try:
                message = body["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                if len(calls) != 1 or calls[0]["function"]["name"] != "run_experiment":
                    excerpt = (message.get("content") or "<empty content>").replace("\n", " ")[:300]
                    finish_reason = body["choices"][0].get("finish_reason", "unknown")
                    last_problem = f"attempt {attempt} returned {len(calls)} run_experiment tool calls; finish_reason={finish_reason!r}; content={excerpt!r}"
                    messages.append({"role": "assistant", "content": message.get("content") or ""})
                    messages.append({"role": "user", "content": "Your previous response did not execute an experiment. Do not explain or answer in prose. Call the provided run_experiment function exactly once now."})
                    continue
                arguments = json.loads(calls[0]["function"]["arguments"])
                intervention = Intervention(**{key: arguments[key] for key in ("temperature", "pressure", "catalyst", "field")})
                rationale = str(arguments["rationale"])
                break
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_problem = f"attempt {attempt} returned an invalid tool call: {error}"
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                messages.append({"role": "user", "content": "The tool arguments were invalid. Call run_experiment exactly once with every required field and valid enum values."})
        else:
            failed_cost = ((input_tokens - cached) * self.INPUT_PER_MILLION + cached * self.CACHED_INPUT_PER_MILLION + output_tokens * self.OUTPUT_PER_MILLION) / 1_000_000
            raise MetaAPIError(
                f"Muse did not provide a valid experiment after 3 real API attempts: {last_problem}",
                round(failed_cost, 8),
            )

        latency = time.monotonic() - started
        cost = ((input_tokens - cached) * self.INPUT_PER_MILLION + cached * self.CACHED_INPUT_PER_MILLION + output_tokens * self.OUTPUT_PER_MILLION) / 1_000_000
        return Decision(intervention, rationale, ",".join(response_ids), str(body.get("model", self.MODEL)), latency, Usage(input_tokens, output_tokens, cached, round(cost, 8)), len(response_ids))

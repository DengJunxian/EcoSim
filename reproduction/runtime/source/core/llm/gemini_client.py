"""Native Gemini calls with one HTTP attempt and explicit, private credentials.

The native usage fields are normalized for the research adapter: output tokens
include candidate and thought tokens exactly once. Raw bodies remain available
for audit; unavailable counts remain unknown.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import stat
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from core.llm.base import LLMResponse, redact_sensitive


GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def native_sdk_version() -> str:
    return f'stdlib-urllib-v1;python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


@dataclass(frozen=True, slots=True)
class GeminiSettings:
    api_key: str = field(default='', repr=False)
    base_url: str = GEMINI_BASE_URL
    timeout_seconds: float = 60.0
    max_retries: int = 0
    default_provider: str = 'gemini'

    def __post_init__(self):
        if self.base_url != GEMINI_BASE_URL:
            raise ValueError('LIVE_ENDPOINT_MISMATCH')
        if self.max_retries != 0:
            raise ValueError('SDK_RETRIES_MUST_BE_ZERO')
        if self.timeout_seconds <= 0:
            raise ValueError('POSITIVE_TIMEOUT_REQUIRED')


def load_gemini_settings(path: str | Path, *, timeout_seconds: float = 60.0) -> GeminiSettings:
    """Read only the selected local env file; do not modify process environment."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise ValueError('LOCAL_CREDENTIAL_FILE_MISSING')
    if stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise ValueError('CREDENTIAL_FILE_REQUIRES_MODE_600')
    lines = [line.strip() for line in target.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    matches = [line.split('=', 1)[1].strip().strip('"').strip("'")
               for line in lines if line.startswith('GEMINI_API_KEY=')]
    if len(matches) != 1:
        raise ValueError('ONE_GEMINI_API_KEY_REQUIRED')
    key = matches[0]
    if not key.startswith(('AQ.', 'AIza')) or any(c.isspace() for c in key):
        raise ValueError('LOCAL_CREDENTIAL_NOT_CONFIGURED')
    return GeminiSettings(api_key=key, timeout_seconds=timeout_seconds)


def normalize_usage(native) -> dict:
    """Normalize native counts, retaining missing or inconsistent usage as unknown.

    Gemini totalTokenCount includes prompt, candidate and thought tokens. If a
    subset is omitted, the total can identify the output count, but no absent
    count is silently converted to zero. Tool use is not enabled by this client.
    """
    if not isinstance(native, dict):
        return {}
    result = {'native_usage_metadata': native}
    def count(name):
        value = native.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('INVALID_NATIVE_USAGE_COUNT')
        return value
    try:
        prompt, candidate, thought, total, cached, tool = (
            count(name) for name in ('promptTokenCount', 'candidatesTokenCount',
                'thoughtsTokenCount', 'totalTokenCount', 'cachedContentTokenCount',
                'toolUsePromptTokenCount'))
        output = None
        if candidate is not None and thought is not None:
            output = candidate + thought
        if total is not None and prompt is not None and not tool:
            residual = total - prompt
            if residual < 0 or (output is not None and residual != output):
                raise ValueError('INCONSISTENT_NATIVE_USAGE_TOTAL')
            if candidate is not None and candidate > residual:
                raise ValueError('INCONSISTENT_NATIVE_CANDIDATE_USAGE')
            if thought is not None and thought > residual:
                raise ValueError('INCONSISTENT_NATIVE_THOUGHT_USAGE')
            output = residual
        if cached is not None and prompt is not None and cached > prompt:
            raise ValueError('INCONSISTENT_NATIVE_CACHE_USAGE')
        result.update(prompt_tokens=prompt, completion_tokens=output,
                      prompt_tokens_details={'cached_tokens': cached},
                      completion_tokens_details={'reasoning_tokens': thought})
    except ValueError as exc:
        result.update(prompt_tokens=None, completion_tokens=None,
                      native_usage_error=str(exc))
    return result


class GeminiClient:
    provider = 'gemini'

    def __init__(self, *, settings: GeminiSettings):
        self.settings = settings

    def _safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        if isinstance(value, str):
            if self.settings.api_key:
                value = value.replace(self.settings.api_key, '[REDACTED_KEY]')
            return redact_sensitive(value)
        return value

    def build_request_snapshot(self, messages, *, model=None, temperature=0.2,
                               max_tokens=None, thinking=None, json_mode=False, **_):
        if not model or not re.fullmatch(r'gemini-[A-Za-z0-9._-]+', model):
            raise ValueError('EXPLICIT_GEMINI_MODEL_REQUIRED')
        if thinking is not True:
            raise ValueError('GEMINI_NATIVE_V1_REQUIRES_THINKING_LOW')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('EXPLICIT_POSITIVE_OUTPUT_CAP_REQUIRED')
        contents, system = [], []
        for message in messages:
            role, text = message.get('role'), message.get('content')
            if type(text) is not str or role not in {'system', 'user', 'assistant'}:
                raise ValueError('NATIVE_TEXT_MESSAGES_REQUIRED')
            parts = [{'text': text}]
            if role == 'system':
                system.extend(parts)
            else:
                contents.append({'role': 'model' if role == 'assistant' else 'user', 'parts': parts})
        if not contents:
            raise ValueError('NATIVE_CONTENT_REQUIRED')
        generation = {'temperature': temperature, 'maxOutputTokens': max_tokens,
                      'thinkingConfig': {'thinkingLevel': 'LOW'}}
        if json_mode:
            generation['responseMimeType'] = 'application/json'
        body = {'contents': contents, 'generationConfig': generation}
        if system:
            body['systemInstruction'] = {'parts': system}
        return {'method': 'POST', 'url': f'{self.settings.base_url}/models/{model}:generateContent',
                'headers': {'Content-Type': 'application/json', 'x-goog-api-key': '[REDACTED_KEY]'},
                'body': body}

    def _complete_sync(self, messages, **kwargs):
        model = kwargs.get('model') or ''
        start = time.perf_counter()
        raw = {}
        try:
            snapshot = self.build_request_snapshot(messages, **kwargs)
            if not self.settings.api_key:
                raise ValueError('LIVE_CREDENTIALS_NOT_CONFIGURED')
            request = Request(snapshot['url'],
                data=json.dumps(snapshot['body'], ensure_ascii=False, allow_nan=False).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'x-goog-api-key': self.settings.api_key},
                method='POST')
            # urllib performs one attempt here; there is no provider fallback or retry loop.
            with build_opener(_NoRedirect()).open(request, timeout=kwargs.get('timeout') or self.settings.timeout_seconds) as response:
                body = response.read().decode('utf-8')
                raw = {'http_status': response.status, 'body_text': self._safe(body)}
            native = json.loads(body)
            if not isinstance(native, dict):
                raise ValueError('NATIVE_RESPONSE_OBJECT_REQUIRED')
            raw['native_response'] = self._safe(native)
            candidates = native.get('candidates') or []
            if len(candidates) > 1:
                raise ValueError('ONE_NATIVE_CANDIDATE_REQUIRED')
            parts = ((candidates[0].get('content') or {}).get('parts') or []) if candidates else []
            text = ''.join(part.get('text', '') for part in parts if not part.get('thought'))
            return LLMResponse(text=self._safe(text), provider=self.provider,
                model=str(native.get('modelVersion') or ''),
                latency_ms=(time.perf_counter()-start)*1000.0,
                usage=normalize_usage(native.get('usageMetadata')), raw=raw, ok=bool(candidates))
        except HTTPError as exc:
            try:
                body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                body = ''
            raw.update(http_status=exc.code, body_text=self._safe(body), error=self._safe(str(exc)))
        except Exception as exc:
            raw.update(error_type=type(exc).__name__, error=self._safe(str(exc)))
        return LLMResponse(text='', provider=self.provider, model=model,
            latency_ms=(time.perf_counter()-start)*1000.0, raw=raw, ok=False)

    async def complete(self, messages, **kwargs):
        return await asyncio.to_thread(self._complete_sync, messages, **kwargs)


class FixedGeminiRouter:
    """Research-only native route. A failure cannot select another provider."""
    allow_offline_fallback = False

    def __init__(self, client):
        self.client = client

    async def complete(self, messages, *, mode, allow_provider_fallback=False, **kwargs):
        if mode != 'gemini' or allow_provider_fallback:
            raise ValueError('GEMINI_FIXED_ROUTE_REQUIRED')
        response = await self.client.complete(messages, **kwargs)
        if not response.ok:
            raise RuntimeError('Gemini native request failed; no fallback is configured')
        return response

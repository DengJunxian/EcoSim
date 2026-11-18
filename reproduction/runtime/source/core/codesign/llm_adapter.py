"""Fixed-identity, raw-first calls through the existing Router and provider clients."""
from __future__ import annotations

import asyncio
import importlib.metadata
import math
from pathlib import Path
from typing import Literal
from pydantic import Field

from core.experiments.strict_types import StrictResearchModel, strict_json_loads
from core.experiments.identity import research_hash
from core.experiments.archive import durable_json
from .schema import ModelCallConfig


class DesignAction(StrictResearchModel):
    action: Literal['propose', 'query_information', 'stop']
    candidate_id: str | None
    information_id: str | None
    claimed_agreement: bool
    rationale: str


class DesignCallError(RuntimeError):
    def __init__(self,code,raw_refs):
        self.code,self.raw_refs=code,raw_refs
        super().__init__(code)


def _safe_raw(value):
    from core.llm.base import redact_sensitive
    if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
    if isinstance(value,dict):
        return {str(k):('[REDACTED]' if str(k).lower() in {'api_key','authorization','access_token','x-goog-api-key'} else _safe_raw(v)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe_raw(v) for v in value]
    if isinstance(value,str):return redact_sensitive(value)
    if type(value) is float and not math.isfinite(value):return {'non_json_number':str(value)}
    if value is None or type(value) in (bool,int,float):return value
    return redact_sensitive(str(value))


def extract_usage(usage):
    """Provider prompt/completion totals include cache/reasoning detail, not added twice."""
    if not isinstance(usage,dict):return None,None,{'raw_usage':_safe_raw(usage),'valid':False}
    if usage.get('native_usage_error'):raise ValueError('INVALID_NATIVE_USAGE')
    incoming,outgoing=usage.get('prompt_tokens'),usage.get('completion_tokens')
    for x in (incoming,outgoing):
        if x is not None and (type(x) is not int or x<0):raise ValueError('INVALID_USAGE_COUNT')
    prompt=usage.get('prompt_tokens_details') or {}
    completion=usage.get('completion_tokens_details') or {}
    cached=prompt.get('cached_tokens',usage.get('prompt_cache_hit_tokens'))
    reasoning=completion.get('reasoning_tokens')
    for value,total in [(cached,incoming),(reasoning,outgoing)]:
        if value is not None and (type(value) is not int or value<0 or (total is not None and value>total)):
            raise ValueError('INVALID_USAGE_BREAKDOWN')
    return incoming,outgoing,{'cached_tokens':cached,'reasoning_tokens':reasoning,
        'detail_semantics':'subsets_of_prompt_and_completion_totals','raw_usage':_safe_raw(usage),'valid':True}


class _CaptureTransport:
    def __init__(self,client,adapter):self.client,self.adapter=client,adapter

    async def complete(self,messages,**kwargs):
        from core.llm.base import redact_sensitive
        try:
            if hasattr(self.client,'build_request_snapshot'):
                snapshot=self.client.build_request_snapshot(messages,**kwargs)
                self.adapter._capture_request(snapshot)
            response=await self.client.complete(messages,**kwargs)
        except Exception as exc:
            self.adapter._capture({'kind':'transport_exception','error_type':type(exc).__name__,
                                   'error':redact_sensitive(exc),'text':None,'usage':None})
            raise
        # Executed before LLMRouter's ok/fallback guard, parsing or local semantic guards.
        self.adapter._capture({'kind':'provider_response','text':response.text,'provider':response.provider,
            'model':response.model,'ok':response.ok,'latency_ms':float(response.latency_ms),
            'usage':response.usage,'raw':_safe_raw(response.raw)})
        self.adapter.response=response
        return response


class DesignAgentClient:
    def __init__(self,config:ModelCallConfig,ledger,output,*,transport=None,allow_live=False,settings=None,spend_ledger=None):
        self.config,self.ledger,self.output=config,ledger,Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.allow_live=allow_live;self.calls=[];self.lock=asyncio.Lock();self.response=None
        self.spend_ledger=spend_ledger
        if config.source=='live' and not allow_live:raise ValueError('LIVE_CALLS_USER_DEFERRED')
        if config.source=='rule':raise ValueError('RULE_METHODS_DO_NOT_USE_LLM_TRANSPORT')
        if config.source in {'mock','recorded'} and transport is None:raise ValueError('EXPLICIT_NONLIVE_TRANSPORT_REQUIRED')
        if config.source=='live' and transport is not None:raise ValueError('LIVE_CANNOT_USE_INJECTED_MOCK_TRANSPORT')
        # Imports are late: rule-only methods never instantiate a provider or read its environment.
        if config.provider=='gemini':
            from core.llm.gemini_client import GeminiSettings,GeminiClient,FixedGeminiRouter,native_sdk_version
            if native_sdk_version()!=config.sdk_version:raise ValueError('SDK_VERSION_MISMATCH')
            if settings is None:settings=GeminiSettings(timeout_seconds=config.timeout_seconds)
            if not isinstance(settings,GeminiSettings):raise ValueError('GEMINI_SETTINGS_REQUIRED')
            if settings.max_retries!=0:raise ValueError('SDK_RETRIES_MUST_BE_ZERO')
            if config.source=='live':
                if not settings.api_key:raise ValueError('LIVE_CREDENTIALS_NOT_CONFIGURED')
                if spend_ledger is None:raise ValueError('LIVE_STUDY_SPEND_LEDGER_REQUIRED')
                if config.max_request_attempts!=1:raise ValueError('LIVE_PILOT_RETRIES_MUST_BE_ZERO')
                transport=GeminiClient(settings=settings)
            self.settings=settings;self.transport=transport
            self.router=FixedGeminiRouter(_CaptureTransport(transport,self))
        else:
            from core.llm.config import LLMSettings
            from core.llm.router import LLMRouter
            from core.llm.deepseek_client import DeepSeekClient
            from core.llm.zhipu_client import ZhipuClient
            if importlib.metadata.version('openai')!=config.sdk_version:raise ValueError('SDK_VERSION_MISMATCH')
            if settings is None:
                settings=LLMSettings(deepseek_api_key='',zhipu_api_key='',default_provider=config.provider,
                    timeout_seconds=config.timeout_seconds,max_retries=0,
                    deepseek_base_url='https://api.deepseek.com',zhipu_base_url='https://open.bigmodel.cn/api/paas/v4')
            if settings.max_retries!=0:raise ValueError('SDK_RETRIES_MUST_BE_ZERO')
            if config.source=='live':
                key=settings.deepseek_api_key if config.provider=='deepseek' else settings.zhipu_api_key
                if not key:raise ValueError('LIVE_CREDENTIALS_NOT_CONFIGURED')
                if spend_ledger is None:raise ValueError('LIVE_STUDY_SPEND_LEDGER_REQUIRED')
                if config.max_request_attempts!=1:raise ValueError('LIVE_PILOT_RETRIES_MUST_BE_ZERO')
                if settings.deepseek_base_url!='https://api.deepseek.com':raise ValueError('LIVE_ENDPOINT_MISMATCH')
                transport=(DeepSeekClient if config.provider=='deepseek' else ZhipuClient)(settings=settings)
            self.settings=settings;self.transport=transport
            captured=_CaptureTransport(transport,self)
            self.router=LLMRouter(settings=settings,deepseek_client=captured if config.provider=='deepseek' else None,
                zhipu_client=captured if config.provider=='zhipu' else None,allow_offline_fallback=False)
        durable_json(self.output/'configuration.json',config.model_dump(mode='json'))

    def _capture_request(self,body):
        path=self.output/(self.current_attempt+'-native-request.json')
        durable_json(path,{'source':self.config.source,'attempt_id':self.current_attempt,**_safe_raw(body)})
        self.calls[-1]['native_request_ref']=str(path.resolve())

    def _capture(self,body):
        path=self.output/(self.current_attempt+'-response.json')
        durable_json(path,{'source':self.config.source,'attempt_id':self.current_attempt,**_safe_raw(body)})
        self.current_raw_ref=str(path.resolve());self.calls[-1]['response_ref']=self.current_raw_ref

    async def request(self,messages:list[dict],*,request_id:str,response_model=DesignAction,semantic_guard=None):
        async with self.lock:
            return await self._request(messages,request_id=request_id,response_model=response_model,semantic_guard=semantic_guard)

    async def _request(self,messages,*,request_id,response_model,semantic_guard):
        refs=[]
        # Validate input identity and prevent the visible prompt exceeding the common reserved cap.
        request={'messages':messages,'configuration':self.config.model_dump(mode='json'),'request_id':request_id,
                 'response_schema':response_model.model_json_schema()}
        request_hash=research_hash(request)
        encoded=__import__('json').dumps(messages,ensure_ascii=False).encode('utf-8')
        if len(encoded)+(1024 if self.config.source=='live' else 0)>self.config.input_reservation_tokens:raise ValueError('PROMPT_EXCEEDS_REGISTERED_BYTE_CONSERVATIVE_INPUT_CAP')
        if any(c['request_id']==request_id for c in self.calls):raise ValueError('REQUEST_ID_REUSED')
        for attempt in range(self.config.max_request_attempts):
            attempt_id=f'{len(self.calls)+1:05d}-{research_hash({"request":request_hash,"attempt":attempt})[:12]}'
            self.ledger.reserve_llm(attempt_id,input_cap=self.config.input_reservation_tokens,
                output_cap=self.config.max_output_tokens,request_hash=request_hash)
            spend_id=self.ledger.episode+':'+attempt_id
            if self.spend_ledger:
                self.spend_ledger.reserve(spend_id,self.config.input_reservation_tokens,self.config.max_output_tokens,request_hash)
            self.current_attempt=attempt_id;self.current_raw_ref=None;self.response=None
            request_path=self.output/(attempt_id+'-request.json')
            durable_json(request_path,{'source':self.config.source,'attempt_id':attempt_id,**_safe_raw(request)})
            self.calls.append({'request_id':request_id,'attempt_id':attempt_id,'request_ref':str(request_path.resolve()),'request_hash':request_hash})
            error=None
            try:
                response=await self.router.complete(messages,mode=self.config.provider,model=self.config.model_id,
                    temperature=self.config.temperature,max_tokens=self.config.max_output_tokens,
                    timeout=self.config.timeout_seconds,thinking=self.config.thinking,json_mode=True,
                    task_type='research_policy_design',allow_provider_fallback=False)
            except Exception as exc:
                error=exc;response=self.response
            if self.current_raw_ref:refs.append(self.current_raw_ref)
            try:incoming,outgoing,breakdown=extract_usage({} if response is None else response.usage)
            except ValueError:
                incoming,outgoing,breakdown=None,None,{'valid':False,'raw_usage':_safe_raw(response.usage)}
            settlement=self.ledger.settle_llm(attempt_id,input_tokens=incoming,output_tokens=outgoing,
                breakdown=breakdown,status='transport_failed' if error else 'received')
            if self.spend_ledger:
                self.calls[-1]['spend_settlement']=self.spend_ledger.settle(spend_id,incoming,outgoing,breakdown.get('cached_tokens'))
            self.calls[-1]['settlement']=settlement;durable_json(self.output/'call_index.json',self.calls)
            if error:
                if attempt+1<self.config.max_request_attempts:continue
                raise DesignCallError('TRANSPORT_FAILED',refs) from error
            if (response.provider!=self.config.provider or response.model not in
                    [self.config.model_id]+self.config.model_aliases.get(self.config.model_id,[])):
                if self.spend_ledger:self.spend_ledger.block('PROVIDER_MODEL_IDENTITY_MISMATCH')
                raise DesignCallError('PROVIDER_MODEL_IDENTITY_MISMATCH',refs)
            if self.spend_ledger and self.spend_ledger.snapshot()['blocked_reason']:
                raise DesignCallError('LIVE_USAGE_RECONCILIATION_REQUIRED',refs)
            if not breakdown.get('valid') or settlement['details']['over_reservation']:
                raise DesignCallError('USAGE_INVALID_OR_CAP_EXCEEDED',refs)
            try:
                value=strict_json_loads(response.text)
                if type(value) is not dict:raise ValueError('Object required')
                result=response_model.model_validate(value)
                if semantic_guard is not None:semantic_guard(result)
            except (ValueError,TypeError) as exc:
                # A repair is a new explicit request, with a new charge; no auto repair.
                raise DesignCallError('INVALID_RESPONSE',refs) from exc
            self.calls[-1]['parsed_hash']=research_hash(result)
            durable_json(self.output/'call_index.json',self.calls)
            return result
        raise AssertionError('Finite retry loop exhausted without a receipt')

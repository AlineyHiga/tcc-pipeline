"""LLM client abstraction for OpenAI and local models."""
import os
import logging
from typing import Optional
from openai import OpenAI
from .logging_setup import log_event, save_artifact, sha256_text, get_run_artifacts_dir, should_sample_llm


def get_llm_client():
    """Get configured LLM client."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class OpenAIClient:
    """OpenAI-compatible client (works with Ollama too)."""
    
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.max_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "4096"))
    
    def generate(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        """Generate text completion with JSON mode and grammar support."""
        import time
        logger = logging.getLogger(__name__)
        
        try:
            # Build messages
            if messages:
                msgs = messages
                full_prompt = "\n".join([m.get("content", "") for m in msgs])
            else:
                msgs = [{"role": "user", "content": prompt}]
                full_prompt = prompt or ""
            
            # Build payload
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", 0.0)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            stop = kwargs.get("stop", [])
            seed = kwargs.get("seed")
            top_p = kwargs.get("top_p")
            
            payload = {
                "model": model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add optional parameters
            if stop:
                payload["stop"] = stop
            if seed is not None:
                payload["seed"] = seed
            if top_p is not None:
                payload["top_p"] = top_p
            
            # Grammar/JSON mode detection
            response_format = kwargs.get("response_format")
            grammar = kwargs.get("grammar")
            has_grammar = bool(grammar)
            grammar_sha256 = sha256_text(str(grammar)) if grammar else None
            
            if response_format:
                payload["response_format"] = response_format
            if grammar:
                payload["grammar"] = grammar
            
            # Log request
            prompt_hash = sha256_text(full_prompt)
            preview = full_prompt[:1000] if full_prompt else ""
            
            artifact_path = ""
            if should_sample_llm():
                artifact_path = save_artifact(
                    f"{get_run_artifacts_dir()}/llm",
                    f"prompt_{prompt_hash}.txt",
                    full_prompt
                )
            
            request_log = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_sha256": prompt_hash,
                "preview": preview,
                "artifact_path": artifact_path
            }
            
            if seed is not None:
                request_log["seed"] = seed
            if stop:
                request_log["stop"] = stop
            if top_p is not None:
                request_log["top_p"] = top_p
            if has_grammar:
                request_log["has_grammar"] = True
                request_log["grammar_sha256"] = grammar_sha256
            if response_format:
                request_log["response_format"] = response_format
            
            log_event("llm.request", **request_log)
            
            # Make API call with timing
            start_time = time.time()
            response = self.client.chat.completions.create(**payload)
            latency_ms = int((time.time() - start_time) * 1000)
            
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            # Extract usage if available
            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage else None
            completion_tokens = getattr(usage, 'completion_tokens', None) if usage else None
            
            # Detect grammar/parsing results
            applied_grammar = has_grammar and content
            stopped_on = None
            if stop and finish_reason == "stop":
                for stop_seq in stop:
                    if stop_seq in content:
                        stopped_on = stop_seq
                        break
            
            # Parse result detection
            parsed = False
            parse_error = None
            if response_format and response_format.get("type") == "json_object":
                try:
                    import json
                    json.loads(content)
                    parsed = True
                except json.JSONDecodeError as e:
                    parse_error = str(e)[:100]  # Short message
            
            # Log response
            response_hash = sha256_text(content)
            response_preview = content[:1000] if content else ""
            
            response_artifact_path = ""
            if should_sample_llm():
                response_artifact_path = save_artifact(
                    f"{get_run_artifacts_dir()}/llm",
                    f"response_{response_hash}.txt",
                    content
                )
            
            response_log = {
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
                "content_sha256": response_hash,
                "preview": response_preview,
                "artifact_path": response_artifact_path
            }
            
            if prompt_tokens is not None:
                response_log["prompt_tokens"] = prompt_tokens
            if completion_tokens is not None:
                response_log["completion_tokens"] = completion_tokens
            if applied_grammar:
                response_log["applied_grammar"] = True
            if stopped_on:
                response_log["stopped_on"] = stopped_on
            if response_format and response_format.get("type") == "json_object":
                response_log["parsed"] = parsed
                if parse_error:
                    response_log["parse_error"] = parse_error
            
            log_event("llm.response", **response_log)
            
            return content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            log_event("llm.error", error=str(e), model=kwargs.get("model", self.model))
            return ""
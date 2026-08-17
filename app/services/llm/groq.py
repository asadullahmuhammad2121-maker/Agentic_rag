"""Groq LLM provider implementation."""

from __future__ import annotations

from typing import Any, cast

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq
from groq.types.chat import ChatCompletionMessageParam

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.services.llm.base import LLMService

logger = get_logger(__name__)


class GroqLLMService(LLMService):
    """Groq-backed LLM service for grounded answer generation."""

    def __init__(self, settings: Settings, client: Groq | None = None) -> None:
        self._settings = settings
        self._model = settings.groq_model
        try:
            self._client = client or Groq(
                api_key=settings.groq_api_key.get_secret_value(),
                timeout=settings.groq_timeout_seconds,
            )
        except Exception as exc:
            logger.error(
                "groq_client_init_failed",
                extra={"operation": "init", "provider": "groq", "error_type": type(exc).__name__},
            )
            raise ProviderError(
                "Failed to initialize Groq client",
                provider="groq",
            ) from exc

        logger.info(
            "groq_llm_service_initialized",
            extra={
                "operation": "init",
                "provider": "groq",
                "model": self._model,
            },
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def health_check(self) -> bool:
        """Confirm the client and model configuration are present."""
        api_key = self._settings.groq_api_key.get_secret_value()
        healthy = bool(api_key) and bool(self._model) and self._client is not None
        logger.debug(
            "groq_health_check",
            extra={"operation": "health_check", "provider": "groq", "healthy": healthy},
        )
        return healthy

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion.

        Optional kwargs:
        - system_prompt: str
        - temperature: float
        - max_tokens: int
        """
        system_prompt = kwargs.get("system_prompt")
        temperature = float(kwargs.get("temperature", self._settings.llm_temperature))
        max_tokens = int(kwargs.get("max_tokens", self._settings.llm_max_tokens))

        messages: list[ChatCompletionMessageParam] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(
            "groq_generation_started",
            extra={
                "operation": "generate",
                "provider": "groq",
                "model": self._model,
                "message_count": len(messages),
                "user_prompt_length": len(prompt),
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        try:
            completion = self._create_completion(
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            content, finish_reason = self._extract_message_content(completion)
            if (not content or not content.strip()) and finish_reason == "length":
                retry_tokens = min(max_tokens * 2, 8192)
                logger.warning(
                    "groq_empty_content_retry",
                    extra={
                        "operation": "generate",
                        "provider": "groq",
                        "model": self._model,
                        "finish_reason": finish_reason,
                        "original_max_completion_tokens": max_tokens,
                        "retry_max_completion_tokens": retry_tokens,
                    },
                )
                completion = self._create_completion(
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=retry_tokens,
                )
                content, finish_reason = self._extract_message_content(completion)
        except APITimeoutError as exc:
            logger.error(
                "groq_generation_timeout",
                extra={
                    "operation": "generate",
                    "provider": "groq",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Groq request timed out",
                provider="groq",
                details={"reason": "timeout"},
            ) from exc
        except APIConnectionError as exc:
            logger.error(
                "groq_generation_connection_error",
                extra={
                    "operation": "generate",
                    "provider": "groq",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Unable to reach Groq",
                provider="groq",
                details={"reason": "connection_error"},
            ) from exc
        except APIStatusError as exc:
            status_code = cast("int | None", getattr(exc, "status_code", None))
            logger.error(
                "groq_generation_api_error",
                extra={
                    "operation": "generate",
                    "provider": "groq",
                    "error_type": type(exc).__name__,
                    "status_code": status_code,
                },
            )
            if status_code == 429:
                message = "Groq rate limit exceeded — retry shortly or switch models"
            elif status_code == 401:
                message = "Groq API key is invalid or missing"
            elif status_code == 404:
                message = "Groq model not found or deprecated — update GROQ_MODEL"
            else:
                message = "Groq API request failed"
            raise ProviderError(
                message,
                provider="groq",
                details={
                    "reason": "api_error",
                    "status_code": status_code,
                },
            ) from exc
        except Exception as exc:
            logger.error(
                "groq_generation_failed",
                extra={
                    "operation": "generate",
                    "provider": "groq",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Groq generation failed",
                provider="groq",
                details={"reason": "unexpected_error", "error_type": type(exc).__name__},
            ) from exc

        if not content or not str(content).strip():
            raise ProviderError(
                "Groq returned an empty response (completion token budget exhausted)",
                provider="groq",
                details={
                    "reason": "empty_response",
                    "finish_reason": finish_reason,
                },
            )

        answer = str(content).strip()
        logger.info(
            "groq_generation_completed",
            extra={
                "operation": "generate",
                "provider": "groq",
                "model": self._model,
                "answer_length": len(answer),
            },
        )
        return answer

    def _is_reasoning_model(self) -> bool:
        return "gpt-oss" in self._model

    def _create_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        max_completion_tokens: int,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if self._is_reasoning_model():
            request_kwargs["reasoning_effort"] = "low"
        return self._client.chat.completions.create(**request_kwargs)

    @staticmethod
    def _extract_message_content(completion: Any) -> tuple[str | None, str | None]:
        choice = completion.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        if content is None:
            return None, finish_reason
        return str(content), finish_reason

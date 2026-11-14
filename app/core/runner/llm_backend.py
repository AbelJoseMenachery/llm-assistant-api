import logging
import os
from typing import List

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion


class LLMBackend:
    """
    openai chat 接口封装
    """

    def __init__(self, base_url: str, api_key) -> None:
        self.base_url = base_url + "/" if base_url else None
        self.api_key = api_key
        timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "90"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
        client_class = OpenAI
        # Langfuse's OpenAI wrapper is optional. When configured, it records
        # generations without changing the OpenAI-compatible provider contract.
        if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                from langfuse.openai import OpenAI as LangfuseOpenAI
                client_class = LangfuseOpenAI
            except ImportError:
                logging.warning("Langfuse credentials are set but the SDK is not installed; tracing is disabled")
        self.client = client_class(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def run(
        self,
        messages: List,
        model: str,
        tools: List = None,
        tool_choice="auto",
        stream=False,
        stream_options=None,
        extra_body=None,
        temperature=None,
        top_p=None,
        response_format=None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        chat_params = {
            "messages": messages,
            "model": model,
            "stream": stream,
        }
        if extra_body:
            model_params = extra_body.get("model_params")
            if model_params:
                if "n" in model_params:
                    raise ValueError("n is not allowed in model_params")
                chat_params.update(model_params)
        if stream_options:
            if isinstance(stream_options, dict):
                if "include_usage" in stream_options:
                    chat_params["stream_options"] = {"include_usage": bool(stream_options["include_usage"])}
        if temperature:
            chat_params["temperature"] = temperature
        if top_p:
            chat_params["top_p"] = top_p
        if tools:
            chat_params["tools"] = tools
            chat_params["tool_choice"] = tool_choice if tool_choice else "auto"
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            chat_params["response_format"] = {"type": "json_object"}
        # Never log prompts, completions, tool arguments, or credentials.
        logging.info(
            "LLM request model=%s stream=%s tools=%s message_count=%s",
            model, stream, bool(tools), len(messages),
        )
        response = self.client.chat.completions.create(**chat_params)
        logging.info("LLM response received model=%s stream=%s", model, stream)
        return response

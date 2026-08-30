"""LLMClient translation layer: message/tool encoding and response decoding are pure
functions tested without litellm. The real-provider smoke tests are opt-in."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from history.records import Message, ToolCall, ToolResult
from llm.client import Completion, ToolSchema, Usage, decode_response, encode_messages, encode_tools


def test_encode_fans_tool_results_out_per_call():
    message = Message(
        role="tool",
        tool_results=(
            ToolResult("c1", '{"written": "mem_a"}'),
            ToolResult("c2", '{"results": []}'),
        ),
    )
    encoded = encode_messages([Message(role="user", content="hi"), message])
    assert encoded[0] == {"role": "user", "content": "hi"}
    assert encoded[1] == {"role": "tool", "tool_call_id": "c1", "content": '{"written": "mem_a"}'}
    assert encoded[2] == {"role": "tool", "tool_call_id": "c2", "content": '{"results": []}'}


def test_encode_assistant_tool_calls_json_encodes_arguments():
    message = Message(
        role="assistant",
        content="",
        tool_calls=(ToolCall("c1", "memory_search", {"query": "fox"}),),
    )
    (encoded,) = encode_messages([message])
    assert encoded["tool_calls"][0]["function"] == {
        "name": "memory_search",
        "arguments": '{"query": "fox"}',
    }


def test_encode_tools_wraps_function_schemas():
    schema = ToolSchema(name="finish", description="end", parameters={"type": "object"})
    assert encode_tools([schema]) == [
        {
            "type": "function",
            "function": {"name": "finish", "description": "end", "parameters": {"type": "object"}},
        }
    ]


def fake_response(**overrides):
    defaults = dict(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="finish", arguments='{"decision": "sleep"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_decode_response_normalizes_everything():
    completion = decode_response(fake_response(), cost_usd=0.012)
    assert isinstance(completion, Completion)
    assert completion.text == "hello"
    assert completion.tool_calls == (ToolCall("c1", "finish", {"decision": "sleep"}),)
    assert completion.usage == Usage(input_tokens=100, output_tokens=20, cache_read_tokens=64, cost_usd=0.012)
    assert completion.usage.total_tokens == 120
    assert completion.stop_reason == "tool_calls"


def test_decode_tolerates_bad_arguments_and_missing_usage():
    response = fake_response(usage=None)
    response.choices[0].message.tool_calls[0].function.arguments = "not json {"
    completion = decode_response(response, cost_usd=0.0)
    assert completion.tool_calls[0].arguments == {}
    assert completion.usage.total_tokens == 0


def test_decode_no_tool_calls():
    response = fake_response()
    response.choices[0].message.tool_calls = None
    response.choices[0].message.content = "plain answer"
    completion = decode_response(response, cost_usd=0.0)
    assert completion.tool_calls == ()
    assert completion.text == "plain answer"


# ------------------------------------------------------------------ integration

requires_key = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="no provider API key in the environment",
)


@pytest.mark.integration
@requires_key
def test_real_completion_smoke():
    from llm.client import LLMClient

    model = os.environ.get("HARNESS_COMPLETION_MODEL", "anthropic/claude-opus-5")
    client = LLMClient(model, max_tokens=64)
    completion = client.complete([Message(role="user", content="Reply with the single word: ok")])
    assert completion.text.strip()
    assert completion.usage.total_tokens > 0


@pytest.mark.integration
@requires_key
def test_real_embedding_smoke():
    from llm.embedder import LiteLLMEmbedder

    model = os.environ.get("HARNESS_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    dim = int(os.environ.get("HARNESS_EMBEDDING_DIM", "1536"))
    embedder = LiteLLMEmbedder(model, dim)
    vectors = embedder.embed(["the red fox", "a red fox", "sqlite economics"])
    from memory.store import _cosine

    assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])

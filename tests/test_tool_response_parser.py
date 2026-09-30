from __future__ import annotations

from rl.tool_protocol import configure_tool_response_parser, parse_tool_response_text


def test_executes_only_one_closed_tool_call_per_turn() -> None:
    parsed = parse_tool_response_text(
        '<tool_call>{"name":"text_search","arguments":{"query":"Sydney"}}</tool_call>'
        '<tool_call>{"name":"visit","arguments":{"url":"https://example.com"}}</tool_call>'
    )

    assert [call["function"]["name"] for call in parsed["tool_calls"]] == ["text_search"]
    assert parsed["content"] == ""


def test_recovers_only_first_call_when_closing_tag_is_missing() -> None:
    parsed = parse_tool_response_text(
        '<tool_call>\n{"name":"text_search","arguments":{"query":"Sydney"}}\n'
        '<tool_call>\n{"name":"image_search","arguments":{"image":"image_0"}}'
    )

    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["function"]["name"] == "text_search"


def test_keeps_truncated_invalid_call_as_content() -> None:
    text = '<tool_call>{"name":"text_search","arguments":{"query":"Syd'
    assert parse_tool_response_text(text) == {"role": "assistant", "content": text}


def test_parses_final_answer_as_content() -> None:
    assert parse_tool_response_text("The answer is Sydney.<|im_end|>") == {
        "role": "assistant",
        "content": "The answer is Sydney.",
    }


def test_configured_tokenizer_decodes_ids() -> None:
    class Tokenizer:
        def decode(self, response):
            assert response == [1, 2, 3]
            return '<tool_call>{"name":"text_search","arguments":{"query":"Sydney"}}'

    tokenizer = configure_tool_response_parser(Tokenizer())
    parsed = tokenizer.parse_response([1, 2, 3], prefix=[9])

    assert parsed["tool_calls"][0]["function"]["arguments"] == {"q": "Sydney"}


def test_normalizes_learned_image_search_arguments() -> None:
    parsed = parse_tool_response_text(
        '<tool_call>{"name":"image_search","arguments":'
        '{"image":"Main Street in Red Lodge","topK":5}}</tool_call>'
    )

    assert parsed["tool_calls"][0]["function"] == {
        "name": "image_search",
        "arguments": {"url": "Main Street in Red Lodge"},
    }


def test_normalizes_legacy_aliases() -> None:
    parsed = parse_tool_response_text(
        '<tool_call>{"name":"local_search","arguments":'
        '{"q":"Red Lodge","hl":"en","top_k":3}}</tool_call>'
    )

    assert parsed["tool_calls"][0]["function"] == {
        "name": "text_search",
        "arguments": {"q": "Red Lodge", "hl": "en", "top_k": 3},
    }

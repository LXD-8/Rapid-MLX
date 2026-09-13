# SPDX-License-Identifier: Apache-2.0
r"""A parameter value keeps the indentation of its FIRST line (#3401).

Reported by andreiiliuta against 0.14.1: every code edit a coding agent made
through Rapid-MLX lost the leading whitespace of the opening line of
``new_string``/``old_string``, while lines 2..n kept theirs. Four of six
otherwise-correct Python files in a real agent transcript did not compile.

The cause was ``split_marked_parameters`` returning ``segmented[0].strip()``.
On this wire exactly ONE newline per side is markup -- the Qwen3-Coder chat
template renders ``<parameter=NAME>\n`` + value + ``\n</parameter>\n`` -- so
``.strip()`` also ate the payload indentation of line 1. The asymmetry is why
it survived so long: a single-line value looked merely "trimmed", and a
multi-line value looked correct everywhere except its first line.

Both mature engines serving this format drop one newline per side and nothing
else, in streaming and non-streaming alike:

  * vLLM ``vllm/parser/qwen3.py::_trim_wrapping_newlines``
  * SGLang ``srt/function_call/qwen3_coder_detector.py`` ("Remove prefixing
    and trailing \n"), in both ``detect_and_parse`` and the streaming lexer.

``tests/test_tool_call_value_fidelity.py`` guards the same class of defect for
INTERIOR whitespace; this file guards the edges.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_call_scan import split_marked_parameters, trim_wrapping_newlines
from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

PARAM_OPENER = r"<parameter=([^>]+)>"
PARAM_CLOSER = "</parameter>"

# The exact edit shape from the report: 8-space indent on line 1, deeper
# indent on line 2. Line 2 always survived; line 1 did not.
INDENTED_BODY = "        if is_literal_job(job):\n            raise ValueError()"

EDIT_REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                },
            },
        }
    ]
}


def _wire(**params: str) -> str:
    """Render params exactly as tool_chat_template_qwen3coder.jinja does."""
    body = "".join(
        f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in params.items()
    )
    return f"<tool_call>\n<function=Edit>\n{body}</function>\n</tool_call>"


def _arguments(text: str, request: dict | None = EDIT_REQUEST) -> dict:
    parser = Qwen3CoderToolParser(tokenizer=None)
    parser.reset()
    result = parser.extract_tool_calls(text, request=request)
    assert result.tools_called, f"no tool call recovered from {text!r}"
    return json.loads(result.tool_calls[0]["arguments"])


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # One newline per side is markup, and only one.
        ("\n        return 2\n", "        return 2"),
        # A value that legitimately opens or closes on a blank line keeps it:
        # only the single wrapping newline belongs to the wire.
        ("\n\nleading blank kept\n\n", "\nleading blank kept\n"),
        # \r\n is NOT a two-byte wrapper: under the template's LF framing a
        # payload ending in \r is indistinguishable from CRLF markup, so the
        # byte is kept rather than guessed away (codex review, round 1).
        ("\n  x  \r\n", "  x  \r"),
        ("\r\ntext\r\n", "\r\ntext\r"),
        # Nothing to trim: an inline value is returned byte-identical, which
        # is what keeps single-line scalars working.
        ("42", "42"),
        (" spaced ", " spaced "),
        # Degenerate inputs must not underflow into the payload.
        ("", ""),
        ("\n", ""),
        # Both newlines are the wire's own: an empty value, not a newline.
        ("\n\n", ""),
    ],
)
def test_trim_wrapping_newlines(raw: str, expected: str) -> None:
    assert trim_wrapping_newlines(raw) == expected


def test_split_marked_parameters_keeps_first_line_indentation() -> None:
    block = f"<parameter=new_string>\n{INDENTED_BODY}\n</parameter>\n"
    assert split_marked_parameters(block, PARAM_OPENER, PARAM_CLOSER) == [
        ("new_string", INDENTED_BODY)
    ]


def test_parameter_names_are_still_stripped() -> None:
    """Names are identifiers, not payload -- that half of the old behaviour
    was correct and must not regress with the value fix."""
    block = "<parameter= spaced_name >\nv\n</parameter>\n"
    parsed = split_marked_parameters(block, PARAM_OPENER, PARAM_CLOSER)
    assert parsed == [("spaced_name", "v")]


# ---------------------------------------------------------------------------
# The reported failure, end to end
# ---------------------------------------------------------------------------


def test_edit_arguments_keep_first_line_indentation() -> None:
    args = _arguments(
        _wire(
            file_path="/tmp/x.py",
            old_string="        return 1",
            new_string=INDENTED_BODY,
        )
    )
    assert args["old_string"] == "        return 1"
    assert args["new_string"] == INDENTED_BODY
    # The report's actual symptom: line 1 de-indented while line 2 kept its
    # indentation. Assert the asymmetry itself, not just the whole value.
    first, second = args["new_string"].split("\n")
    assert first.startswith("        "), "first line lost its indentation"
    assert second.startswith("            "), "second line lost its indentation"


def test_edited_python_still_compiles() -> None:
    """The consequence, stated as the reporter experienced it."""
    source = "def f(job):\n" + _arguments(_wire(new_string=INDENTED_BODY))["new_string"]
    compile(source, "<edit>", "exec")


def test_trailing_newline_in_payload_survives() -> None:
    """A file body ending in a newline must arrive with it: the wire adds one
    newline, the payload's own is the second. Dropping it is what makes git
    report "\\ No newline at end of file" (same rationale as
    ``_decode_json_like``)."""
    assert _arguments(_wire(new_string="body\n"))["new_string"] == "body\n"


def test_streaming_finalize_matches_non_streaming() -> None:
    """Bare (non-JSON-quoted) values are recovered by
    ``finalize_legacy_raw_stream``, which re-enters ``extract_tool_calls``.
    Pin that the two paths agree on the indentation."""
    from unittest.mock import MagicMock

    from vllm_mlx.service.postprocessor import StreamingPostProcessor

    text = _wire(file_path="/tmp/x.py", new_string=INDENTED_BODY)
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.tool_parser_instance = None
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()
    pp.tool_accumulated_text = text
    streamed = [ev for ev in pp.finalize() if ev.type == "tool_call"]
    assert streamed, "finalize emitted no tool call"
    args = streamed[0].tool_calls[0]["function"]["arguments"]
    assert json.loads(args)["new_string"] == INDENTED_BODY


# ---------------------------------------------------------------------------
# Guards on the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared_type", "emitted", "expected"),
    [
        ("boolean", " true ", True),
        ("boolean", "\n true \n", True),
        ("integer", " 42 ", 42),
        ("number", " 1.5 ", 1.5),
        ("string", " null ", None),
    ],
)
def test_padded_scalars_still_convert(declared_type, emitted, expected) -> None:
    """Values used to reach ``_convert_param_value`` already ``.strip()``-ed.
    Now that only the wrapping newline is removed, a model that pads a scalar
    must still resolve to the scalar -- otherwise this fix would trade a
    string bug for a boolean bug (``" true "`` silently becoming ``False``)."""
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {"v": {"type": declared_type}},
                    },
                },
            }
        ]
    }
    text = f"<tool_call>\n<function=Edit>\n<parameter=v>{emitted}</parameter>\n</function>\n</tool_call>"
    assert _arguments(text, request)["v"] == expected


def test_nemotron_xml_body_keeps_indentation() -> None:
    """``split_marked_parameters`` is shared: the Nemotron XML body carries the
    identical ``<parameter=…>`` markup, so the same rule has to hold there or
    the two wires disagree about the same bytes."""
    from vllm_mlx.api.tool_calling import parse_tool_calls

    text = (
        "<tool_call>\n<function=Edit>\n"
        f"<parameter=new_string>\n{INDENTED_BODY}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, calls = parse_tool_calls(text, None)
    assert calls, "nemotron scanner recovered no call"
    assert json.loads(calls[0].function.arguments)["new_string"] == INDENTED_BODY


# ---------------------------------------------------------------------------
# Conversion must happen exactly once (codex adversarial review, round 2)
# ---------------------------------------------------------------------------


def _stream_arguments(chunks: list[str], request: dict) -> dict:
    """Concatenate streamed ``function.arguments`` fragments and parse them."""
    parser = Qwen3CoderToolParser(tokenizer=None)
    parser.reset()
    previous = ""
    fragments: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        for tc in (delta or {}).get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments")
            if args:
                fragments.append(args)
        previous = current
    return json.loads("".join(fragments))


@pytest.mark.parametrize(
    "wire",
    [
        # The value IS the four characters n-u-l-l. One conversion decodes the
        # JSON string; a second one reads that result as the null keyword.
        '"null"',
        # Padded: only reachable once values stop being .strip()-ed, which is
        # how this PR surfaced a defect that predates it.
        '" null "',
        '"true"',
        '"42"',
    ],
)
def test_json_quoted_value_is_converted_exactly_once(wire: str) -> None:
    """A JSON-quoted string value must survive streaming unchanged.

    ``_close_string_increment`` used to re-run ``_convert_param_value`` on a
    value its caller had already converted. That conversion is not idempotent:
    ``'"null"'`` decodes to the Python string ``"null"`` on the first pass and
    to ``None`` on the second, so the streamed arguments disagreed with the
    non-streamed ones. Pre-existing on main for ``'"null"'`` -- this PR widened
    it to the padded form before fixing both.
    """
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ]
    }
    chunks = [
        "<tool_call>\n<function=f>\n",
        "<parameter=x>\n" + wire[:1],
        wire[1:] + "\n</parameter>\n</function>\n</tool_call>",
    ]
    streamed = _stream_arguments(chunks, request)
    non_streamed = _arguments("".join(chunks), request)
    assert streamed == non_streamed, (
        f"stream/non-stream divergence for {wire!r}: {streamed!r} != {non_streamed!r}"
    )
    assert streamed["x"] == json.loads(wire)


def test_padded_bare_boolean_agrees_across_paths() -> None:
    """The scalar-keyword guard also closed a divergence that predates this
    PR: a padded bare ``true`` streamed as ``False`` while the non-streaming
    path returned ``True``."""
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "boolean"}},
                    },
                },
            }
        ]
    }
    chunks = [
        "<tool_call>\n<function=f>\n",
        "<parameter=x>\n ",
        "true \n</parameter>\n</function>\n</tool_call>",
    ]
    assert _stream_arguments(chunks, request) == {"x": True}
    assert _arguments("".join(chunks), request) == {"x": True}

"""Gemma 4's chat format for the OpenAI API: the model's own template (tokenizer.chat_template in the GGUF, rendered with
jinja2 as transformers and llama.cpp do) and the reading of the tool calls the model writes.

    prompt = ChatTemplate(template_text).render(messages, tools)   # messages / tools as an OpenAI request has them
    content, calls = split_tool_calls(model_text)                    # calls: [(name, {argument: value})]
"""
import json
import re
from datetime import datetime

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"
CHANNEL_OPEN = "<|channel>"    # the model's thinking: <|channel>thought\n...<channel|>
CHANNEL_CLOSE = "<channel|>"
QUOTE = '<|"|>'   # Gemma 4 writes a string argument between two of these


class ChatTemplate:
    def __init__(self, template: str, bos: str = "<bos>"):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(template)
        self.bos = bos

    def render(self, messages: list[dict], tools: list[dict] | None = None, enable_thinking: bool = False) -> str:
        try:
            return self.template.render(messages=normalize(messages), tools=tools or None, bos_token=self.bos,
                                        add_generation_prompt=True, enable_thinking=enable_thinking)
        except TemplateError as e:
            raise ValueError(f"the chat template refused the messages: {e}") from e


def normalize(messages: list[dict]) -> list[dict]:
    """OpenAI messages as the template reads them: tool-call arguments as objects, no null content, text parts."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        content = m.get("content")
        if content is None:
            m["content"] = ""
        elif isinstance(content, list):
            m["content"] = [p if isinstance(p, dict) else {"type": "text", "text": str(p)} for p in content]
        calls = m.get("tool_calls")
        if calls:
            fixed = []
            for c in calls:
                f = dict(c.get("function") or {})
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                f["arguments"] = args if isinstance(args, dict) else {}
                fixed.append({**c, "function": f})
            m["tool_calls"] = fixed
        out.append(m)
    return out


# ------------------------------------------------------------------------------------------------ tool calls
class _Reader:
    """The argument syntax: `{key:<|"|>text<|"|>,n:3,flag:true,list:[...],obj:{...}}`."""

    NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

    def __init__(self, text: str, pos: int = 0):
        self.s = text
        self.i = pos

    def error(self, what: str):
        raise ValueError(f"tool call: {what} at {self.i}")

    def value(self):
        s, i = self.s, self.i
        if s.startswith(QUOTE, i):
            end = s.find(QUOTE, i + len(QUOTE))
            if end < 0:
                self.error("an unterminated string")
            self.i = end + len(QUOTE)
            return s[i + len(QUOTE):end]
        if s.startswith("{", i):
            return self.mapping()
        if s.startswith("[", i):
            self.i += 1
            items = []
            while not s.startswith("]", self.i):
                items.append(self.value())
                if s.startswith(",", self.i):
                    self.i += 1
                elif not s.startswith("]", self.i):
                    self.error("a list that does not end")
            self.i += 1
            return items
        for word, val in (("true", True), ("false", False), ("null", None)):
            if s.startswith(word, i) and not s[i + len(word):i + len(word) + 1].isalnum():
                self.i += len(word)
                return val
        m = self.NUMBER.match(s, i)
        if m and not s[m.end():m.end() + 1].isalpha():
            self.i = m.end()
            text = m.group(0)
            return float(text) if any(c in text for c in ".eE") else int(text)
        end = i
        while end < len(s) and s[end] not in ",}]":   # an unquoted word
            end += 1
        if end == i:
            self.error("a value was expected")
        self.i = end
        return s[i:end].strip()

    def key(self) -> str:
        s, i = self.s, self.i
        if s.startswith(QUOTE, i):
            end = s.find(QUOTE, i + len(QUOTE))
            if end < 0:
                self.error("an unterminated key")
            self.i = end + len(QUOTE)
            return s[i + len(QUOTE):end]
        end = s.find(":", i)
        if end < 0:
            self.error("a key without a value")
        self.i = end
        return s[i:end].strip()

    def mapping(self) -> dict:
        if not self.s.startswith("{", self.i):
            self.error("an object was expected")
        self.i += 1
        out = {}
        while not self.s.startswith("}", self.i):
            k = self.key()
            if not self.s.startswith(":", self.i):
                self.error("a ':' was expected")
            self.i += 1
            out[k] = self.value()
            if self.s.startswith(",", self.i):
                self.i += 1
            elif not self.s.startswith("}", self.i):
                self.error("an object that does not end")
        self.i += 1
        return out


def split_tool_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """(the text before the first tool call, the calls).  Text that does not parse as a call stays text."""
    first = text.find(TOOL_OPEN)
    if first < 0:
        return text, []
    calls: list[tuple[str, dict]] = []
    pos = first
    while text.startswith(TOOL_OPEN, pos):
        start = pos + len(TOOL_OPEN)
        if not text.startswith("call:", start):
            break
        brace = text.find("{", start)
        if brace < 0:
            break
        name = text[start + 5:brace].strip()
        r = _Reader(text, brace)
        try:
            args = r.mapping()
        except ValueError:
            break
        if not text.startswith(TOOL_CLOSE, r.i):
            break
        calls.append((name, args))
        pos = r.i + len(TOOL_CLOSE)
        while pos < len(text) and text[pos] in " \n":
            pos += 1
    if not calls:
        return text, []
    return text[:first].strip(), calls


def strip_channels(text: str) -> str:
    """The answer without the model's thinking: finished <|channel>...<channel|> blocks are dropped, and so is a block
    still being written."""
    out = []
    pos = 0
    while True:
        at = text.find(CHANNEL_OPEN, pos)
        if at < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:at])
        end = text.find(CHANNEL_CLOSE, at)
        if end < 0:
            break
        pos = end + len(CHANNEL_CLOSE)
    return "".join(out)


def safe_prefix_len(text: str, markers=(TOOL_OPEN, CHANNEL_OPEN)) -> int:
    """How much of `text` can be shown while a marker may still be starting: everything before the first marker, or before
    a tail that is the beginning of one."""
    cut = len(text)
    for marker in markers:
        at = text.find(marker)
        if at >= 0:
            cut = min(cut, at)
            continue
        for n in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:n]):
                cut = min(cut, len(text) - n)
                break
    return cut

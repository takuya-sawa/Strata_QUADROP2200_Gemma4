import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_chat_format as F  # noqa: E402

# the part of Gemma 4's template these tests need: turns, the system block and the tool-call / tool-response syntax
TEMPLATE = """{{- bos_token -}}
{%- if messages[0]['role'] == 'system' -%}
<|turn>system
{{ messages[0]['content'] | trim }}<turn|>
{%- set loop_messages = messages[1:] -%}
{%- else -%}
{%- set loop_messages = messages -%}
{%- endif -%}
{%- for m in loop_messages -%}
<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}
{% if m.get('tool_calls') %}{% for c in m['tool_calls'] %}<|tool_call>call:{{ c['function']['name'] }}{{ c['function']['arguments'] | tojson }}<tool_call|>{% endfor %}{% endif %}{{ m['content'] }}<turn|>
{% endfor -%}
{%- if add_generation_prompt -%}<|turn>model
{% endif -%}"""


class ChatFormat(unittest.TestCase):
    def test_render_normalises_openai_messages(self):
        t = F.ChatTemplate(TEMPLATE)
        out = t.render([{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [{"id": "c1", "type": "function",
                                         "function": {"name": "f", "arguments": '{"a": 1}'}}]}])
        self.assertTrue(out.startswith("<bos><|turn>system\nBe brief.<turn|>"))
        self.assertIn('<|tool_call>call:f{"a": 1}<tool_call|><turn|>', out)   # arguments reached the template as an object
        self.assertTrue(out.endswith("<|turn>model\n"))

    def test_bad_arguments_do_not_break_a_request(self):
        msgs = F.normalize([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{not json"}}]}])
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_tool_call_is_read(self):
        text = ('Checking.<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>,days:2,deep:{a:[1,2.5,true,null],'
                'b:<|"|>x, y<|"|>}}<tool_call|><|tool_call>call:ping{}<tool_call|>')
        content, calls = F.split_tool_calls(text)
        self.assertEqual(content, "Checking.")
        self.assertEqual(calls, [("get_weather", {"city": "Tokyo", "days": 2,
                                                  "deep": {"a": [1, 2.5, True, None], "b": "x, y"}}),
                                 ("ping", {})])

    def test_broken_call_stays_text(self):
        text = "<|tool_call>call:f{a:<|\"|>unterminated"
        self.assertEqual(F.split_tool_calls(text), (text, []))
        self.assertEqual(F.split_tool_calls("no call here"), ("no call here", []))

    def test_marker_is_held_back_while_it_may_be_starting(self):
        self.assertEqual(F.safe_prefix_len("hello <|to"), 6)
        self.assertEqual(F.safe_prefix_len("a<|tool_call>call:f"), 1)
        self.assertEqual(F.safe_prefix_len("plain"), 5)
        self.assertEqual(F.safe_prefix_len("ok <|chan"), 3)

    def test_thinking_is_not_part_of_the_answer(self):
        self.assertEqual(F.strip_channels("<|channel>thought\n<channel|>The answer."), "The answer.")
        self.assertEqual(F.strip_channels("A<|channel>thought\nhmm<channel|>B"), "AB")
        self.assertEqual(F.strip_channels("A<|channel>thought\nstill thinking"), "A")


if __name__ == "__main__":
    unittest.main()

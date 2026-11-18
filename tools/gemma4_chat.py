#!/usr/bin/env python3
"""Chat with a Gemma 4 GGUF through Strata's Gemma 4 engine (`strata --serve`).

    python3 tools/gemma4_chat.py MODEL.gguf [--strata build/strata] [--max-new 256] [--max-context 4096] [--think]

Each turn is formatted with Gemma 4's chat template (no thinking unless --think) and tokenised with
tools/strata_tokenizer.py; the engine's `GEN` / `T` / `DONE` protocol streams the reply back.  The engine keeps the KV
cache between turns, so a follow-up only reads its new tokens.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--strata", default=str(Path(__file__).resolve().parent.parent / "build" / "strata"))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="let the model think before it answers")
    args = ap.parse_args()

    tok = Tokenizer.from_gguf(args.model)
    engine = subprocess.Popen(
        [args.strata, "--native", args.model, "--serve", "--max-context", str(args.max_context)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    ready = engine.stdout.readline()
    if not ready.startswith("READY"):
        print("the engine did not start:", ready, file=sys.stderr)
        return 1

    history = ""
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                break
            if not user.strip():
                continue
            history += f"<|turn>user\n{user}<turn|>\n<|turn>model\n"
            prompt = "<bos>" + history + ("" if args.think else "<|channel>thought\n<channel|>")
            ids = tok.encode(prompt, parse_special=True)
            engine.stdin.write(f"GEN {args.max_new} temperature={args.temperature} {','.join(map(str, ids))}\n")
            engine.stdin.flush()
            reply: list[int] = []
            print("gemma> ", end="", flush=True)
            shown = ""
            for line in engine.stdout:
                if line.startswith("T "):
                    tid = int(line.split()[1])
                    if tid in (1, 106):   # <eos> / <turn|>: the end of the reply, not text
                        continue
                    reply.append(tid)
                    text = tok.decode(reply)
                    print(text[len(shown):], end="", flush=True)
                    shown = text
                elif line.startswith(("DONE", "ERR")):
                    break
            print()
            history += tok.decode(reply) + "<turn|>\n"
    finally:
        engine.stdin.close()
        engine.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

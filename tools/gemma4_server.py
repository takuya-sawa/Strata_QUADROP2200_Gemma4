#!/usr/bin/env python3
"""An HTTP server for Strata's Gemma 4 engine: an OpenAI-style chat API plus a small web page.

    python3 tools/gemma4_server.py MODEL.gguf --strata build/strata [--host 0.0.0.0] [--port 8088]

    GET  /                      the chat page
    GET  /monitor               live status page (GPU, the running request, speed, history)
    GET  /status                the same as JSON
    GET  /v1/models
    POST /v1/chat/completions   messages, max_tokens, temperature, top_p, top_k, stream

There is no authentication: bind it to a trusted network only.  The engine holds one conversation's KV cache, so requests
are served one at a time (a second one waits for the first); a follow-up that starts with the same messages reuses it.
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_chat_format import ChatTemplate, safe_prefix_len, split_tool_calls, strip_channels  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from strata_tokenizer import Tokenizer  # noqa: E402

STOP_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")   # the end of a turn; the tool's answer is the runtime's to write

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Gemma 4 (Strata)</title>
<style>body{font:16px system-ui;max-width:800px;margin:0 auto;padding:12px;display:flex;flex-direction:column;height:100vh;box-sizing:border-box}
#log{flex:1;overflow:auto;white-space:pre-wrap}.u{color:#06c;margin-top:12px}.m{margin-top:4px}
form{display:flex;gap:8px;margin-top:8px}textarea{flex:1;font:inherit;height:3.2em}</style>
<h3>Gemma 4 26B-A4B (Strata) <small><a href="/monitor">monitor</a></small></h3><div id=log></div>
<form id=f><textarea id=q placeholder="Enter: send / Shift+Enter: new line" autofocus></textarea><button>Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q');let msgs=[];
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();f.requestSubmit()}});
f.onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';msgs.push({role:'user',content:t});
 log.insertAdjacentHTML('beforeend','<div class=u></div><div class=m></div>');
 const u=log.children[log.children.length-2],m=log.lastChild;u.textContent=t;
 const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:msgs,stream:true,max_tokens:1024})});
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='',out='';
 for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
  let i;while((i=buf.indexOf('\\n\\n'))>=0){const l=buf.slice(0,i);buf=buf.slice(i+2);
   if(l.startsWith('data: ')&&l!=='data: [DONE]'){const d=JSON.parse(l.slice(6)).choices[0].delta;
    if(d.content){out+=d.content;m.textContent=out;log.scrollTop=log.scrollHeight}}}}
 msgs.push({role:'assistant',content:out})};
</script>"""


MONITOR = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Gemma 4 monitor</title>
<style>body{font:15px system-ui;max-width:900px;margin:0 auto;padding:12px}h3{margin:14px 0 4px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:3px 8px;text-align:left}
.bar{background:#eee;height:14px;border-radius:7px;overflow:hidden}.bar>i{display:block;height:100%;background:#06c}
.busy{color:#c60;font-weight:bold}.idle{color:#080;font-weight:bold}a{color:#06c}</style>
<h3>Gemma 4 26B-A4B (Strata) - monitor <small><a href="/">chat</a></small></h3>
<div id=st></div><h3>GPU</h3><div id=gpu></div><h3>Server</h3><div id=srv></div>
<h3>Recent requests</h3><table id=hist></table>
<script>
const f=(x,d=1)=>x==null?'-':Number(x).toFixed(d);
async function tick(){try{const s=await (await fetch('/status')).json();
 const c=s.current;
 document.getElementById('st').innerHTML=c?`<span class=busy>BUSY</span> ${c.generated} tokens generated for a ${c.prompt_tokens}-token prompt,
  ${f(c.elapsed,1)} s`+(c.tok_s?`, ${f(c.tok_s)} tok/s`:`, reading the prompt...`)+` (waiting: ${s.waiting})`:`<span class=idle>IDLE</span> (waiting: ${s.waiting})`;
 const g=s.gpu;document.getElementById('gpu').innerHTML=g?`${g.name}: ${g.mem_used} / ${g.mem_total} MiB
  <div class=bar><i style="width:${100*g.mem_used/g.mem_total}%"></i></div>utilization ${g.util}%, ${g.temp} C`:'(nvidia-smi is not available)';
 document.getElementById('srv').innerHTML=`up ${f(s.uptime/3600,2)} h; requests ${s.requests} (errors ${s.errors}); prompt tokens ${s.prompt_tokens}, generated ${s.generated_tokens};
  context ${s.context_used} / ${s.max_context} used by the last request<div class=bar><i style="width:${100*s.context_used/s.max_context}%"></i></div>
  RAM free ${f(s.mem_available_mib,0)} MiB, load ${s.load.map(x=>f(x,2)).join(' ')}`;
 document.getElementById('hist').innerHTML='<tr><th>time<th>prompt<th>generated<th>prompt s<th>tok/s<th>finish</tr>'+s.history.map(h=>
  `<tr><td>${new Date(h.finished*1000).toLocaleTimeString()}<td>${h.prompt_tokens}<td>${h.generated}<td>${f(h.prompt_s,1)}<td>${f(h.tok_s)}<td>${h.finish}</tr>`).join('');
}catch(e){document.getElementById('st').textContent='(no answer from the server)'}}
tick();setInterval(tick,1000);
</script>"""


def gpu_info(_cache={"t": 0.0, "v": None}):
    """The GPU's memory, utilisation and temperature from nvidia-smi (read at most once a second)."""
    now = time.time()
    if now - _cache["t"] < 1.0:
        return _cache["v"]
    _cache["t"] = now
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, used, total, util, temp = [x.strip() for x in out.split(",")]
        _cache["v"] = {"name": name, "mem_used": int(used), "mem_total": int(total), "util": int(util), "temp": int(temp)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        _cache["v"] = None
    return _cache["v"]


class Stats:
    """What the monitor shows: the running request, the totals and the last requests."""

    def __init__(self, max_context: int):
        self.lock = threading.Lock()
        self.started = time.time()
        self.max_context = max_context
        self.waiting = 0
        self.requests = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.context_used = 0
        self.current = None
        self.history = collections.deque(maxlen=20)

    def enqueue(self):
        with self.lock:
            self.waiting += 1

    def begin(self, prompt_tokens: int):
        with self.lock:
            self.waiting -= 1
            self.current = {"prompt_tokens": prompt_tokens, "generated": 0, "start": time.time(), "first": None, "last": None}

    def token(self):
        with self.lock:
            c = self.current
            c["generated"] += 1
            c["last"] = time.time()
            if c["first"] is None:
                c["first"] = c["last"]

    def end(self, finish: str):
        with self.lock:
            c, self.current = self.current, None
            if c is None:
                return
            n = c["generated"]
            rate = (n - 1) / (c["last"] - c["first"]) if n > 1 and c["last"] > c["first"] else None
            self.requests += 1
            self.errors += finish == "error"
            self.prompt_tokens += c["prompt_tokens"]
            self.generated_tokens += n
            self.context_used = c["prompt_tokens"] + n
            self.history.appendleft({"finished": time.time(), "prompt_tokens": c["prompt_tokens"], "generated": n,
                                     "prompt_s": (c["first"] or time.time()) - c["start"], "tok_s": rate, "finish": finish})

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            c = self.current
            cur = None
            if c:
                rate = (c["generated"] - 1) / (c["last"] - c["first"]) if c["generated"] > 1 and c["last"] > c["first"] else None
                cur = {"prompt_tokens": c["prompt_tokens"], "generated": c["generated"], "elapsed": now - c["start"], "tok_s": rate}
            mem = 0.0
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            mem = int(line.split()[1]) / 1024
            except OSError:
                pass
            return {"uptime": now - self.started, "waiting": self.waiting, "requests": self.requests, "errors": self.errors,
                    "prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens,
                    "context_used": self.context_used, "max_context": self.max_context, "current": cur,
                    "history": list(self.history), "gpu": gpu_info(), "load": list(os.getloadavg()),
                    "mem_available_mib": mem}


IMAGE_MARK = "<|image|>"   # where the chat template puts a picture


class Vision:
    """Pictures through strata-gemma4-vision (the vision tower on the CPU), remembered by the picture's hash."""

    def __init__(self, helper: str, text_gguf: str, mmproj: str, threads: int):
        self.helper, self.text_gguf, self.mmproj, self.threads = helper, text_gguf, mmproj, threads
        self.cache = Path(tempfile.gettempdir()) / "gemma4-vision-cache"
        self.cache.mkdir(exist_ok=True)
        self.lock = threading.Lock()

    def segments(self, data: bytes):
        """[("t", [token ids]) | ("e", count, n_embd, key, float bytes)] for one picture."""
        digest = hashlib.sha256(data).hexdigest()
        out = self.cache / f"{digest}.emb"
        with self.lock:
            if not out.exists():
                src = self.cache / f"{digest}.img"
                src.write_bytes(data)
                tmp = self.cache / f"{digest}.part"
                try:
                    run = subprocess.run([self.helper, self.text_gguf, self.mmproj, str(src), str(tmp), str(self.threads)],
                                         capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                    if run.returncode != 0 or not tmp.exists():
                        raise ValueError("the picture could not be read: " + (run.stderr.strip().splitlines() or ["?"])[-1])
                    tmp.rename(out)
                finally:
                    src.unlink(missing_ok=True)
                    tmp.unlink(missing_ok=True)
        raw = out.read_bytes()
        key = int(digest[:16], 16)
        n, pos, segs = struct.unpack_from("<I", raw, 4)[0], 8, []
        for _ in range(n):
            kind, count = struct.unpack_from("<II", raw, pos)
            pos += 8
            if kind == 0:
                segs.append(("t", list(struct.unpack_from(f"<{count}i", raw, pos))))
                pos += 4 * count
            else:
                ne = struct.unpack_from("<I", raw, pos)[0]
                pos += 4
                segs.append(("e", count, ne, key, raw[pos:pos + 4 * count * ne]))
                pos += 4 * count * ne
        return segs


def image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"a picture is not valid base64: {e}")
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read(32 << 20)
    raise ValueError("a picture must be a data: or http(s): URL")


def message_images(messages: list[dict]) -> list[bytes]:
    """The pictures of the messages, in the order the template writes its <|image|> markers."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    ref = part.get("image_url") or part.get("source") or {}
                    out.append(image_bytes(ref.get("url", "") if isinstance(ref, dict) else str(ref)))
    return out


def write_prompt(path: str, pieces: list) -> int:
    """The prompt file the engine reads (G4P1); the number of positions it holds."""
    total = 0
    with open(path, "wb") as f:
        f.write(b"G4P1" + struct.pack("<I", len(pieces)))
        for p in pieces:
            if p[0] == "t":
                f.write(struct.pack("<II", 0, len(p[1])) + struct.pack(f"<{len(p[1])}i", *p[1]))
                total += len(p[1])
            else:
                f.write(struct.pack("<IIIQ", 1, p[1], p[2], p[3]) + p[4])
                total += p[1]
    return total


class Engine:
    """`strata --serve` as a child process, one request at a time."""

    def __init__(self, strata: str, model: str, max_context: int, extra: list[str]):
        self.max_context = max_context
        self.proc = subprocess.Popen(
            [strata, "--native", model, "--serve", "--max-context", str(max_context), *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.proc.stdout.readline()
        if not ready.startswith("READY"):
            raise RuntimeError(f"the engine did not start: {ready!r}")
        self.lock = threading.Lock()

    def generate(self, ids: list[int] | None, max_new: int, sampling: dict, prompt_file: str | None = None):
        """Yields token ids; the caller holds self.lock.  Closing the generator cancels the reply."""
        keys = " ".join(f"{k}={v}" for k, v in sampling.items())
        if prompt_file:
            line = f"GENE {max_new} {keys} {prompt_file}"
        else:
            line = f"GEN {max_new} {keys} {','.join(map(str, ids))}"
        self.proc.stdin.write(line.replace("  ", " ") + "\n")
        self.proc.stdin.flush()
        done = False
        try:
            for line in self.proc.stdout:
                if line.startswith("T "):
                    yield int(line.split()[1])
                elif line.startswith(("DONE", "ERR")):
                    done = True
                    return
        finally:
            if not done:   # the client went away: cancel, then read to the end of the answer
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
                for line in self.proc.stdout:
                    if line.startswith(("DONE", "ERR")):
                        break


def make_handler(engine: Engine, tok: Tokenizer, template: ChatTemplate, stop_ids: tuple, name: str, stats: Stats,
                 vision: Vision | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def _error(self, code: int, msg: str):
            self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _route(self) -> str:
            """The path without its query, with the /v1 prefix optional or repeated (clients differ on the base URL)."""
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path.startswith(("/v1/", "/chat/", "/models")):
                path = "/v1/" + re.sub(r"^(/v1)*/", "", path)
            return path

        def do_GET(self):
            path = self._route()
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/monitor":
                self._send(200, MONITOR.encode(), "text/html; charset=utf-8")
            elif path == "/status":
                self._json(200, stats.snapshot())
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "strata"}]})
            elif path in ("/health", "/healthz"):
                self._json(200, {"status": "ok"})
            else:
                self._error(404, "not found")

        def do_POST(self):
            if self._route() != "/v1/chat/completions":
                return self._error(404, "not found")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                tools = req.get("tools") or None
                text = template.render(req["messages"], tools)
                images = message_images(req["messages"])
                ids, prompt_file = None, None
                if images:
                    if vision is None:
                        return self._error(400, "pictures are not available: no vision projector (mmproj) was found")
                    texts = text.split(IMAGE_MARK)
                    if len(texts) != len(images) + 1:
                        return self._error(400, "the pictures do not match the template's markers")
                    pieces = []
                    for i, t in enumerate(texts):
                        if t:
                            pieces.append(("t", tok.encode(t, parse_special=True)))
                        if i < len(images):
                            pieces.extend(vision.segments(images[i]))
                    fd, prompt_file = tempfile.mkstemp(prefix="gemma4-prompt-", suffix=".bin")
                    os.close(fd)
                    n_prompt = write_prompt(prompt_file, pieces)
                else:
                    ids = tok.encode(text, parse_special=True)
                    n_prompt = len(ids)
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
                return self._error(400, f"bad request: {e}")
            max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
            room = engine.max_context - n_prompt
            if room < 1:
                if prompt_file:
                    os.unlink(prompt_file)
                return self._error(400, f"the prompt ({n_prompt} tokens) does not fit the context ({engine.max_context})")
            max_new = max(1, min(max_new, room))
            sampling = {"temperature": float(req.get("temperature", 0.7))}
            if req.get("top_p") is not None:
                sampling["top_p"] = float(req["top_p"])
            if req.get("top_k") is not None:
                sampling["top_k"] = int(req["top_k"])
            if req.get("seed") is not None:
                sampling["seed"] = int(req["seed"])
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            stream = bool(req.get("stream"))

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            stats.enqueue()
            with engine.lock:
                stats.begin(n_prompt)
                reply: list[int] = []
                shown = 0   # characters of the text already sent
                finish = "length"
                gen = engine.generate(ids, max_new, sampling, prompt_file)
                try:
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()

                        def emit(obj):
                            data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
                                    + "\n\n").encode()
                            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                            self.wfile.flush()
                        emit(chunk({"role": "assistant", "content": ""}))
                    for t in gen:
                        stats.token()
                        if t in stop_ids:
                            finish = "stop"
                            break
                        reply.append(t)
                        raw = tok.decode(reply)
                        if raw.endswith("\ufffd"):   # a character still being completed
                            continue
                        text = strip_channels(raw)   # the thinking is not part of the answer
                        if stream:
                            upto = safe_prefix_len(text) if tools else safe_prefix_len(text, ("<|channel>",))
                            if upto > shown:
                                emit(chunk({"content": text[shown:upto]}))
                                shown = upto
                    text = strip_channels(tok.decode(reply))
                    content, calls = split_tool_calls(text) if tools else (text, [])
                    if calls:
                        finish = "tool_calls"
                    elif len(reply) >= max_new and finish != "stop":
                        finish = "length"
                    tool_calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                   "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                                  for n, a in calls]
                    if stream:
                        if not calls and len(text) > shown:
                            emit(chunk({"content": text[shown:]}))
                        elif calls and not shown and content:
                            emit(chunk({"content": content}))
                        if calls:
                            emit(chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(tool_calls)]}))
                        emit(chunk({}, finish))
                        emit("[DONE]")
                        self.wfile.write(b"0\r\n\r\n")
                    else:
                        message = {"role": "assistant", "content": content or None}
                        if tool_calls:
                            message["tool_calls"] = tool_calls
                        self._json(200, {
                            "id": cid, "object": "chat.completion", "created": created, "model": name,
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(reply),
                                      "total_tokens": n_prompt + len(reply)}})
                except (BrokenPipeError, ConnectionResetError):
                    finish = "disconnected"
                finally:
                    gen.close()
                    stats.end(finish)
                    if prompt_file:
                        os.unlink(prompt_file)
    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--strata", default=str(Path(__file__).resolve().parent.parent / "build" / "strata"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--name", default="gemma-4-26b-a4b")
    ap.add_argument("--mmproj", default=None, help="the vision projector GGUF (default: mmproj-*.gguf beside the model)")
    ap.add_argument("--vision-threads", type=int, default=4)
    args, engine_args = ap.parse_known_args()   # what it does not know goes to the engine (--no-gpu, --kv-type, ...)

    tok = Tokenizer.from_gguf(args.model)
    template = ChatTemplate(GGUFFile(Path(args.model)).metadata["tokenizer.chat_template"])
    stop_ids = tuple(tok.encode(t, parse_special=True)[0] for t in STOP_TOKENS)
    if "--stop-ids" not in engine_args:
        engine_args += ["--stop-ids", ",".join(map(str, stop_ids))]
    engine = Engine(args.strata, args.model, args.max_context, engine_args)
    mmproj = args.mmproj or next(iter(sorted(Path(args.model).parent.glob("mmproj-gemma-4-26B*.gguf"))), None)
    helper = Path(args.strata).resolve().parent / "strata-gemma4-vision"
    vision = Vision(str(helper), args.model, str(mmproj), args.vision_threads) if mmproj and helper.exists() else None
    print("pictures:", f"on ({Path(mmproj).name})" if vision else "off (no mmproj or no strata-gemma4-vision)", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine, tok, template, stop_ids, args.name, Stats(args.max_context), vision))
    httpd.daemon_threads = True
    print(f"Gemma 4 server on http://{args.host}:{args.port}/ (OpenAI API: /v1/chat/completions, monitor: /monitor)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

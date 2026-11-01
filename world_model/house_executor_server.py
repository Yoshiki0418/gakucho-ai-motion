"""G7-3: the executor as a small HTTP service, for the conversation brain (a cloud LLM).

    python -m world_model.house_executor_server [--port 4013]

The brain passes the user's instruction on as it is (no decomposition); the executor (``house_agent``: the
local VLM looks and decides, the tracker moves the body) carries it out in the visible house:

    POST /runs  {"instruction": "冷蔵庫の前まで来て"}   -> 202 {"id": "live-...", "position": n in the queue}
    GET  /runs/<id>                                   -> {"status": "queued|running|done|error", "result": {...}}

One episode at a time, each from the usual start in the house (the LDK door, facing north). A finished run is
published next to the test tasks (/training-monitor/agent/, page /world-model/house). There is no ground
truth for a free instruction: ``success`` means the VLM checked on arrival that it is there (or said it cannot
be done) without a fall or a contact.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from .house_agent import Agent, publish

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/world-model/house2f/agent-live"


class Executor:
    def __init__(self, args):
        self.args = SimpleNamespace(model=args.model, vlm=args.vlm, vlm_model=args.vlm_model, base=args.base)
        self.jobs, self.runs, self.lock = queue.Queue(), {}, threading.Lock()
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, instruction):
        run_id = time.strftime("live-%Y%m%d-%H%M%S")
        with self.lock:
            while run_id in self.runs:
                run_id += "b"
            self.runs[run_id] = {"id": run_id, "instruction": instruction, "status": "queued", "submittedAt": time.strftime("%H:%M:%S")}
        self.jobs.put(run_id)
        return run_id, self.jobs.qsize()

    def work(self):
        while True:
            run_id = self.jobs.get()
            run = self.runs[run_id]
            run["status"], run["startedAt"] = "running", time.strftime("%H:%M:%S")
            folder = OUT / run_id
            folder.mkdir(parents=True, exist_ok=True)
            task = {"id": run_id, "text": run["instruction"], "kind": "free", "target": None}
            try:
                agent = Agent(self.args, folder)
                started = time.time()
                try:
                    result = agent.episode(task)
                    agent.save(task, result)
                finally:
                    agent.close()
                run["result"] = {k: result.get(k) for k in ("success", "finished", "claimed", "message", "stoppedBy", "fell",
                                                            "contactCount", "decisions", "looks", "seconds")}
                run["status"] = "done"
                summary = {"id": run_id, "text": run["instruction"], "source": "brain", **run["result"],
                           "wall": round(time.time() - started, 1)}
                index_path = OUT / "index.json"
                index = json.loads(index_path.read_text()) if index_path.exists() else {"version": 1, "runs": []}
                index["runs"] = [r for r in index["runs"] if r["id"] != run_id] + [summary]
                index["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
                publish(OUT)
            except Exception as error:     # keep serving the next instruction
                run["status"], run["error"] = "error", f"{error}\n{traceback.format_exc()[-1500:]}"


def handler_for(executor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/runs":
                return self.reply(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                instruction = str(body.get("instruction", "")).strip()
            except ValueError:
                return self.reply(400, {"error": "JSON で {\"instruction\": \"...\"} を送ってください"})
            if not instruction or len(instruction) > 200:
                return self.reply(400, {"error": "instruction は 1〜200 文字"})
            run_id, position = executor.submit(instruction)
            self.reply(202, {"id": run_id, "position": position})

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if parts == ["runs"]:
                return self.reply(200, {"runs": list(executor.runs.values())})
            if len(parts) == 2 and parts[0] == "runs" and parts[1] in executor.runs:
                return self.reply(200, executor.runs[parts[1]])
            self.reply(404, {"error": "not found"})

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=4013)
    parser.add_argument("--model", default="v14", help="tracker run")
    parser.add_argument("--vlm", default="http://127.0.0.1:11999")
    parser.add_argument("--vlm-model", default="qwen3-vl:30b-a3b-instruct")
    parser.add_argument("--base", default="http://127.0.0.1:4010", help="the house page used to render the colour images")
    args = parser.parse_args()
    executor = Executor(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(executor))
    print(f"executor ready on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

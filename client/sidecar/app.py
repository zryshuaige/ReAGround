import hashlib
import hmac
import json
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from desktop import capture_desktop, click_once, draw_point, enable_dpi_awareness
from grounding import Grounder, load_specs


class ReAGroundService:
    def __init__(self):
        self.lock = threading.Lock()
        self.grounder = None
        self.captures = {}
        self.predictions = {}
        self.temp_root = Path(tempfile.mkdtemp(prefix="reaground-sidecar-"))

    def models(self):
        return [{
            "id": spec.key,
            "name": spec.name,
            "available": spec.available,
            "hasResults": spec.results is not None,
        } for spec in load_specs()]

    def results(self):
        rows = []
        for spec in load_specs():
            if spec.results is None:
                continue
            row = {"modelId": spec.key, "modelName": spec.name,
                   "path": str(spec.results), "visualTokens": spec.visual_tokens,
                   "metrics": None, "error": None}
            try:
                row["metrics"] = json.loads(spec.results.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                row["error"] = str(exc)
            rows.append(row)
        return rows

    def capture(self):
        started = time.perf_counter()
        with self.lock:
            self.predictions.clear()
            self.captures.clear()
            for path in self.temp_root.iterdir():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            desktop = capture_desktop()
            capture_id = uuid.uuid4().hex
            directory = self.temp_root / capture_id
            directory.mkdir()
            image_path = directory / "capture.png"
            desktop.image.save(image_path, compress_level=1)
            self.captures[capture_id] = {
                "image": desktop.image, "origin": desktop.origin, "directory": directory,
            }
            return {
                "captureId": capture_id, "imagePath": str(image_path),
                "width": desktop.image.width, "height": desktop.image.height,
                "timings": {"captureMs": round((time.perf_counter() - started) * 1000, 1)},
            }

    def predict(self, model_id, capture_id, instruction, visual_tokens=1280):
        if type(visual_tokens) is not int or not 256 <= visual_tokens <= 2048 \
                or visual_tokens % 128:
            raise ValueError("视觉 token 上限必须是 256–2048 之间 128 的整数倍")
        with self.lock:
            capture = self.captures[capture_id]
            spec = next(spec for spec in load_specs() if spec.key == model_id)
            if self.grounder is not None and self.grounder.spec.key != spec.key:
                self.grounder.close()
                self.grounder = None
            if self.grounder is None:
                self.grounder = Grounder(spec)
            started = time.perf_counter()
            prediction = self.grounder.predict(
                capture["image"], instruction, visual_tokens)
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            marked = draw_point(capture["image"], prediction.x, prediction.y)
            marked_path = capture["directory"] / "prediction.png"
            marked.save(marked_path, compress_level=1)
            prediction_id = uuid.uuid4().hex
            self.predictions[prediction_id] = {
                "x": prediction.x, "y": prediction.y,
                "origin": capture["origin"], "active": True,
            }
            return {
                "predictionId": prediction_id, "raw": prediction.raw,
                "x": prediction.x, "y": prediction.y,
                "resizedWh": prediction.resized_wh, "imagePath": str(marked_path),
                "visualTokens": visual_tokens, "attention": "sdpa",
                "timings": {"inferenceMs": inference_ms},
            }

    def settle(self, prediction_id):
        prediction = self.predictions.get(prediction_id)
        if prediction is None or not prediction["active"]:
            raise ValueError("预测已失效或已经处理")
        prediction["active"] = False

    def cancel(self, prediction_id):
        with self.lock:
            self.settle(prediction_id)
            return {"cancelled": True}

    def click(self, prediction_id):
        with self.lock:
            self.settle(prediction_id)
            prediction = self.predictions[prediction_id]
            click_once((prediction["x"], prediction["y"]), prediction["origin"])
            return {"clicked": True}

    def close(self):
        with self.lock:
            if self.grounder is not None:
                self.grounder.close()
                self.grounder = None
        shutil.rmtree(self.temp_root, ignore_errors=True)


SERVICE = ReAGroundService()
TOKEN = secrets.token_hex(24)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("sidecar: " + fmt % args + "\n")

    def authorized(self):
        supplied = self.headers.get("X-ReAGround-Token", "")
        digest = hashlib.sha256(supplied.encode()).digest()
        expected = hashlib.sha256(TOKEN.encode()).digest()
        return hmac.compare_digest(digest, expected)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("请求过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def handle_request(self):
        if not self.authorized():
            self.send_json({"error": "未授权"}, 403)
            return
        try:
            if self.command == "GET" and self.path == "/api/health":
                self.send_json({"ok": True})
            elif self.command == "GET" and self.path == "/api/models":
                self.send_json({"models": SERVICE.models()})
            elif self.command == "GET" and self.path == "/api/results":
                self.send_json({"results": SERVICE.results()})
            elif self.command == "POST" and self.path == "/api/capture":
                self.send_json(SERVICE.capture())
            elif self.command == "POST" and self.path == "/api/predict":
                body = self.read_json()
                self.send_json(SERVICE.predict(
                    body["modelId"], body["captureId"], body["instruction"].strip(),
                    body.get("visualTokens", 1280)))
            elif self.command == "POST" and self.path == "/api/cancel":
                self.send_json(SERVICE.cancel(self.read_json()["predictionId"]))
            elif self.command == "POST" and self.path == "/api/click":
                self.send_json(SERVICE.click(self.read_json()["predictionId"]))
            elif self.command == "POST" and self.path == "/api/shutdown":
                self.send_json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_json({"error": "接口不存在"}, 404)
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    do_GET = handle_request
    do_POST = handle_request


def main():
    enable_dpi_awareness()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(json.dumps({"port": server.server_address[1], "token": TOKEN}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        SERVICE.close()


if __name__ == "__main__":
    main()

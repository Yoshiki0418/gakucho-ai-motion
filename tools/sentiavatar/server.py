"""SentiAvatar の常駐推論サーバ(研究評価専用)。

ライセンス: SentiAvatar のコード・重みは SentiPulse Non-Commercial Source License v1.0。
学内の研究評価(会話画面の「リズムの層」の比較実験)にだけ使う。本番運用はしない。

公式の single_case_infer.py は呼び出しのたびに HuBERT / RVQVAE / Mask Transformer を
読み込み直し、プランナ(Qwen2-0.5B)を別プロセスの vLLM に投げる。ここでは全モデルを
1 プロセスに常駐させ、プランナも transformers で直接動かす(vLLM は torch 2.8 固定で
この環境の他の仕組みと同居できないため。docs/sentiavatar-eval.md 参照)。

    POST /generate  {"audio_wav_base64": ..., "action_text": "动作：说话", "seed": 0}
      → {"bvh": 学長アバター用に組み直した BVH, "bvh_raw": 公式の BVH, "fps": 30, "timing": {...}}
    GET  /health

起動は tools/sentiavatar/start.sh。
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(os.environ.get("SENTIAVATAR_ROOT", Path(__file__).resolve().parents[2] / ".runtime/sentiavatar/SentiAvatar"))
MODULE_DIR = ROOT / "motion_generation"
CHECKPOINTS = ROOT / "checkpoints"
sys.path.insert(0, str(MODULE_DIR))
os.chdir(MODULE_DIR)  # 公式コードが相対パスでメタデータを読むため

from actions.postprocess import MotionPostprocesser  # noqa: E402
from infer import load_config_from_checkpoint, load_model  # noqa: E402
from pipeline_infer import (  # noqa: E402
    VLLMClient, ensure_length, extract_mids_from_string, interpolate_sequence,
    load_mask_transformer, sparse_to_keyframes, construct_llm_prompt,
)
from reconstruct_from_tokens import decode_body_tokens  # noqa: E402
from single_case_infer import ApplyKmeans, resample_fps_1d  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, HubertModel, Wav2Vec2FeatureExtractor  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_FPS = 30.0
TEMPLATE = "Human: {prompt}<|im_end|>\nAssistant:"
CACHE_TOKENS = 2048  # 入力 + 計画トークンの上限(約 70 秒の発話まで)


class LocalPlanner(VLLMClient):
    """vLLM サーバの代わりに、同じプロセス内の transformers で計画トークンを出す。"""

    def __init__(self, path: Path) -> None:
        super().__init__("local")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE).eval()
        # 静的 KV キャッシュ + torch.compile で約 2 倍速くなる(37 → 90 tok/s)。
        # キャッシュ長が変わると再コンパイルになるので、入力 + 生成の合計を常に一定にする。
        self.model.generation_config.cache_implementation = "static"
        self.stop_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def health_check(self) -> bool:
        return True

    @torch.no_grad()
    def predict_motion_plan(self, text_list, temperature=0.5, top_p=0.4, max_tokens=1024, stop=None, base_token_start=0):
        results = []
        for text in text_list:
            inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
            budget = max(CACHE_TOKENS - inputs["input_ids"].shape[1], 64)
            output = self.model.generate(
                **inputs, max_new_tokens=budget, do_sample=True, temperature=max(temperature, 0.05),
                top_p=top_p, eos_token_id=self.stop_id, pad_token_id=self.tokenizer.pad_token_id or self.stop_id,
            )
            generated = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            motion = generated.strip().split("<|im_end|>")[0].replace("<unk>", "").replace(" ", "")
            # 以下は公式 VLLMClient.predict_motion_plan の解析と同じ
            motion_clean = motion.replace("[res_", "[res")
            sequence = motion_clean
            for sep in ["[step_4]", "[STEP_4]", "[step_1]", "[STEP_1]"]:
                if sep.lower() in sequence.lower():
                    idx = sequence.lower().rfind(sep.lower())
                    sequence = sequence[idx + len(sep):]
                    break
            parsed = extract_mids_from_string(sequence)
            full = extract_mids_from_string(motion_clean)
            tokens = {f"res_{k}": parsed.get(f"res{k}", []) for k in range(1, 5)}
            shortest = min(len(v) for v in tokens.values())
            tokens = {k: [t - base_token_start for t in v[:shortest]] for k, v in tokens.items()}
            results.append({"tokens": tokens, "total_len": full.get("len", [None])[0] if "len" in full else None,
                            "raw_output": motion})
        return results


class Engine:
    def __init__(self) -> None:
        started = time.time()
        hubert_path = CHECKPOINTS / "chinese-hubert-base"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
        self.hubert = HubertModel.from_pretrained(hubert_path).to(DEVICE).eval()
        self.kmeans = ApplyKmeans(str(CHECKPOINTS / "hubert_kmeans/model.mdl")).to(DEVICE)
        self.planner = LocalPlanner(CHECKPOINTS / "llm")
        self.mask_model = load_mask_transformer(str(CHECKPOINTS / "mask_transformer"), device=DEVICE)
        rvq_path = str(CHECKPOINTS / "rvqvae/model/epoch_30.pth")
        self.rvq = load_model(rvq_path, load_config_from_checkpoint(rvq_path), torch.device(DEVICE))
        meta = MODULE_DIR / "meta/mta_gen_demo"
        self.mean = torch.tensor(np.load(meta / "mean.npy")).to(DEVICE)
        self.std = torch.tensor(np.load(meta / "std.npy")).to(DEVICE)
        self.placeholder = np.load(MODULE_DIR / "meta/xiu_joint_quat_vecs/Daiji_A_001_V001.npy", allow_pickle=True).item()
        self.postprocess = MotionPostprocesser()
        self.lock = threading.Lock()
        # 初回だけ cuDNN の初期化とコンパイルで 20 秒以上かかるので、起動時に済ませる
        noise = (np.random.default_rng(0).standard_normal(44100 * 3) * 0.05).astype(np.float32)
        try:
            self.generate(noise, 44100, "动作：说话", 0, 0.2, 0.2, 6)  # TTS と同じ 44.1kHz(リサンプラも温める)
        except Exception as error:  # noqa: BLE001  (ノイズでは計画が短すぎることがある)
            print(f"[sentiavatar] warmup: {error}", flush=True)
        self.load_seconds = time.time() - started

    @torch.no_grad()
    def audio_features(self, wav: np.ndarray, sr: int):
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        values = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(DEVICE)
        hidden = self.hubert(values, output_hidden_states=True).hidden_states[8].squeeze(0).cpu().numpy()
        features = resample_fps_1d(hidden, src_fps=50.0, tgt_fps=10.0).astype(np.float32)
        tokens = self.kmeans.feat2token(torch.tensor(features, dtype=torch.float32).to(DEVICE))
        return features, tokens

    def generate(self, wav: np.ndarray, sr: int, action_text: str, seed: int | None,
                 temperature: float, top_p: float, steps: int) -> dict:
        with self.lock:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            timing = {}
            t0 = time.time()
            features, audio_tokens = self.audio_features(wav, sr)
            timing["audio"] = time.time() - t0
            prompt, _ = construct_llm_prompt(action_text, audio_tokens, 0, 4)
            t1 = time.time()
            plan = self.planner.predict_motion_plan([TEMPLATE.format(prompt=prompt)], temperature=temperature,
                                                    top_p=top_p, max_tokens=1024)[0]
            timing["planner"] = time.time() - t1
            keyframes = sparse_to_keyframes(plan["tokens"])
            if len(keyframes) < 2:
                raise RuntimeError(f"planner returned {len(keyframes)} keyframes")
            t2 = time.time()
            dense = ensure_length(interpolate_sequence(self.mask_model, keyframes, features, generate_steps=steps),
                                  features.shape[0])
            timing["infill"] = time.time() - t2
            t3 = time.time()
            motion = decode_body_tokens(self.rvq, dense, self.placeholder, self.mean, self.std, torch.device(DEVICE),
                                        src_fps=20.0, tgt_fps=OUTPUT_FPS)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "out.bvh"
                self.postprocess.save_quat_motion_to_bvh(motion=motion, save_path=str(path))
                raw = path.read_text()
            raw = re.sub(r"(Frame Time:\s*)([\d.]+)", lambda m: f"{m.group(1)}{1.0 / OUTPUT_FPS:.6f}", raw, count=1)
            timing["decode"] = time.time() - t3
            timing["total"] = time.time() - t0
            return {"bvh_raw": raw, "keyframes": len(keyframes), "frames": int(motion["quat"].shape[0]),
                    "audioSeconds": round(len(wav) / sr, 3), "timing": {k: round(v, 3) for k, v in timing.items()}}


def rebase_bvh(text: str) -> str:
    """レスト姿勢を 1 フレーム目へ組み直し、cm → m にする(docs/sentiavatar-eval.md)。

    公式 BVH の OFFSET は UE のボーンローカル値で、全回転 0 のレストが
    「全ボーンが一直線」になる。各フレームの関節位置は変えずに、
    レスト = 1 フレーム目の立ち姿になるよう階層を組み直す。
    """
    sys.path.insert(0, str(_conversation_motion_path()))
    from conversation_motion.bvh import Motion, Skeleton, forward_kinematics, local_from_global, qinv, qmul, write_bvh, parse_bvh

    motion = parse_bvh(text)
    positions, rotations = forward_kinematics(motion)
    skeleton = motion.skeleton
    rebased_rot = qmul(rotations, qinv(rotations[0])[None])
    joints = []
    for j, joint in enumerate(skeleton.joints):
        offset = positions[0, j] - (positions[0, joint.parent] if joint.parent >= 0 else positions[0, j])
        end = None
        if joint.end_site is not None:
            end = qrotate_np(rotations[0, j], joint.end_site) / 100.0
        joints.append(type(joint)(joint.name, joint.parent, offset / 100.0, joint.channels, end))
    new_skeleton = Skeleton(joints)
    local = local_from_global(new_skeleton, rebased_rot)
    return write_bvh(Motion(new_skeleton, positions[:, 0] / 100.0, local, motion.fps))


def qrotate_np(q, v):
    from conversation_motion.bvh import qrotate
    return qrotate(q, v)


def _conversation_motion_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    for candidate in (repo / "backend/app/agent/general_conversation/tools",
                      repo / ".runtime/staging/backend-app/agent/general_conversation/tools"):
        if (candidate / "conversation_motion/bvh.py").exists():
            return candidate
    raise RuntimeError("conversation_motion パッケージが見つかりません")


class GenerateRequest(BaseModel):
    audio_wav_base64: str
    action_text: str = "动作：说话"
    seed: int | None = None
    temperature: float = 0.2
    top_p: float = 0.2
    generate_steps: int = 6
    format: str = "bvh"


app = FastAPI(title="SentiAvatar research server")
engine: Engine | None = None


@app.on_event("startup")
def _load() -> None:
    global engine
    engine = Engine()
    print(f"[sentiavatar] models loaded in {engine.load_seconds:.1f}s on {DEVICE}", flush=True)


@app.get("/health")
def health() -> dict:
    return {"ok": engine is not None, "device": DEVICE, "license": "SentiPulse-NC-1.0 (research only)",
            "loadSeconds": round(engine.load_seconds, 1) if engine else None}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    if engine is None:
        raise HTTPException(503, "models are loading")
    try:
        wav, sr = sf.read(io.BytesIO(base64.b64decode(request.audio_wav_base64)), dtype="float32")
    except Exception as error:  # noqa: BLE001
        raise HTTPException(400, f"audio decode failed: {error}") from error
    if len(wav) < sr * 0.3:
        raise HTTPException(400, "audio shorter than 0.3 s")
    try:
        result = engine.generate(wav, sr, request.action_text, request.seed, request.temperature,
                                 request.top_p, request.generate_steps)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(500, f"generation failed: {error}") from error
    result["bvh"] = rebase_bvh(result["bvh_raw"])
    result["actionText"] = request.action_text
    return result

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtraSkillSage main.py (secure config edition)
- TTS/Ollama の host(IP) をコードに直書きせず、外部config.jsonから読むのだ
- journald監視 → 文生成 →（接頭辞後ろだけ）Ollama要約 → TTS読み上げ の流れなのだ

依存:
  pip install requests simpleaudio
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import requests
import simpleaudio as sa


# -------------------------
# Config loader
# -------------------------
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/extraskillsage/config.json")
CONFIG_PATH = os.getenv("SAGE_CONFIG_PATH", DEFAULT_CONFIG_PATH)

def _load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告。設定ファイルが見つかりません。{path}。環境変数やデフォルトで動かします。", file=sys.stderr, flush=True)
        return {}
    except Exception as e:
        print(f"警告。設定ファイルの読み込みに失敗。{e}。環境変数やデフォルトで動かします。", file=sys.stderr, flush=True)
        return {}

CFG = _load_config(CONFIG_PATH)

def _cfg_get(d: dict, keys: List[str], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# Settings (config first, then env fallback)
# -------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in ("0", "false", "False", "no", "NO")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


# journal
COOLDOWN_SEC = _env_int("SAGE_COOLDOWN_SEC", int(_cfg_get(CFG, ["journal", "cooldown_sec"], 60)))
SUMMARY_INTERVAL_SEC = _env_int("SAGE_SUMMARY_SEC", int(_cfg_get(CFG, ["journal", "summary_sec"], 300)))
MAX_MESSAGE_LEN = _env_int("SAGE_MAX_MSG_LEN", int(_cfg_get(CFG, ["journal", "max_message_len"], 180)))

# journald units: config -> env JOURNAL_UNITS -> empty
_cfg_units = _cfg_get(CFG, ["journal", "units"], [])
if isinstance(_cfg_units, list):
    UNITS_FILTER = [str(u).strip() for u in _cfg_units if str(u).strip()]
else:
    UNITS_FILTER = []

_env_units = [u.strip() for u in _env_str("JOURNAL_UNITS", "").split(",") if u.strip()]
if _env_units:
    UNITS_FILTER = _env_units

BASE_CMD = ["journalctl", "-f", "-o", "json", "--no-pager"]

# TTS (host/port/path are sensitive -> from config only by default)
TTS_ENABLED = _env_bool("SAGE_TTS_ENABLED", bool(_cfg_get(CFG, ["tts", "enabled"], True)))
TTS_HOST = _env_str("SAGE_TTS_HOST", str(_cfg_get(CFG, ["tts", "host"], "")))   # env override allowed if you want
TTS_PORT = _env_int("SAGE_TTS_PORT", int(_cfg_get(CFG, ["tts", "port"], 0)))
TTS_PATH = _env_str("SAGE_TTS_PATH", str(_cfg_get(CFG, ["tts", "path"], "/voice")))
TTS_MODEL = _env_str("SAGE_TTS_MODEL", str(_cfg_get(CFG, ["tts", "model_name"], "zundamon")))
TTS_TIMEOUT_SEC = _env_float("SAGE_TTS_TIMEOUT_SEC", float(_cfg_get(CFG, ["tts", "timeout_sec"], 5.0)))

def _build_tts_url() -> str:
    if not TTS_HOST or not TTS_PORT:
        return ""
    return f"http://{TTS_HOST}:{TTS_PORT}{TTS_PATH}"

TTS_URL = _env_str("SAGE_TTS_URL", _build_tts_url())  # allow full URL override if needed

# Ollama (host/port are sensitive -> from config only by default)
OLLAMA_ENABLED = _env_bool("SAGE_OLLAMA_ENABLED", bool(_cfg_get(CFG, ["ollama", "enabled"], True)))
OLLAMA_HOST = _env_str("SAGE_OLLAMA_HOST", str(_cfg_get(CFG, ["ollama", "host"], "")))
OLLAMA_PORT = _env_int("SAGE_OLLAMA_PORT", int(_cfg_get(CFG, ["ollama", "port"], 11434)))
OLLAMA_MODEL = _env_str("SAGE_OLLAMA_MODEL", str(_cfg_get(CFG, ["ollama", "model"], "llama3")))
OLLAMA_TIMEOUT_SEC = _env_float("SAGE_OLLAMA_TIMEOUT_SEC", float(_cfg_get(CFG, ["ollama", "timeout_sec"], 6.0)))
OLLAMA_MAX_CACHE = _env_int("SAGE_OLLAMA_MAX_CACHE", int(_cfg_get(CFG, ["ollama", "max_cache"], 256)))


# -------------------------
# Core logic
# -------------------------
@dataclass
class Event:
    ts: float
    unit: str
    level: str
    kind: str
    detail: str
    raw_msg: str
    priority: int


PAT_SYSTEMD: List[Tuple[str, re.Pattern]] = [
    ("SYSTEMD_FAILED_START", re.compile(r"\bFailed to start\b", re.I)),
    ("SYSTEMD_FAILED_UNIT", re.compile(r"\bentered failed state\b|\bFailed with result\b", re.I)),
    ("SYSTEMD_EXITED", re.compile(r"\bMain process exited\b|\bcode=exited\b|\bstatus=\d+/", re.I)),
    ("SYSTEMD_RESTART", re.compile(r"\bScheduled restart job\b|\brestart counter is\b", re.I)),
    ("SYSTEMD_TOO_FAST", re.compile(r"\bStart request repeated too quickly\b", re.I)),
    ("SYSTEMD_STARTED", re.compile(r"\bStarted\b|\bStarting\b", re.I)),
]
PAT_OOM = re.compile(r"\boom-kill\b|\bOut of memory\b|\bKilled process\b", re.I)
PAT_UFW_BLOCK = re.compile(r"\[UFW BLOCK\]", re.I)


def _shorten(s: str, n: int = MAX_MESSAGE_LEN) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else (s[: n - 1] + "…")


def _get_unit(j: dict) -> str:
    unit = j.get("_SYSTEMD_UNIT") or j.get("SYSTEMD_UNIT") or ""
    if unit:
        return unit
    return j.get("SYSLOG_IDENTIFIER") or j.get("_COMM") or "unknown"


def _get_priority(j: dict) -> int:
    try:
        return int(j.get("PRIORITY", 6))
    except Exception:
        return 6


def _classify_kind(message: str) -> str:
    if PAT_UFW_BLOCK.search(message):
        return "UFW_BLOCK"
    if PAT_OOM.search(message):
        return "OOM"
    for kind, pat in PAT_SYSTEMD:
        if pat.search(message):
            return kind
    if re.search(r"\bwarning\b|\bwarn\b|\bcapability\b", message, re.I):
        return "APP_WARN"
    return "OTHER"


def _infer_level(kind: str, priority: int) -> str:
    if kind in ("OOM", "SYSTEMD_FAILED_START", "SYSTEMD_FAILED_UNIT", "SYSTEMD_EXITED", "SYSTEMD_TOO_FAST"):
        return "CRIT"
    if kind in ("SYSTEMD_RESTART", "APP_WARN", "UFW_BLOCK"):
        return "WARN"
    if priority <= 3:
        return "CRIT"
    if priority == 4:
        return "WARN"
    return "INFO"


def _parse_kv_tokens(message: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for tok in message.split():
        if "=" in tok and not tok.startswith("["):
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip().strip(",")
    return kv


def _render_ufw_block(message: str) -> str:
    kv = _parse_kv_tokens(message)
    in_if = kv.get("IN", "")
    src = kv.get("SRC", "")
    dst = kv.get("DST", "")
    proto = kv.get("PROTO", "")
    dpt = kv.get("DPT", "")

    parts = []
    if src and dst:
        parts.append(f"{src} から {dst} への通信")
    elif src:
        parts.append(f"送信元 {src} の通信")
    elif dst:
        parts.append(f"宛先 {dst} の通信")
    else:
        parts.append("通信")

    if proto:
        parts.append(f"{proto}")
    if dpt:
        parts.append(f"宛先ポート {dpt}")
    if in_if:
        parts.append(f"受信IF {in_if}")

    return "、".join(parts)


def _extract_detail(kind: str, msg: str) -> str:
    if kind == "UFW_BLOCK":
        return _render_ufw_block(msg)
    if kind == "SYSTEMD_RESTART":
        m = re.search(r"restart counter is (\d+)", msg, re.I)
        return f"再起動回数 {m.group(1)}" if m else "再起動が予定されています"
    if kind == "SYSTEMD_TOO_FAST":
        return "短時間に起動失敗が連続しています"
    if kind == "OOM":
        return "メモリ不足の兆候です"
    if kind == "SYSTEMD_STARTED":
        return "起動イベントです"
    return _shorten(msg)


def _compose_line(ev: Event, streak: int = 0) -> str:
    unit = ev.unit

    if ev.level == "CRIT":
        if ev.kind == "SYSTEMD_EXITED":
            return f"報告。{unit} が停止しました。{ev.detail}。"
        if ev.kind in ("SYSTEMD_FAILED_START", "SYSTEMD_FAILED_UNIT"):
            return f"報告。{unit} の起動に失敗。理由、{ev.detail}。"
        if ev.kind == "SYSTEMD_TOO_FAST":
            return f"報告。{unit} が起動失敗を連発。{ev.detail}。"
        if ev.kind == "OOM":
            return f"報告。{unit} 周辺でメモリ不足を検出。{ev.detail}。"
        return f"報告。{unit} で重大イベント。{ev.detail}。"

    if ev.level == "WARN":
        if ev.kind == "SYSTEMD_RESTART":
            extra = f"直近で {streak} 回目です。" if streak >= 2 else ""
            return f"警告。{unit} が再起動しています。{ev.detail}。{extra}".strip()
        if ev.kind == "UFW_BLOCK":
            return f"警告。ファイアウォールが通信を遮断。{ev.detail}。"
        return f"警告。{unit} に異常兆候。{ev.detail}。"

    if ev.kind == "SYSTEMD_STARTED":
        return f"報告。{unit} は起動しました。"

    return f"報告。{unit}。{ev.detail}。"


class Deduper:
    def __init__(self, cooldown_sec: int):
        self.cooldown = cooldown_sec
        self.last_emit: Dict[str, float] = {}

    def allow(self, key: str, now: float) -> bool:
        t = self.last_emit.get(key, 0.0)
        if now - t >= self.cooldown:
            self.last_emit[key] = now
            return True
        return False


# -------------------------
# Ollama summarize (after prefix)
# -------------------------
_ollama_session = requests.Session()
_ollama_cache: "OrderedDict[str, str]" = OrderedDict()
PREFIX_RE = re.compile(r"^(報告。|警告。)\s*")

def _ollama_ready() -> bool:
    # hostが空なら要約しない（設定ファイルに入ってない想定）なのだ
    return OLLAMA_ENABLED and bool(OLLAMA_HOST)

def ollama_generate(model_name: str, prompt: str, host: str, port: int) -> str:
    url = f"http://{host}:{port}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "prompt": prompt, "stream": False}
    r = _ollama_session.post(url, headers=headers, data=json.dumps(data), timeout=OLLAMA_TIMEOUT_SEC)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()

def summarize_after_prefix(line: str) -> str:
    if not _ollama_ready():
        return line

    m = PREFIX_RE.match(line)
    if not m:
        return line

    prefix = m.group(1)
    body = line[m.end():].strip()
    if not body:
        return line

    key = f"{prefix}|{body}"
    if key in _ollama_cache:
        _ollama_cache.move_to_end(key)
        return f"{prefix} {_ollama_cache[key]}"

    prompt = (
            "あなたは冷静なシステム監視アシスタントです。\n"
            "デスマス調で回答してください。\n"
            "大まかな内容だけ要約し、細かい数字は無視してください。\n"
            "次の文を、音声読み上げ向けに日本語で1文で要約してください。\n"
            "余計な前置きや敬語は不要なので、要約分のみ出力してください。\n"
            "アルファベットや英単語を使用するときは読みをひらがなに変換してください。\n"
            "日本語音声に変換するため、数字や英単語は使用しないでください。\n"
            f"文: {body}\n"
            "要約:"
        )
    )

    try:
        out = ollama_generate(OLLAMA_MODEL, prompt, host=OLLAMA_HOST, port=OLLAMA_PORT)
        out = _shorten(out, MAX_MESSAGE_LEN)
        if not out:
            return line

        _ollama_cache[key] = out
        _ollama_cache.move_to_end(key)
        while len(_ollama_cache) > OLLAMA_MAX_CACHE:
            _ollama_cache.popitem(last=False)

        return f"{prefix} {out}"
    except Exception as e:
        print(f"警告。要約に失敗。{e}。", file=sys.stderr, flush=True)
        return line


# -------------------------
# TTS
# -------------------------
_tts_session = requests.Session()

def _tts_ready() -> bool:
    # URLが空ならしゃべらないのだ
    return TTS_ENABLED and bool(TTS_URL)

def tts_speak(text: str) -> None:
    if not _tts_ready():
        return

    headers = {"accept": "audio/wav"}
    params = {
        "text": text,
        "encodeng": "utf-8",  # サーバ互換のため残すのだ
        "encoding": "utf-8",
        "model_name": TTS_MODEL,
    }

    try:
        r = _tts_session.post(TTS_URL, headers=headers, params=params, timeout=TTS_TIMEOUT_SEC)
        r.raise_for_status()
        wav_bytes = r.content

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        sa.play_buffer(frames, channels, sampwidth, framerate).wait_done()

    except requests.exceptions.RequestException as e:
        print(f"警告。音声サーバに接続できません。{e}。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"警告。音声出力に失敗。{e}。", file=sys.stderr, flush=True)


# -------------------------
# Runner
# -------------------------
def _build_cmd() -> List[str]:
    cmd = list(BASE_CMD)
    for u in UNITS_FILTER:
        cmd.extend(["-u", u])
    return cmd


def main() -> int:
    # 設定の最低限チェック（IPが設定ファイルに無いとき気づけるようにするのだ）
    if TTS_ENABLED and not TTS_URL:
        print("警告。TTSが有効ですがURLが未設定です。config.json の tts.host/port/path を確認するのだ。", file=sys.stderr, flush=True)
    if OLLAMA_ENABLED and not OLLAMA_HOST:
        print("警告。Ollamaが有効ですがhostが未設定です。config.json の ollama.host を確認するのだ。", file=sys.stderr, flush=True)

    cmd = _build_cmd()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    deduper = Deduper(COOLDOWN_SEC)
    restart_streak: Dict[str, int] = {}
    counts = {"CRIT": 0, "WARN": 0, "INFO": 0}
    last_summary = time.time()

    start_line = "報告。監視を開始します。"
    start_line = summarize_after_prefix(start_line)
    print(start_line, flush=True)
    tts_speak(start_line)

    assert proc.stdout is not None
    for line in proc.stdout:
        now = time.time()

        if now - last_summary >= SUMMARY_INTERVAL_SEC:
            scope = "全体" if not UNITS_FILTER else ", ".join(UNITS_FILTER)
            summary = (
                f"報告。直近{SUMMARY_INTERVAL_SEC//60}分の集計。"
                f"重大{counts['CRIT']}件、警告{counts['WARN']}件。監視対象は {scope}。"
            )
            summary = summarize_after_prefix(summary)
            print(summary, flush=True)
            tts_speak(summary)

            counts = {"CRIT": 0, "WARN": 0, "INFO": 0}
            last_summary = now

        line = line.strip()
        if not line:
            continue

        try:
            j = json.loads(line)
        except Exception:
            continue

        unit = _get_unit(j)
        msg = j.get("MESSAGE", "") or ""
        priority = _get_priority(j)

        kind = _classify_kind(msg)
        level = _infer_level(kind, priority)
        detail = _extract_detail(kind, msg)

        if kind == "SYSTEMD_RESTART":
            restart_streak[unit] = restart_streak.get(unit, 0) + 1
        elif kind.startswith("SYSTEMD_"):
            restart_streak[unit] = 0

        ev = Event(ts=now, unit=unit, level=level, kind=kind, detail=detail, raw_msg=msg, priority=priority)
        counts[level] += 1

        key = f"{unit}|{level}|{kind}"
        if not deduper.allow(key, now):
            continue

        out = _compose_line(ev, streak=restart_streak.get(unit, 0))
        out = summarize_after_prefix(out)

        print(out, flush=True)
        tts_speak(out)

    err = ""
    if proc.stderr:
        err = (proc.stderr.read() or "").strip()
    if err:
        msg = f"報告。journalctl の出力に問題。{_shorten(err)}。"
        msg = summarize_after_prefix(msg)
        print(msg, flush=True)
        tts_speak(msg)

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        end_line = "報告。監視を終了します。"
        end_line = summarize_after_prefix(end_line)
        print(end_line, flush=True)
        tts_speak(end_line)
        raise





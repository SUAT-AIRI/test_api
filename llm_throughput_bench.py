# -*- coding: utf-8 -*-
"""
llm_throughput_bench.py

一个用于测量 Chat Completions API 吞吐率/时延的通用压测脚本，
支持串行或并发（多线程）模式、流式/非流式两种请求方式。

用法示例：
  # 基础：串行、流式
  API_URL=http://<your-endpoint>/v1/chat/completions \
  API_KEY=sk-xxx \
  python llm_throughput_bench.py --questions questions.txt

  # 并发 8 路，循环复用问题直到发满 80 个请求
  API_URL=... API_KEY=... \
  python llm_throughput_bench.py --questions questions.txt \
      --concurrency 8 --total-requests 80

  # 非流式（一次性返回）
  API_URL=... API_KEY=... \
  python llm_throughput_bench.py --questions questions.txt --no-stream

环境变量：
  - API_URL (必需)   例如: http://<your-endpoint>/v1/chat/completions
  - API_KEY (必需)
  - ORGANIZATION_ID (可选)

输出：
  - results.csv：每条请求的明细（TTFT、总耗时、token 数、TPS 等）
  - 控制台：总体统计（成功率、TTFT/总耗时分位数、全局吞吐率等）

依赖：
  - requests
  - 可选：tiktoken（用于更准确的 token 统计；没有则退化到近似统计）
"""

import os
import re
import csv
import sys
import json
import time
import math
import random
import argparse
import threading
from typing import Callable, Dict, Any, Generator, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ===================== 环境变量 =====================
API_URL = os.getenv("API_URL", "")
API_KEY = os.getenv("API_KEY", "")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", "")

# ===================== 默认参数 =====================
DEFAULT_MODEL = ""
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_QUESTIONS_FILE = "questions.txt"
DEFAULT_CSV = "results.csv"
DEFAULT_TIMEOUT = 300
DEFAULT_CONCURRENCY = 10
DEFAULT_TOTAL_REQUESTS = 10
DEFAULT_STREAM = True
DEFAULT_SEED = 42

# ===================== Token 计数器 =====================
def _fallback_token_counter(text: str) -> int:
    """
    近似 token 统计：
    - 英文/数字连续串按 1 token
    - CJK 单字按 1 token
    - 其它符号按 1 token
    """
    if not text:
        return 0
    words = re.findall(r"[A-Za-z0-9]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    cleaned = re.sub(r"[A-Za-z0-9\u4e00-\u9fff]", " ", text)
    symbols = [s for s in cleaned.split() if s.strip()]
    return len(words) + len(cjk) + len(symbols)


def _build_token_counter() -> Callable[[str], int]:
    """
    优先使用 tiktoken 的 cl100k_base；失败则退化到 _fallback_token_counter
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s or ""))
    except Exception:
        return _fallback_token_counter


TOKEN_COUNT: Callable[[str], int] = _build_token_counter()

# ===================== 工具函数 =====================
def build_headers() -> Dict[str, str]:
    if not API_KEY:
        return {"Content-Type": "application/json"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    if ORGANIZATION_ID:
        headers["OpenAI-Organization"] = ORGANIZATION_ID
    return headers


def load_questions(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到问题文件: {path}")
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                qs.append(s)
    if not qs:
        raise ValueError("问题文件为空。请至少提供 1 条问题。")
    return qs


def percentile(sorted_vals: List[float], p: float) -> float:
    """p in [0,1]"""
    if not sorted_vals:
        return math.nan
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    idx = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def aggregate(nums: List[float]) -> Dict[str, float]:
    if not nums:
        return {"count": 0, "avg": math.nan, "p50": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan}
    s = sorted(nums)
    return {
        "count": len(nums),
        "avg": sum(nums) / len(nums),
        "p50": percentile(s, 0.5),
        "p95": percentile(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }


# ===================== 调用实现 =====================
def call_streaming(
    session: requests.Session,
    prompt_message: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, float, float]:
    """
    以流式返回，测量：
      - ttft_sec: first token 从发出请求到收到首个增量的时间
      - total_sec: 总耗时
    返回：(response_text, ttft_sec, total_sec)
    """
    if not API_URL or not API_KEY:
        raise RuntimeError("请先设置 API_URL 与 API_KEY 环境变量。")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    headers = build_headers()
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    chunks: List[str] = []

    with session.post(API_URL, headers=headers, data=json.dumps(payload), stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                # 某些实现可能会输出心跳/注释等
                continue
            if "choices" in obj and obj["choices"]:
                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(content)

    end = time.perf_counter()
    ttft = float("inf") if first_token_time is None else max(0.0, first_token_time - start)
    total = max(0.0, end - start)
    return ("".join(chunks), ttft, total)


def call_non_streaming(
    session: requests.Session,
    prompt_message: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, float, float]:
    """
    非流式：无法准确测 TTFT（记为 inf），只记总耗时。
    返回：(response_text, ttft_sec, total_sec)
    """
    if not API_URL or not API_KEY:
        raise RuntimeError("请先设置 API_URL 与 API_KEY 环境变量。")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    headers = build_headers()
    start = time.perf_counter()
    resp = session.post(API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    obj = resp.json()
    end = time.perf_counter()

    text = ""
    try:
        text = obj["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(obj, ensure_ascii=False)

    return (text, float("inf"), max(0.0, end - start))


def run_one(
    session: requests.Session,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    stream: bool,
) -> Dict[str, Any]:
    """
    返回一条请求的明细结果。
    """
    result: Dict[str, Any] = {
        "prompt": prompt,
        "prompt_tokens": TOKEN_COUNT(prompt),
        "ttft_sec": math.nan,
        "total_sec": math.nan,
        "response_text": "",
        "response_tokens": 0,
        "tokens_per_sec": math.nan,
        "ok": False,
        "error": "",
    }

    try:
        if stream:
            text, ttft, total = call_streaming(session, prompt, model, max_tokens, temperature, timeout)
        else:
            text, ttft, total = call_non_streaming(session, prompt, model, max_tokens, temperature, timeout)

        result["response_text"] = text or ""
        result["response_tokens"] = TOKEN_COUNT(result["response_text"])
        result["ttft_sec"] = ttft
        result["total_sec"] = total
        if total > 0 and result["response_tokens"] > 0:
            result["tokens_per_sec"] = result["response_tokens"] / total
        result["ok"] = True
    except Exception as e:
        result["error"] = repr(e)

    return result


# ===================== 汇总与输出 =====================
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def safe_vals(key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            if r.get("ok") and isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key])):
                out.append(float(r[key]))
        return out

    prompt_tokens = safe_vals("prompt_tokens")
    resp_tokens = safe_vals("response_tokens")
    ttft = safe_vals("ttft_sec")
    total = safe_vals("total_sec")
    tps = safe_vals("tokens_per_sec")

    return {
        "prompt_tokens": aggregate(prompt_tokens),
        "response_tokens": aggregate(resp_tokens),
        "ttft_sec": aggregate(ttft),
        "total_sec": aggregate(total),
        "tokens_per_sec": aggregate(tps),
        "success_rate": (sum(1 for r in rows if r.get("ok")) / len(rows)) if rows else 0.0,
    }


def print_summary(summary: Dict[str, Any], global_tps: Optional[float]) -> None:
    def line(title: str, stats: Dict[str, float], unit: str = ""):
        print(f"- {title:<18} n={stats['count']:>3} | "
              f"avg={stats['avg']:.4f}{unit} | "
              f"p50={stats['p50']:.4f}{unit} | "
              f"p95={stats['p95']:.4f}{unit} | "
              f"min={stats['min']:.4f}{unit} | "
              f"max={stats['max']:.4f}{unit}")

    print("\n======== Overall Summary ========")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    line("Prompt tokens", summary["prompt_tokens"])
    line("Response tokens", summary["response_tokens"])
    line("TTFT (sec)", summary["ttft_sec"], "s")
    line("Total (sec)", summary["total_sec"], "s")
    line("Per-req TPS", summary["tokens_per_sec"], " tok/s")
    if global_tps is not None:
        print(f"- Global TPS        {global_tps:.4f} tok/s (基于总返回 token / 墙钟总时长)")
    print("================================\n")


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fields = [
        "idx", "ok", "prompt", "prompt_tokens",
        "ttft_sec", "total_sec", "response_tokens",
        "tokens_per_sec", "error"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow({
                "idx": i,
                "ok": r.get("ok"),
                "prompt": r.get("prompt"),
                "prompt_tokens": r.get("prompt_tokens"),
                "ttft_sec": f"{r.get('ttft_sec'):.4f}" if isinstance(r.get("ttft_sec"), (int, float)) and math.isfinite(r.get("ttft_sec")) else r.get("ttft_sec"),
                "total_sec": f"{r.get('total_sec'):.4f}" if isinstance(r.get("total_sec"), (int, float)) and math.isfinite(r.get("total_sec")) else r.get("total_sec"),
                "response_tokens": r.get("response_tokens"),
                "tokens_per_sec": f"{r.get('tokens_per_sec'):.4f}" if isinstance(r.get("tokens_per_sec"), (int, float)) and math.isfinite(r.get("tokens_per_sec")) else r.get("tokens_per_sec"),
                "error": r.get("error"),
            })


# ===================== 主逻辑 =====================
def make_workload(questions: List[str], total_requests: int, seed: int, shuffle: bool) -> List[str]:
    """把问题列表扩展/循环到 total_requests 条"""
    rng = random.Random(seed)
    items = questions[:]
    if shuffle:
        rng.shuffle(items)
    if len(items) >= total_requests:
        return items[:total_requests]
    # 轮询补足
    out = []
    i = 0
    while len(out) < total_requests:
        out.append(items[i % len(items)])
        i += 1
    return out


def run_sequential(
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    stream: bool,
    tiny_sleep: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with requests.Session() as session:
        for idx, q in enumerate(prompts, start=1):
            print(f"[{idx:03d}] PromptTokens={TOKEN_COUNT(q)} | 发送请求 …")
            r = run_one(session, q, model, max_tokens, temperature, timeout, stream)
            rows.append(r)
            if r["ok"]:
                ttft_s = "inf" if not math.isfinite(r["ttft_sec"]) else f"{r['ttft_sec']:.3f}"
                print(f"    ✓ OK | TTFT={ttft_s}s | Total={r['total_sec']:.3f}s | "
                      f"RespTokens={r['response_tokens']} | TPS={0.0 if math.isnan(r['tokens_per_sec']) else r['tokens_per_sec']:.2f} tok/s")
            else:
                print(f"    ✗ ERR | {r['error']}")
            if tiny_sleep > 0:
                time.sleep(tiny_sleep)
    return rows


def worker_task(
    session: requests.Session,
    worker_id: int,
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    stream: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, q in enumerate(prompts, start=1):
        r = run_one(session, q, model, max_tokens, temperature, timeout, stream)
        out.append(r)
        tag = f"[W{worker_id:02d}-{i:03d}]"
        if r["ok"]:
            ttft_s = "inf" if not math.isfinite(r["ttft_sec"]) else f"{r['ttft_sec']:.3f}"
            print(f"{tag} ✓ OK | TTFT={ttft_s}s | Total={r['total_sec']:.3f}s | "
                  f"RespTokens={r['response_tokens']} | TPS={0.0 if math.isnan(r['tokens_per_sec']) else r['tokens_per_sec']:.2f} tok/s")
        else:
            print(f"{tag} ✗ ERR | {r['error']}")
    return out


def run_concurrent(
    prompts: List[str],
    concurrency: int,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    stream: bool,
) -> List[Dict[str, Any]]:
    """
    将总的 prompts 切成 concurrency 份，分别由 worker 线程跑。
    每个 worker 复用一个 requests.Session（对端若要求连接复用更有效）。
    """
    # 均分
    shards: List[List[str]] = [[] for _ in range(concurrency)]
    for i, p in enumerate(prompts):
        shards[i % concurrency].append(p)

    results_lock = threading.Lock()
    all_rows: List[Dict[str, Any]] = []

    def run_worker(wid: int, shard: List[str]) -> None:
        with requests.Session() as sess:
            rows = worker_task(sess, wid, shard, model, max_tokens, temperature, timeout, stream)
            with results_lock:
                all_rows.extend(rows)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_worker, i + 1, shard) for i, shard in enumerate(shards)]
        for _ in as_completed(futures):
            pass

    return all_rows


def main():
    if not API_URL or not API_KEY:
        print("错误：请先设置环境变量 API_URL 与 API_KEY。")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="LLM 吞吐率压测工具（串行/并发，流式/非流式）")
    parser.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS_FILE, help="问题文件（每行一个问题）")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="结果 CSV 输出文件名")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="每次请求的超时秒数")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发线程数（1 = 串行）")
    parser.add_argument("--total-requests", type=int, default=DEFAULT_TOTAL_REQUESTS, help="总请求数（会循环复用问题）")
    parser.add_argument("--no-stream", action="store_true", help="使用非流式请求（默认走流式）")
    parser.add_argument("--shuffle", action="store_true", help="在构建 workload 时打乱问题顺序")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（用于 --shuffle）")
    parser.add_argument("--tiny-sleep", type=float, default=0.0, help="串行模式下每次请求之间的节流秒数")
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions)
    except Exception as e:
        print(f"加载问题失败：{e}")
        sys.exit(1)

    prompts = make_workload(questions, args.total_requests, seed=args.seed, shuffle=args.shuffle)

    print(f"读取到 {len(questions)} 个原始问题；本次将发起 {len(prompts)} 次请求。")
    print(f"模式：{'并发' if args.concurrency > 1 else '串行'} | "
          f"{'流式' if not args.no_stream else '非流式'} | "
          f"并发度={args.concurrency} | 模型={args.model}\n")

    wall_start = time.perf_counter()

    if args.concurrency <= 1:
        rows = run_sequential(
            prompts=prompts,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stream=not args.no_stream,
            tiny_sleep=args.tiny_sleep,
        )
    else:
        rows = run_concurrent(
            prompts=prompts,
            concurrency=args.concurrency,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stream=not args.no_stream,
        )

    wall_end = time.perf_counter()
    wall_time = max(0.0, wall_end - wall_start)
    total_resp_tokens = sum(r.get("response_tokens", 0) for r in rows if r.get("ok"))
    global_tps = (total_resp_tokens / wall_time) if wall_time > 0 else None

    # 保存明细
    save_csv(rows, args.csv)
    print(f"\n明细已保存：{args.csv}")

    # 汇总
    summary = summarize(rows)
    print_summary(summary, global_tps)


if __name__ == "__main__":
    main()

# LLM Throughput Bench

A tiny, **OpenAI‑compatible** CLI tool to benchmark Chat Completions API latency & throughput — supports **serial or concurrent** load, **streaming or non‑streaming** modes, accurate **per‑request** metrics (TTFT / total latency / tok/s), and a **global throughput** number based on wall‑clock time.

> Works with any endpoint that speaks the OpenAI Chat Completions protocol.

---

## ✨ Features

* **OpenAI‑compatible**: point to any `v1/chat/completions` endpoint.
* **Streaming & non‑streaming**: measure TTFT precisely in streaming; fall back gracefully otherwise.
* **Concurrency**: simple thread‑based load gen (`--concurrency N`).
* **Deterministic workloads**: reuse / shuffle questions, fix the seed.
* **CSV logs**: one row per request with all the details.
* **Global throughput**: total returned tokens ÷ wall‑clock time.
* **No hard deps**: optional `tiktoken` for better token counting; otherwise a reasonable fallback.

---

## 📦 Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`

```txt
requests>=2.31.0
tiktoken>=0.7.0  # optional
```

---

## 🔧 Configure

Set environment variables for your endpoint:

```bash
export API_URL="http://<your-endpoint>/v1/chat/completions"
export API_KEY="sk-xxxxx"
# optional
export ORGANIZATION_ID="org_xxx"
```

> If your endpoint requires a custom header, adapt `build_headers()` in the script.

---

## ▶️ Run

### 1) Basic (serial, streaming)

```bash
python llm_throughput_bench.py --questions questions.txt
```

### 2) Concurrency (8 threads, 80 total requests)

```bash
python llm_throughput_bench.py \
  --questions questions.txt \
  --concurrency 8 \
  --total-requests 80
```

### 3) Non‑streaming

```bash
python llm_throughput_bench.py --questions questions.txt --no-stream
```

### 4) Shuffle & seed (deterministic)

```bash
python llm_throughput_bench.py --questions questions.txt --shuffle --seed 42
```

### 5) Throttle serial calls (light pacing)

```bash
python llm_throughput_bench.py --questions questions.txt --tiny-sleep 0.2
```

---

## 📚 CLI Options

```text
--questions PATH          Question file (one per line). Default: questions.txt
--csv PATH                Output CSV filename. Default: results.csv
--model NAME              Model name. Default: DeepSeek-R1-bf16-hfd-w8a8
--max-tokens N            max_tokens per request. Default: 8192
--temperature FLOAT       Temperature. Default: 0.7
--timeout SECONDS         Per-request timeout. Default: 300
--concurrency N           Number of worker threads. Default: 1 (serial)
--total-requests N        Total number of requests (questions will be reused). Default: 10
--no-stream               Use non-streaming mode (default is streaming)
--shuffle                 Shuffle questions when building workload
--seed INT                RNG seed used with --shuffle. Default: 42
--tiny-sleep FLOAT        Serial mode only: sleep between requests (seconds)
```

---

## 📝 questions.txt (example)

Put **one prompt per line**; blank lines are ignored. Example:

```txt
Explain the difference between precision and recall with a small numerical example.
给我一份关于Transformer位置编码(PE/ALiBi/RoPE)的直观解释，最好配图思路。
Write a Python function that computes the Levenshtein distance and show its complexity.
Summarize the main ideas behind variational autoencoders in 5 bullet points.
设计一个小型数据库表结构来存储论文、作者、机构与引用关系，并给出3条示例SQL查询。
What are the trade-offs between Mixture-of-Experts and dense LLMs? Provide a table.
Provide a step-by-step guide to implement binary search on a rotated sorted array.
比较RAG与Agentic RAG的架构差异、适用场景与评估指标。
Why does label smoothing help generalization? Any caveats?
用通俗比喻解释马尔可夫决策过程(MDP)以及价值迭代的核心思想。
```

> The tool will **cycle** through your questions when `--total-requests` exceeds the number of lines.

---

## 📊 Outputs

### Console summary

* **Success Rate**
* **Prompt tokens** (avg/p50/p95/min/max)
* **Response tokens** (avg/p50/p95/min/max)
* **TTFT** (Time‑To‑First‑Token; streaming only, else `inf`)
* **Total latency** (request start → done)
* **Per‑request TPS** (response tokens ÷ total sec)
* **Global TPS** *(wall‑clock throughput)*: $\texttt{sum(response tokens)} / \texttt{wall time}$

### CSV schema (`results.csv`)

| Column           | Meaning                                                |
| ---------------- | ------------------------------------------------------ |
| idx              | Request index (1‑based)                                |
| ok               | Boolean success flag                                   |
| prompt           | Original user prompt                                   |
| prompt\_tokens   | Token count of prompt                                  |
| ttft\_sec        | Time to first token (streaming; otherwise `inf`)       |
| total\_sec       | End‑to‑end latency in seconds                          |
| response\_tokens | Token count of generated text                          |
| tokens\_per\_sec | Per‑request throughput = response\_tokens / total\_sec |
| error            | Exception text, if any                                 |

> Token counting uses `tiktoken` if available (`cl100k_base`); otherwise a CJK‑aware heuristic.

---

## 🧪 Methodology & Metrics

* **TTFT**: Captured from the moment the HTTP request is sent until the first streamed content token is observed. Useful for *perceived responsiveness*.
* **Total latency**: Full request round‑trip time.
* **Per‑request TPS**: Each request’s `response_tokens / total_sec`.
* **Global TPS (wall‑clock)**: Sum of all successful response tokens divided by the **overall elapsed time** of the run. Reflects system‑level throughput with concurrency.

> In non‑streaming mode, TTFT is undefined and recorded as `inf`.

---

## 🧰 Tips for Fair Benchmarking

* Pin `--model`, `--max-tokens`, and `--temperature` across runs.
* Use the **same workload** (same `questions.txt`; optionally `--shuffle --seed` for randomization that’s reproducible).
* Warm up once before measuring.
* Prefer **streaming** when you care about perceived latency (TTFT).
* If testing rate limits, gradually increase `--concurrency` and watch **success rate** & errors.
* Keep an eye on **server logs** and GPU/CPU utilization to contextualize numbers.

---

## 🛡️ Troubleshooting

**401 Unauthorized**

* Check `API_KEY`. Some backends require a prefix (`sk-`), some don’t.

**404 or 405**

* Verify `API_URL` ends with `/v1/chat/completions` and the method is POST.

**429 Too Many Requests**

* Reduce `--concurrency`, add backoff (you can insert sleeps or a retry wrapper if your backend allows).

**Timeouts**

* Increase `--timeout` or reduce output lengths.

**Garbled streaming / JSONDecodeError**

* Some servers interleave heartbeats or comments; the script drops non‑JSON `data:` lines safely.

**Token counts look off**

* Install `tiktoken` for more accurate counts.

---

## 🔒 Security Notes

* API keys are read from env vars and **never** written to disk.
* Avoid committing real keys in examples or CI.
* When filing issues, sanitize logs.

---

## 🔁 Reproducibility

* The workload is deterministic with `--seed`.
* Concurrency uses threads; exact scheduling still depends on OS/runtime.
* Report: OS, Python, `requests`, `tiktoken` versions, and exact CLI.

---

## 🤝 Compatibility

Tested against OpenAI‑compatible servers. If your provider uses extra headers or different fields, tweak `build_headers()` or payload in `llm_throughput_bench.py`.

---

## 📄 License

MIT

---

## 🙌 Contributing

Issues and PRs are welcome! Please include:

* Endpoint type & version
* Full CLI
* A redacted snippet of `results.csv`
* Any server‑side logs/metrics you can share

---

## 📁 Project Layout

```
.
├─ llm_throughput_bench.py   # the CLI tool
├─ requirements.txt          # requests (+ optional tiktoken)
└─ questions.txt             # your prompts (one per line)
```

Happy benchmarking! 🚀

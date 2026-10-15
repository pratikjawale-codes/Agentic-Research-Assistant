# Agentic AI Research Assistant

An autonomous multi-agent research and task execution platform built on **FastAPI**, **LangGraph**, and **Hybrid RAG** (Dense Vector + Sparse BM25 Fusion).

Given a high-level goal, the system decomposes the objective, queries private documents and the live web, computes precise arithmetic figures using validated tools, and produces a structured, citation-grounded research report.

---

## 🏗️ Architecture & Component Flow

```
┌──────────────────────┐
│   FastAPI Backend    │ ← JWT / API Key Dual-Auth, Multi-Tenant Data Isolation
└──────────┬───────────┘
           │ Injects user_id via server-side config
           ▼
┌─────────────────────────────────────────────┐
│          SUPERVISOR AGENT (LangGraph)       │
│  1. Intent Understanding (Pydantic schema)  │
│  2. Dynamic Upfront Planning & Decomposition│
│  3. Deterministic Routing & Step Budgets    │
└───────┬──────────────┬──────────────┬───────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  RESEARCH   │ │DATA ANALYST │ │   REPORT    │
│    AGENT    │ │    AGENT    │ │    AGENT    │
│  Evidence   │ │ Quantitative│ │   Grounded  │
│  Collection │ │ Calculations│ │  Synthesis  │
└──────┬──────┘ └──────┬──────┘ └─────────────┘
       │               │                ▲
       │               │                │ (No Tools:
       ▼               ▼                │  Prevents Wandering)
┌────────────────── TOOL MANAGER ───────┴─────────────────┐
│ • retrieve_documents (Hybrid Vector + BM25, Cutoff)     │
│ • web_search (DuckDuckGo / Tavily with TTL Cache)       │
│ • get_market_stat (Parameterized Read-Only SQL)         │
│ • cagr (Arithmetic Tool: ((End/Start)^(1/n)) - 1)       │
│ • Standard Envelope Validation:                         │
│   {status: 'ok'|'empty'|'error', data, sources, message}│
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation & Verification

Benchmark metrics across multi-domain research tasks (`python -m agentic_ai_research_assistant.eval.run_benchmark`):

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Recall@5 (Hit Rate)** | **85.0%** | Share of test queries where relevant chunks appear in the top 5 fused results. |
| **Tool Success Rate** | **82.0%** | Tool calls executed successfully without errors and passed schema validation. |
| **Task Completion** | **88.0%** | End-to-end task fulfillment rate at 0.8 completion threshold. |
| **Anti-Hallucination Cutoff** | **95.0%** | Correct refusal rate on unanswerable queries (`not found in uploaded documents`). |
| **Arithmetic Precision** | **100.0%** | Calculations delegated to deterministic code rather than LLM text prediction. |

### Retrieval Ablation:
- **Vector-Only Retrieval**: 72.0% Recall@5
- **BM25-Only Retrieval**: 68.0% Recall@5
- **Hybrid Retrieval (RRF Fusion)**: **85.0% Recall@5**

---

## 🔒 Security & Multi-Tenancy

- **Multi-Tenant Data Isolation**: `user_id` is derived from verified JWT tokens or hashed API keys and injected server-side via `RunnableConfig`. The LLM never provides `user_id`, preventing confused-deputy prompt-injection attacks.
- **Constant-Time Verification**: API keys (`rk_...`) are shown once, hashed with SHA-256, and verified with `hmac.compare_digest`.
- **SQL Injection Prevention**: Parameterized queries and strict allow-listed metrics prevent malicious SQL execution.
- **Anti-Hallucination Score Cutoff**: Cosine similarity threshold (default `0.35`) stops low-confidence RAG generation.

---

## 🚀 Quickstart

Install dependencies with [uv](https://docs.astral.sh/uv/), then start the API and dashboard together:

```bash
uv sync
uv run python application.py
```

Open the dashboard at `http://127.0.0.1:8501` and the API documentation at `http://127.0.0.1:8000/docs`. Press **Ctrl+C** to stop both services. The launcher waits for the API health check before starting the dashboard.

The application works without provider credentials using its local workflow. To use a hosted model, add its key to `.env` and install the matching optional integration:

```bash
uv sync --extra openai     # or: --extra gemini / --extra anthropic
```

Provider keys remain blank until you configure them. `.env.example` is the blank-key template; copy it to `.env` only if you do not already have a local `.env` file.

### Launcher commands

```bash
uv run python application.py                   # Start API and dashboard together
uv run python application.py --serve           # Start only the API
uv run python application.py --ui              # Start only the dashboard
uv run python application.py --verify           # Smoke-test tools and agent workflow
uv run python application.py --test             # Run tests
uv run python application.py --benchmark        # Run evaluation benchmarks
uv run python application.py -q "Research goal" # Run a CLI research task
```

## 🐳 Docker Deployment

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`
- SQLite data is stored in a persistent Docker volume.

## 👤 Author and license

Created by **Pratik Jawale** and distributed under the [MIT License](./LICENSE).

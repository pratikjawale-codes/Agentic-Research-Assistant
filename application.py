import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_ai_research_assistant.agents.graph import research_graph
from agentic_ai_research_assistant.database.session import init_db
from agentic_ai_research_assistant.knowledge_base.bm25 import global_bm25_manager
from agentic_ai_research_assistant.knowledge_base.chunker import create_chunks
from agentic_ai_research_assistant.knowledge_base.vector_store import global_vector_store
from agentic_ai_research_assistant.tools.calculator_tool import cagr
from agentic_ai_research_assistant.tools.db_tool import get_market_stat

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def verify_interconnectivity() -> bool:
    """Smoke-test the local database, tools, retrieval, and agent workflow."""
    print("[1/4] Initializing the local database...")
    asyncio.run(init_db())

    print("[2/4] Checking calculator and database tools...")
    assert cagr.invoke({"start_value": 100.0, "end_value": 200.0, "years": 3.0})["status"] == "ok"
    assert get_market_stat.invoke(
        {"metric": "baseline_market_valuation", "year": 2024}
    )["status"] == "ok"

    print("[3/4] Checking document indexing and retrieval...")
    user_id = 2_000_000_000
    doc_id = int(datetime.now().timestamp())
    chunks = create_chunks(
        [(1, "Target revenue expansion is $300M with 18% annual growth.")],
        user_id=user_id,
        doc_id=doc_id,
        filename="verification.txt",
    )
    global_vector_store.add_chunks(chunks)
    global_bm25_manager.invalidate(user_id)
    global_bm25_manager.get_or_build_index(user_id, chunks)

    print("[4/4] Checking the keyless, document-grounded research workflow...")
    try:
        output = research_graph.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": "Only search my document for target revenue expansion and calculate 5-year CAGR.",
                }],
                "intent": {},
                "plan": [],
                "current_step": 0,
                "findings": {},
                "numbers": {},
                "report": "",
                "replans": 0,
                "errors": [],
            },
            config={"configurable": {"user_id": str(user_id), "thread_id": f"{user_id}:verify"}},
        )
        assert output.get("report"), "Research workflow did not produce a report."
        print("      Local database, tools, retrieval, and agent workflow are operational.")
    finally:
        global_vector_store.delete_by_doc_id(user_id, doc_id)
        global_bm25_manager.invalidate(user_id)
    return True


def run_research_task(query: str, user_id: int = 1) -> None:
    """Run and print a research workflow from the command line."""
    print("=" * 70)
    print(f"RESEARCH GOAL: {query}")
    print(f"User ID: {user_id} | Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    output = research_graph.invoke(
        {
            "messages": [{"role": "user", "content": query}],
            "intent": {},
            "plan": [],
            "current_step": 0,
            "findings": {},
            "numbers": {},
            "report": "",
            "replans": 0,
            "errors": [],
        },
        config={"configurable": {"user_id": str(user_id), "thread_id": f"{user_id}:main_cli"}},
    )
    print("\n--- Research Plan ---")
    for step in output.get("plan", []):
        print(f"  * Step {step['id']} [{step['agent'].upper()}]: {step['description']}")
    print("\n--- Grounded Research Report ---")
    print(output.get("report", "No report generated."))
    print("=" * 70)


def start_api() -> None:
    """Start the FastAPI backend."""
    import uvicorn

    uvicorn.run(
        "agentic_ai_research_assistant.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


def start_ui() -> None:
    """Start the Streamlit dashboard."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "src" / "agentic_ai_research_assistant" / "ui" / "streamlit_app.py"),
            "--server.address=127.0.0.1",
            "--server.port=8501",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def start_application() -> None:
    """Run the API and dashboard together, stopping both when either exits."""
    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "agentic_ai_research_assistant.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    ui_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "src" / "agentic_ai_research_assistant" / "ui" / "streamlit_app.py"),
        "--server.address=127.0.0.1",
        "--server.port=8501",
    ]
    processes: list[subprocess.Popen] = []

    try:
        processes.append(subprocess.Popen(api_command, cwd=PROJECT_ROOT))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if processes[0].poll() is not None:
                raise RuntimeError("The API server exited before it became healthy.")
            try:
                with urlopen("http://127.0.0.1:8000/health", timeout=2):
                    break
            except (OSError, URLError):
                time.sleep(0.5)
        else:
            raise RuntimeError("The API server did not become healthy within 60 seconds.")

        processes.append(subprocess.Popen(ui_command, cwd=PROJECT_ROOT))
        print("Research Assistant is running:")
        print("  Dashboard: http://127.0.0.1:8501")
        print("  API docs:  http://127.0.0.1:8000/docs")
        print("Press Ctrl+C to stop both services.")

        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        failed = [(name, process.returncode) for name, process in zip(("API", "dashboard"), processes)
                  if process.returncode not in (None, 0)]
        if failed:
            raise RuntimeError(f"Application service exited unexpectedly: {failed}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic AI Research Assistant")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--serve", "-s", action="store_true", help="Start only the FastAPI backend")
    actions.add_argument("--ui", "-u", action="store_true", help="Start only the Streamlit dashboard")
    actions.add_argument("--test", "-t", action="store_true", help="Run the test suite")
    actions.add_argument("--benchmark", "-b", action="store_true", help="Run the evaluation benchmark")
    actions.add_argument("--verify", action="store_true", help="Smoke-test local project connectivity")
    actions.add_argument("--query", "-q", type=str, help="Run a research query from the command line")
    args = parser.parse_args()

    if args.serve:
        start_api()
    elif args.ui:
        start_ui()
    elif args.test:
        return subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=PROJECT_ROOT).returncode
    elif args.benchmark:
        return subprocess.run(
            [sys.executable, "-m", "agentic_ai_research_assistant.eval.run_benchmark"],
            cwd=PROJECT_ROOT,
        ).returncode
    elif args.verify:
        verify_interconnectivity()
    elif args.query:
        run_research_task(args.query)
    else:
        start_application()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

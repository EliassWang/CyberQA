"""Interactive CyberQA chat session.

Builds the multi-agent graph once, then repeatedly prompts for a question
and renders the structured result (answer/refusal, sources) in a rich
terminal UI.

Run with: uv run python main.py
"""

import os
import platform
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from agent.graph import build_graph, extract_result
from agent.retrieve_tool import COLLECTION
from agent.retrieval.retrieve import qdrant_status
from agent.retrieval.retrieve import warm_up as warm_up_embedder
from agent.retrieval.rerank import warm_up as warm_up_reranker
from llm.provider import load_client
from rag.device import probe_gpu


BANNER_ART = """
 ███  █   █ ████  █████ ████   ███   ███
█   █ █   █ █   █ █     █   █ █   █ █   █
█      █ █  ████  ████  ████  █   █ █████
█       █   █   █ █     █  █  █  ██ █   █
█   █   █   █   █ █     █   █ █   █ █   █
 ███    █   ████  █████ █   █  ████ █   █
""".strip("\n")

BANNER = (
    "[bold cyan]CyberQA[/bold cyan] — cybersecurity question answering, "
    "grounded in retrieved evidence.\n"
    "Type a question, or [bold]exit[/bold] / [bold]quit[/bold] to leave."
)


def detect_hardware() -> dict:
    """Detect the CPU and GPU available to the local embedding/reranking models.

    A GPU only counts as usable once a real tensor op on it succeeds
    (`rag.device.probe_gpu`) -- `torch.cuda.is_available()` alone doesn't
    catch a broken or mismatched NVIDIA driver, which the models actually
    hit at load time and fall back from too.
    """
    usable, gpu_name, error = probe_gpu()
    return {
        "cpu": platform.processor() or platform.machine() or "unknown CPU",
        "cores": os.cpu_count(),
        "gpu_usable": usable,
        "gpu_name": gpu_name,
        "gpu_error": error,
    }


def _hardware_status_line() -> str:
    """Format the CPU/GPU line for the local embedding and reranking models."""
    hw = detect_hardware()
    cores = f"{hw['cores']} cores" if hw["cores"] else "unknown cores"
    if hw["gpu_usable"]:
        return f"[green]●[/green] CPU: {hw['cpu']} ({cores})  |  GPU: {hw['gpu_name']}"
    elif hw["gpu_name"]:
        return (
            f"[yellow]●[/yellow] CPU: {hw['cpu']} ({cores})  |  "
            f"GPU: {hw['gpu_name']} detected but its drivers failed a test op "
            f"({hw['gpu_error']}) — falling back to CPU"
        )
    return f"[yellow]●[/yellow] CPU: {hw['cpu']} ({cores})  |  GPU: none detected, using CPU"

def render_result(console: Console, result: dict) -> None:
    """Print an answer or refusal, then a sources table if there is one."""
    if result.get("answer"):
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
    else:
        console.print(Panel(result.get("message") or "No answer.", title="Refused", border_style="red"))

    sources = result.get("sources") or []
    if sources:
        table = Table(title="Sources", header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Source")
        for i, chunk in enumerate(sources, start=1):
            table.add_row(str(i), chunk.get("source", "?"))
        console.print(table)


def _qdrant_status_line() -> str:
    """Format the Qdrant reachability/chunk-count line."""
    status = qdrant_status(COLLECTION)
    if not status["reachable"]:
        return f"[red]●[/red] Qdrant unreachable at {status['url']} — [dim]{status['error']}[/dim]"
    elif status["count"] is None:
        return f"[yellow]●[/yellow] Qdrant live at {status['url']}, but {status['error']}"
    return f"[green]●[/green] Qdrant live at {status['url']} — {COLLECTION}: {status['count']:,} chunks"


def _check_llm_credentials(console: Console) -> None:
    """Fail fast if LLM_API_KEY/LLM_MODEL aren't set, before touching Qdrant at all."""
    try:
        load_client()
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}.[/bold red] Set it in .env, then try again.")
        raise SystemExit(1)


def _ensure_qdrant_running(console: Console) -> None:
    """Start the local Qdrant container (scripts/qdrant.sh) if it isn't already reachable."""
    if qdrant_status(COLLECTION)["reachable"]:
        return

    console.print("[dim]Qdrant isn't reachable, attempting to start it...[/dim]")
    qdrant_sh = Path(__file__).parent / "scripts" / "qdrant.sh"
    result = subprocess.run(["bash", "-c", f"source {qdrant_sh} && ensure_qdrant_running"])
    if result.returncode != 0 or not qdrant_status(COLLECTION)["reachable"]:
        console.print("[bold red]Could not start Qdrant.[/bold red] Start it yourself, or point QDRANT_URL at a running instance.")
        raise SystemExit(1)


def main() -> None:
    console = Console()

    _check_llm_credentials(console)
    _ensure_qdrant_running(console)

    art = "\n".join(
        f"[bold cyan]{line[:29]}[/bold cyan] [bold]{line[30:]}[/bold]"
        for line in BANNER_ART.splitlines()
    )
    body = f"{art}\n\n{BANNER}\n\n{_qdrant_status_line()}\n{_hardware_status_line()}"
    console.print(Panel.fit(body, border_style="cyan"))

    with console.status("[dim]Starting up (loading local models)...[/dim]"):
        app = build_graph()
        warm_up_embedder()
        warm_up_reranker()

    while True:
        try:
            query = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        with console.status("[dim]thinking...[/dim]", spinner="dots"):
            try:
                raw = app.invoke({"query": query}, config={"recursion_limit": 50})
            except Exception as exc:  # surface any failure and keep the session alive
                console.print(f"[bold red]Error:[/bold red] {exc}")
                continue

        render_result(console, extract_result(raw))

    console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()

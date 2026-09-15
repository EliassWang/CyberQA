#!/usr/bin/env bash
# Run CyberQA's report evaluations: sample queries, full benchmark, retrieval
# ablation, and Scope Gate FPR. Each subcommand forwards extra arguments to
# its underlying script (see --help on that script for the full option list).
#
# Requires Qdrant running with a populated collection (starts a local one via
# scripts/qdrant.sh if QDRANT_URL isn't already reachable) and LLM_API_KEY /
# LLM_MODEL set -- see README.md.
#
# Run with: ./eval.sh <sample|full|ablation|fpr|all> [args...]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage: ./eval.sh <command> [args...]

  sample              10 example queries (scripts/run_sample_queries.py)
  full [args...]      Full benchmark eval: rewrites CTIBench questions first
                       if needed, then evaluates via benchmark.scripts.eval
                       (default: --task all --start 0 --end 20)
  ablation [args...]  Retrieval-only ablation sweep, alpha x rerank x k
                       (benchmark.scripts.retrieval_ablation; default: --n 50 --seed 42)
  fpr [args...]       Scope Gate false-positive-rate check
                       (benchmark.scripts.scope_gate_fpr; default: --task ate)
  all                 Run sample, full, ablation, and fpr in sequence, with
                       each one's own defaults

Extra arguments replace a command's defaults and are forwarded as-is, e.g.:
  ./eval.sh full --task attackqa --n 200 --seed 42
  ./eval.sh fpr --task attackqa --n 1000
  ./eval.sh ablation --n 1000

Results are written under benchmark/result/ (see --output on any script to
override). --help on any underlying script (e.g.
uv run python -m benchmark.scripts.eval --help) lists all its options.
EOF
}

# --task's value from a command's forwarded args, or a default if not given.
task_from_args() {
    local default="$1"; shift
    local prev=""
    for arg in "$@"; do
        if [ "$prev" = "--task" ]; then
            echo "$arg"
            return
        fi
        prev="$arg"
    done
    echo "$default"
}

ensure_qdrant() {
    set -a
    # shellcheck disable=SC1091
    [ -f .env ] && source .env
    set +a
    # shellcheck disable=SC1091
    source scripts/qdrant.sh
    ensure_qdrant_running
}

# Idempotent: skips rows that already have a rewritten `question` column.
ensure_ctibench_rewritten() {
    echo "==> Ensuring CTIBench questions are rewritten (skips rows already done)..."
    uv run python -m benchmark.scripts.rewrite_ctibench_questions
}

cmd_sample() {
    echo "==> Running sample queries..."
    uv run python -m scripts.run_sample_queries "$@"
}

cmd_full() {
    local args=(--task all --start 0 --end 20)
    [ "$#" -gt 0 ] && args=("$@")
    case "$(task_from_args all "${args[@]}")" in
        ate|all) ensure_ctibench_rewritten ;;
    esac
    echo "==> Running full benchmark eval (${args[*]})..."
    uv run python -m benchmark.scripts.eval "${args[@]}"
}

cmd_ablation() {
    local args=(--n 50 --seed 42)
    [ "$#" -gt 0 ] && args=("$@")
    echo "==> Running retrieval ablation sweep (${args[*]})..."
    uv run python -m benchmark.scripts.retrieval_ablation "${args[@]}"
}

cmd_fpr() {
    local args=(--task ate)
    [ "$#" -gt 0 ] && args=("$@")
    case "$(task_from_args ate "${args[@]}")" in
        ate) ensure_ctibench_rewritten ;;
    esac
    echo "==> Running Scope Gate FPR check (${args[*]})..."
    uv run python -m benchmark.scripts.scope_gate_fpr "${args[@]}"
}

command="${1:-}"
[ "$#" -gt 0 ] && shift

case "$command" in
    sample)   ensure_qdrant; cmd_sample "$@" ;;
    full)     ensure_qdrant; cmd_full "$@" ;;
    ablation) ensure_qdrant; cmd_ablation "$@" ;;
    fpr)      ensure_qdrant; cmd_fpr "$@" ;;
    all)
        ensure_qdrant
        cmd_sample
        cmd_full
        cmd_ablation
        cmd_fpr
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage
        exit 1
        ;;
esac

# Tests and eval

Commands to reproduce the report's figures and tables. Run these from the repository root directory.

```bash
./eval.sh full       # full benchmark eval: AttackQA + CTI-ATE

./eval.sh ablation   # retrieval-only ablation sweep (alpha x rerank x k)

./eval.sh fpr        # Scope Gate false-positive-rate check

./eval.sh --help     # options for each, and how to pass args through
```

# ACE-Atlas: Intelligence on a Budget

ACE-Atlas is a research codebase for a **hybrid recurrent-memory-MoE** language model. It is designed to solve one of the biggest bottlenecks in AI today: the massive compute requirement for training state-of-the-art models.

Our goal is to **beat strong dense baselines in the same parameter class on a low-budget training path.**

## The Problem: The GPU Wall

The current LLM landscape is dominated by dense Transformers that require massive GPU clusters to train and deploy. For startups and researchers without infinite compute, this creates a "GPU Wall" that stifles innovation.

## The Solution: Architectural Efficiency

ACE-Atlas breaks through this wall by using a hybrid architecture that combines the strengths of multiple sequence modeling techniques:

- **Local Attention:** Preserves exact short-range recall without quadratic scaling costs.
- **GRU-fused Recurrent Core:** Provides a fast, linear-scaling path for long-range context.
- **Sparse MoE (Mixture of Experts):** Increases model capacity and knowledge without increasing the compute cost per token.
- **Bounded Memory:** Replaces unbounded KV growth with an explicit, efficient memory substrate.

By choosing the cheapest computation and memory substrate that preserves performance, ACE-Atlas achieves higher quality per dollar than standard dense models.

## Key Results (As of April 2026)

ACE-Atlas has successfully demonstrated that architectural efficiency can win on a budget.

### Beating Dense Baselines

In head-to-head comparisons at matched parameter counts, ACE-Atlas consistently outperforms dense Transformers:

| Size | Dataset | Dense Loss | **ACE-Atlas Loss** | Improvement |
|------|---------|------------|--------------------|-------------|
| 300M | TinyStories | 2.0015 | **1.3264** | ~33% |
| 300M | WikiText-2 | 3.7954 | **3.4819** | ~8% |
| 300M | CodeSearchNet | 6.3760 | **5.8332** | ~8% |

### Cross-Domain Generalization

Unlike specialized models, ACE-Atlas maintains its advantage across diverse domains, including natural language (WikiText-2) and programming (CodeSearchNet Python), proving that the architecture is a robust alternative to dense Transformers.

## Getting Started

### Installation

```bash
python -m pip install -e '.[dev,data]'
```

### Quick Start

Report parameter counts for the 300M hybrid model:
```bash
python scripts/report_model_stats.py --model-name ace_atlas --config configs/hybrid_300m_gru.json
```

Run a training smoke test:
```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_tiny_debug.json \
  --train-config configs/train_tinystories_smoke.json \
  --run-name smoke_test
```

## Documentation

For a deep dive into the project, see:

1. [**Project Dossier**](docs/PROJECT_DOSSIER.md) - Executive summary and proof of concept.
2. [**Architecture Spec**](docs/ARCHITECTURE_SPEC.md) - The technical blueprint of the hybrid core.
3. [**Execution Roadmap**](docs/ROADMAP.md) - Our path from 300M to frontier-class efficiency.

## The Vision

We believe that intelligence shouldn't require a $100M GPU cluster. ACE-Atlas is the first step toward a new class of "frugal" models that deliver high-performance reasoning on commodity hardware.

We are currently focused on converting our modeling advantage into executable-code task wins and scaling our hybrid approach to larger parameter classes.

---
*ACE-Atlas is an open research project. Join us in making AI more accessible.*

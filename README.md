Hierarchical Caching for Agentic Workflows

This repository contains the official implementation of the paper \*\*"Hierarchical Caching for Agentic Workflows: A Multi-Level Architecture to Reduce Tool-Execution Overhead"\*\*, submitted to \*Machine Learning and Knowledge Extraction\* (2026).

This system addresses the high latency and cost of external tool execution in LLM agents. It implements a multi-level caching architecture that captures redundancy at both the workflow and tool levels.

Key Features
The system integrates four key components described in the paper:
1\.  \*\*Hierarchical Caching:\*\* Operates at both workflow (coarse-grained) and tool (fine-grained) levels.
2\.  \*\*Dependency-Aware Invalidation:\*\* Uses graph-based techniques to maintain consistency when write operations affect cached reads.
3\.  \*\*Category-Specific TTL:\*\* Applies tailored Time-to-Live policies for different data types (e.g., weather APIs vs. database queries).
4\.  \*\*Session Isolation:\*\* Ensures multi-tenant cache safety through automatic session scoping.

Repository Structure
\- `src/`: Core implementation of the hierarchical caching architecture.
&nbsp; - `plan\_cache\_proxy.py`: Implements the proxy pattern for caching strategies.
&nbsp; - `dependency\_graph.py`: Logic for dependency tracking and invalidation.
&nbsp; - `adaptive\_ttl.py`: Logic for dynamic TTL adjustment.
&nbsp; - `redis\_cache.py`: Interface for the Redis backend.
\- `experiments/`: Scripts to reproduce the paper's evaluation.
&nbsp; - `run\_complete\_evaluation.py`: Reproduces the main efficiency results (Table 2).
&nbsp; - `workload\_sensitivity.py`: Reproduces the workload sensitivity analysis.
&nbsp; - `run\_multitenant\_test.py`: Reproduces the session isolation experiments.
\- `data/`: Contains the synthetic datasets and raw results.

Getting Started
\Prerequisites

\- Python 3.9+
\- Redis Server

\ Installation
```bash
git clone https://github.com/your-username/hierarchical-caching-agentic.git
cd hierarchical-caching-agentic
pip install -r requirements.txt

\Reproducing Paper Results

The following scripts reproduce the key claims and figures from our paper.
Main Efficiency Results (Table 2): To generate the comparison baseline vs. hierarchical caching (76.5% efficiency, 13.3x speedup):
bash
python -m experiments.run\_complete\_evaluation

Workload Sensitivity (Figure 5): To test performance across varying access distributions (Zipfian parameters):
bash
python -m experiments.workload\_sensitivity

Multi-Tenant Isolation: To verify session isolation with concurrent tenants:
bash
python -m experiments.run\_multitenant\_test

# Energy- and Performance-Aware Task Scheduling in Mobile Cloud Computing

**Research Artifact for arXiv / IEEE Paper**

---

## 1. Overview

This repository provides a **reproducible research artifact** for the paper:

> **Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment**
> IEEE Transactions on Cloud Computing, 2014

The artifact implements the paper’s **two-phase task scheduling framework** for directed acyclic graph (DAG) applications executing across **heterogeneous mobile cores and cloud resources**, with the objective of **minimizing energy consumption under a completion-time constraint**.

The implementation strictly follows the **task model, execution model, energy model, and optimization criteria** defined in the paper, enabling faithful replication of reported results and controlled experimentation.

---

## 2. Problem Setting

### 2.1 Application Model

* Applications are modeled as **directed acyclic graphs (DAGs)**
* Nodes represent tasks with computation requirements
* Edges encode precedence constraints and data dependencies

### 2.2 Execution Environment

* A mobile device with **heterogeneous local CPU cores**
* A remote cloud accessed via a **serialized wireless channel**
* No task preemption; tasks execute to completion once scheduled

### 2.3 Optimization Objective

Minimize total energy consumption while satisfying a hard deadline constraint:

```
T_total ≤ T_max
```

where:

```
T_max = 1.5 × T_initial
```

---

## 3. Algorithmic Framework

The scheduler operates in **two strictly separated phases**, exactly as defined in the paper.

---

### 3.1 Phase 1: Minimal-Delay Scheduling

```
Primary Assignment → Task Prioritization → Execution Unit Selection
```

#### Primary Assignment

* Compares the fastest local execution time against cloud execution time
* Assigns each task to either a local core or the cloud
* Implements **Equations (11–12)**

#### Task Prioritization

* Computes task priorities using DAG topology and computation cost
* Captures downstream criticality via successor aggregation
* Implements **Equations (13–16)**

#### Execution Unit Selection

* Schedules tasks in descending priority order
* Computes ready and finish times for all execution phases
* Produces an initial feasible schedule
* Implements **Equations (3–6)**

---

### 3.2 Phase 2: Energy-Aware Migration

```
Migration Evaluation → Kernel Rescheduling → Energy Assessment → Acceptance
```

#### Migration Evaluation

* Enumerates candidate task migrations between cores and cloud
* Filters migrations violating the deadline constraint

#### Kernel Rescheduling

* Linear-time rescheduling after each migration
* Preserves precedence constraints
* Recomputes all timing variables

#### Migration Selection

* **Stage 1:** Accept migrations that reduce energy with no time increase
* **Stage 2:** Evaluate energy–time trade-offs using:

  ```
  efficiency = Δenergy / Δtime
  ```

---

## 4. Execution and Energy Models

### 4.1 Local Execution (Mobile Device)

* Three heterogeneous CPU cores
* Power consumption increases with performance:

  ```
  P = [1W, 2W, 4W]
  ```
* Core-dependent execution times
* Non-preemptive execution

### 4.2 Cloud Execution (Three-Stage Pipeline)

```
[RF Send] → [Cloud Compute] → [RF Receive]
```

|         Phase | Description     | Time |
| ------------: | --------------- | ---: |
|       RF Send | Wireless upload |    3 |
| Cloud Compute | Cloud execution |    1 |
|    RF Receive | Result download |    1 |

**Constraint:**
Wireless send and receive phases are **globally serialized**.

---

## 5. Experimental Benchmarks and Results

The scheduler is evaluated on **five benchmark task graphs** defined in the original paper.
Each benchmark represents a distinct dependency structure and stress scenario.
All reported results are obtained **exclusively from these benchmarks**.

| Graph | Tasks | Topology   | Purpose                   | Initial Time | Initial Energy | Final Time | Final Energy |
| ----: | ----: | ---------- | ------------------------- | -----------: | -------------: | ---------: | -----------: |
|     1 |    10 | Fan-out    | Parallelism stress        |           18 |          100.5 |         26 |         24.0 |
|     2 |    10 | Balanced   | Mixed dependencies        |           25 |          120.0 |         32 |         15.0 |
|     3 |    20 | Deep chain | Critical-path sensitivity |           34 |          184.0 |         50 |         49.0 |
|     4 |    20 | Wide       | Resource contention       |           33 |          188.0 |         47 |         59.5 |
|     5 |    20 | Irregular  | General case              |           29 |          183.0 |         42 |         73.5 |

**Observation.**
Across all benchmarks, Phase 2 migration achieves substantial energy reduction while respecting the deadline constraint, reproducing the qualitative behavior reported in the paper.

### 6.2 Scheduling States

|              State | Meaning       |
| -----------------: | ------------- |
|      `UNSCHEDULED` | Initial       |
|        `SCHEDULED` | After Phase 1 |
| `KERNEL_SCHEDULED` | After Phase 2 |

### 6.3 Timing Variables

Each task maintains:

* Ready times: `RT_l`, `RT_ws`, `RT_c`, `RT_wr`
* Finish times: `FT_l`, `FT_ws`, `FT_c`, `FT_wr`

---

## 7. Computational Complexity

* **Initial Scheduling:** `O(n² log n)`
* **Kernel Rescheduling:** `O(n)` (linear)
* **Migration Phase:** `O(n × m × k)`
  where `m ≪ n` in practice

---

## 8. Reproducibility

* All benchmarks, parameters, and power models are **fixed**
* Deterministic execution (no randomness)
* Results can be regenerated by running the provided binaries
* Python implementation serves as a readable reference model

### Build and Run

```bash
make
./bin/mcc_scheduler
```

```bash
python3 src/python/mcc.py
```

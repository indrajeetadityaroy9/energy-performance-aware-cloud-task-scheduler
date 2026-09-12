# Energy and Performance-Aware Task Scheduling in Mobile Cloud Computing

Implementation of the scheduling model and two-step algorithm in **Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment**

```bibtex
@INPROCEEDINGS{6973741,
  author={Lin, Xue and Wang, Yanzhi and Xie, Qing and Pedram, Massoud},
  booktitle={2014 IEEE 7th International Conference on Cloud Computing}, 
  title={Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment}, 
  year={2014},
  volume={},
  number={},
  pages={192-199},
  keywords={Mobile handsets;Wireless communication;Energy consumption;Cloud computing;Processor scheduling;Time factors;Scheduling;mobile cloud computing (MCC);energy minimization;hard deadline constraint;task scheduling},
  doi={10.1109/CLOUD.2014.35}}
```


## Algorithms

1. **Initial scheduling:** primary cloud classification, weighted upward ranks, then descending-priority execution-unit selection using the earliest available local/upload time slot, including gaps. Entry tasks follow the same selection rule as other tasks.
2. **Migration:** enumerate local-to-local and local-to-cloud moves. Prefer the largest energy reduction without increasing completion time, otherwise the best energy-saving/time-increase ratio within the explicit deadline. Cloud-to-local moves are excluded as in the paper.
3. **Kernel:** reconstruct destination order using its dependency-derived ready time, then reschedule using a LIFO stack with incremental DAG and resource-sequence updates. Migration trials snapshot only schedule state, not every task's duration table.

The kernel is expected `O(N + E + K)`, linear for sparse graphs with fixed K. The paper's heuristic and underspecified tie-breaking do not justify a global-optimality or exact-table-reproduction claim.

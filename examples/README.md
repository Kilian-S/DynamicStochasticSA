# Examples

Reference implementations that informed the design of the solver, plus a small instance for exercising it.

## Contents

| File | Purpose |
| --- | --- |
| [`global_parameters.py`](./global_parameters.py) | A five-node toy instance with deliberately mis-estimated demand, useful for stepping through the dynamic loop |
| [`sa_example.py`](./sa_example.py) | Textbook simulated annealing on a one-dimensional function, included to make the Metropolis criterion legible in isolation |
| [`static_cvrp_example.py`](./static_cvrp_example.py) | A compact exact CVRP formulation in `docplex`, the starting point for the baseline in [`static/`](../static) |

## The toy instance

`global_parameters.py` defines five nodes whose expected and actual demands diverge sharply in both directions,
which is what makes it a useful smoke test: node 1 is over-estimated by two orders of magnitude, while nodes 2
and 4 are under-estimated badly enough to force tour splits.

```python
from examples.global_parameters import NODES, SYM_DISTANCE_MATRIX, VEHICLE_CAPACITY
```

See the [Quickstart](../README.md#quickstart) in the top-level README for a runnable version.

## Attribution

- `sa_example.py` is adapted from Jason Brownlee,
  [*Simulated Annealing From Scratch in Python*](https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/),
  Machine Learning Mastery.
- `static_cvrp_example.py` is adapted from Hernan Caceres,
  [*CVRP with CPLEX*](https://github.com/industrial-ucn/jupyter-examples/blob/master/optimization/cvrp-cplex.ipynb).

Both are third-party works retained here for reference and are not part of the solver.

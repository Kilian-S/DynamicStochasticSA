# Dynamic Simulated Annealing for the Stochastic and Dynamic CVRP

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/badge/licence-GPL--3.0-green.svg)](./LICENSE.txt)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](./tests)
[![Case study](https://img.shields.io/badge/case%20study-Nurda%C4%9F%C4%B1%2C%20T%C3%BCrkiye-orange.svg)](#the-case-study)

**A vehicle routing solver for disaster relief, built for the case where you do not know how much aid each village needs until the truck arrives.**

Classical routing solvers assume demand is known before the first vehicle leaves the depot. Disaster relief violates that assumption completely. The Dynamic Simulated Annealing (DSA) algorithm in this repository plans a route, drives it, discovers the real demand at each stop, and reoptimises everything still ahead of the convoy.

It was developed for a bachelor thesis at the Technical University of Munich, and evaluated on 48 settlements of the Nurdağı district in Gaziantep Province, Türkiye, using real road distances and post-earthquake population data from February 2023.

---

## The result

Across 350 trials at seven levels of demand volatility, **DSA served 100% of demand every single time.** A perfectly optimal static plan, computed with CPLEX under the same conditions, degrades steadily as uncertainty rises.

<p align="center">
  <img src="docs/images/service-level.png" alt="Service level of DSA versus the exact solver as demand volatility increases" width="620">
</p>

| | Exact solver (CPLEX) | **DSA** |
| --- | --- | --- |
| Service level, low volatility | 98.1% | **100%** |
| Service level, high volatility | 84.1% | **100%** |
| **Worst single trial** | **7.32%** | **100%** |
| Unmet demand at γf = 0.10 | 17,686 loaves | **0** |
| Distance, demand known in advance | **583 km** | 676 km |

That 7.32% is the number worth pausing on. In the worst of 350 draws, the static plan reached fewer than one in thirteen people who needed food that day. DSA has no equivalent failure mode, because it never commits to a plan it cannot revise.

**The honest trade-off:** when demand genuinely is known in advance, DSA drives 16% further than the exact solver. DSA is not a better optimiser. It is a solver that keeps working when the assumption underpinning the better optimiser stops being true.

---

## Table of contents

- [The case study](#the-case-study)
- [What makes the problem hard](#what-makes-the-problem-hard)
- [How the algorithm works](#how-the-algorithm-works)
- [Results in detail](#results-in-detail)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Tests](#tests)
- [Scope and limitations](#scope-and-limitations)
- [Contributing](#contributing)
- [Licence](#licence)
- [Contact](#contact)

---

## The case study

On 6 February 2023 two earthquakes struck southern Türkiye and northern Syria. Nurdağı, a district of roughly 1,200 km² in Gaziantep Province, was among the worst affected: population density fell by over 81% in the days that followed, and damage to buildings was severe even though the major road network held.

<p align="center">
  <img src="docs/images/nurdagi-locality.png" alt="Map of the Nurdağı district showing the depot and 48 demand nodes" width="680">
</p>

<p align="center"><em>The study area. Blue markers are the 48 demand nodes, one per neighbourhood or village. The red marker is the depot, a container district that served as a shelter facility. Black markers bound the ~1,200 km² area.</em></p>

The instance is built from real data, not synthetic benchmarks:

| Parameter | Value | Source |
| --- | --- | --- |
| Demand nodes | 48 settlements + 1 depot | Turkish Statistical Institute, expert insight |
| Population served | 41,322 | Turkish Statistical Institute (2021, 2023) |
| Distances | Road distances, driving mode | Google Maps Distance Matrix API |
| Vehicle | Toyota HiAce H200, capacity 2,000 units | Ubiquitous light commercial van |
| Demand unit | One loaf of *ekmek* per person per day | PAHO minimum ration guidance (~1,700 kcal) |

Distances are road distances rather than travel times, deliberately. Travel times vary by hour, day and season, which would make the instance irreproducible; distances do not. Euclidean distances would be meaningless on a real road network.

## What makes the problem hard

**Stochastic.** Each node carries an *expected* demand (the planning estimate) and an *actual* demand (the truth). Actual demand is drawn from a **Cauchy** distribution rather than a Normal one, because relief needs are heavy-tailed: rare, enormous deviations are exactly the events that break a relief operation, and a Gaussian model prices them at essentially zero.

<p align="center">
  <img src="docs/images/normal-vs-cauchy.png" alt="Histograms comparing the Normal and Cauchy distributions" width="440">
</p>

<p align="center"><em>10,000 samples from each. The Cauchy distribution has a lower peak and far heavier tails; its mean and variance are undefined, so the central limit theorem does not apply.</em></p>

The consequences are not theoretical. Against a baseline network demand of 41,322 loaves, one sampled instance demanded **532,401**, a thirteenfold shock.

<p align="center">
  <img src="docs/images/total-demand.png" alt="Total network demand across trials at each gamma factor" width="620">
</p>

<p align="center"><em>Total demand per trial, extreme outliers above 100,000 excluded for legibility. Dispersion widens sharply as the gamma factor rises.</em></p>

**Dynamic.** Actual demand is revealed only on arrival. A settlement needing more than a vehicle carries requires extra visits that were never planned; one needing less frees capacity. The number of stops therefore changes *while the convoy is already moving*, and a route optimal at *t = 0* is frequently invalid by *t = 1*.

## How the algorithm works

<p align="center">
  <img src="docs/images/dsa-flowchart.png" alt="Activity diagram of the DSA algorithm" width="740">
</p>

Two data structures carry the dynamism:

- **Node family** ([`inputs/node_family.py`](./inputs/node_family.py)). One physical settlement maps to one *family*, which decomposes into `⌈demand / capacity⌉` *child nodes*, each one vehicle-load visit. When actual demand is revealed the family is recomputed, so child nodes appear and disappear as the truth arrives.
- **Dynamic distance matrix** ([`inputs/dynamic_distance_matrix.py`](./inputs/dynamic_distance_matrix.py)). A distance matrix that grows and shrinks in step with the node families, cloning a parent's row and column whenever a new child node is created.

Each outer iteration advances every vehicle one stop, reconciles expected against actual demand, splits any tour whose capacity has been breached (the overflow becomes a *shortfall node* served by a follow-up tour), and reruns simulated annealing on everything not yet traversed. New tours are held at the depot until they reach a configurable **utilisation target**, so vehicles are not dispatched half empty.

### The annealing engine

A candidate solution is generated by extracting one node at random and reinserting it at a random position in a random tour, which may be an empty tour, so the search can both merge and split tours. Improvements are always accepted; worse candidates are accepted with probability `exp(-Δ/T)` (the Metropolis criterion), letting the search escape local optima early and settle as the temperature falls.

The critical constraint is **traversal locking**. Once a vehicle has passed a stop, that prefix of its tour is frozen: inserting a node before it would amount to time travel. Only the untraversed suffix is available to the neighbourhood operator, and the search space therefore shrinks as the convoy advances.

<p align="center">
  <img src="docs/images/cooling-schedules.png" alt="Exponential, linear and sigmoid cooling schedules" width="740">
</p>

Four cooling schedules are implemented (`exponential`, `linear`, `concave`, `sigmoid`) and selected with the `cooling_schedule` argument. Linear decay is the default and produced the reported results.

## Results in detail

Three experiments, 1,730 solver runs in total.

### Experiment 1: what actually matters in the parameters

45 configurations of initial temperature × iteration count × utilisation target, 30 trials each, **1,350 runs**.

<p align="center">
  <img src="docs/images/iterations-vs-distance.png" alt="Total distance against iteration count" width="620">
</p>

| Iterations | Mean distance | Mean runtime |
| ---: | ---: | ---: |
| 10 | 1,398 km | 1.0 s |
| 100 | 867 km | 5.5 s |
| 1,000 | **686 km** | 46.5 s |

Only the iteration count matters. Initial temperature shifts the mean by under 0.6%; the utilisation target by under 0.7%; and no cooling schedule proved clearly superior.

<p align="center">
  <img src="docs/images/cooling-distance-box.png" alt="Box plot of total distance by cooling schedule" width="360">
</p>

That null result is itself informative. Because nodes become locked as vehicles visit them, the search is trapped in local optima early, and no temperature schedule can free it. Extra iterations help only insofar as they widen the search *before* the convoy departs. Execution time scales linearly with iterations (`y = 0.049x + 0.703`), so the design lever is where you spend the iteration budget, not how you cool.

### Experiment 2: the price of not knowing

An exact MIP of the same instance, solved with CPLEX under a 120-second limit, 30 trials.

<p align="center">
  <img src="docs/images/convergence.png" alt="Convergence of DSA and the exact algorithm" width="620">
</p>

| Algorithm | Mean | Min | Max | SD |
| --- | ---: | ---: | ---: | ---: |
| DSA | 676,189 | 623,270 | 736,480 | 25,332 |
| Exact, 1 thread | 602,090 | 591,803 | 603,779 | 3,002 |
| Exact, 8 threads | **583,128** | 580,068 | 585,584 | **1,648** |

*All values in metres.*

When demand is known, CPLEX wins outright. DSA's tours are 15.96% longer, and its standard deviation is 15 times larger. DSA's spread betrays its tendency to settle in local optima. Every CPLEX run hit the time limit, so this is a best incumbent rather than a proven optimum, and it is a deliberately demanding benchmark.

**This is the baseline DSA has to beat, and on this ground it does not.** The interesting question is what happens when the ground shifts.

### Experiment 3: what happens when the assumption breaks

7 volatility levels × 50 trials, **350 runs**. Actual demand is drawn from a Cauchy distribution with scale `γ = γf × expected demand`, so larger settlements carry proportionally larger uncertainty. Both solvers are scored against the same revealed demand.

<p align="center">
  <img src="docs/images/undersupply.png" alt="Total undersupply against gamma factor" width="49%">
  <img src="docs/images/oversupply.png" alt="Total oversupply against gamma factor" width="49%">
</p>

**Undersupply (left) is the headline.** DSA's line is flat on zero, at every volatility level, in all 350 trials. The static plan's unmet demand climbs at roughly 81,588 units per unit of γf, peaking at a mean of 17,686 loaves undelivered. It cannot react, so when a village needs triple what was forecast, the shortfall simply happens.

**Oversupply (right) is the subtler win.** Both approaches waste capacity as volatility rises, but DSA's wasted capacity grows at **half the rate** (28,641 per unit γf against 56,819). DSA's oversupply comes from dispatching more tours; the static plan's comes from sending vehicles to villages whose needs it misjudged.

<p align="center">
  <img src="docs/images/service-level-exact-box.png" alt="Box plot of the exact solver's service level by gamma factor" width="560">
</p>

<p align="center"><em>The static plan does not just get worse on average, it gets less reliable. Spread widens with volatility, and the outliers are demand shocks. The worst reached a 7.32% service level.</em></p>

<p align="center">
  <img src="docs/images/execution-time.png" alt="DSA execution time against gamma factor" width="560">
</p>

DSA is not free: execution time rises with volatility, since more tours and child nodes must be managed. Demand and runtime correlate at a Pearson coefficient of 0.92. For a solver that runs in under two minutes on a 2018 laptop while planning a day's convoy, that is a cost worth paying.

### What it adds up to

| Condition | Use |
| --- | --- |
| Demand known in advance | **Exact solver.** DSA drives 16% further and varies far more. |
| Demand uncertain, service level matters | **DSA.** 100% of demand served at every volatility level tested. |

Predictive methods built on expected demand fail to reach adequate service levels **even when demand is only mildly uncertain**, and the failure compounds as volatility rises. In a setting where the objective is feeding people rather than minimising kilometres, an algorithm that never leaves demand unmet is worth 16% more driving.

## Installation

```bash
git clone https://github.com/Kilian-S/DynamicStochasticSA.git
cd DynamicStochasticSA

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Requires **Python 3.10 or newer**. The core solver needs only `numpy`, `pandas` and `openpyxl`.

Reproducing the exact baseline additionally requires IBM CPLEX via `docplex`. Rebuilding the distance matrix from coordinates requires a Google Maps Distance Matrix API key in the `GOOGLE_MAPS_API_KEY` environment variable. Neither is needed to run the solver: the bundled matrix is committed.

## Quickstart

```python
import numpy as np

from dynamic_behaviour import dynamic_sa
from inputs.node import InputNode
from simulated_annealing import objective

# id, expected demand (the plan), actual demand (the truth, revealed on arrival)
nodes = [
    InputNode('0', 0, 0),        # depot
    InputNode('1', 1000, 10),    # heavily over-estimated
    InputNode('2', 400, 3300),   # heavily under-estimated
    InputNode('3', 700, 1000),
    InputNode('4', 200, 3000),
]

distance_matrix = np.array([
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 44],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0, 4],
    [12, 44, 10, 4, 0],
])

distance, tours, execution_time, final_nodes = dynamic_sa(
    nodes, distance_matrix, objective,
    initial_temperature=100, iterations=1000,
    vehicle_capacity=600, utilisation_target=0.9,
)
```

The full Nurdağı instance:

```python
from inputs.instance import DISTANCES_FILE, NODES_SHEET, read_instance_distance_matrix
from inputs.node import create_nodes_static, create_nodes_cauchy_dependent_on_expected_demand

nodes = create_nodes_static(DISTANCES_FILE, NODES_SHEET)                                  # deterministic
nodes = create_nodes_cauchy_dependent_on_expected_demand(DISTANCES_FILE, NODES_SHEET, 0.10)  # stochastic
distance_matrix = read_instance_distance_matrix()
```

Or from the command line:

```bash
python -m examples.main                                                              # toy instance
python -m experiments.experiment_1_parameter_sensitivity_analysis.parameter_sensitivity_analysis
python -m experiments.experiment_3_stochastics.stochastics
```

## Repository structure

```
.
├── simulated_annealing.py   # SA engine, objective function, feasibility checks
├── dynamic_behaviour.py     # dynamic execution loop and demand reconciliation
├── inputs/                  # nodes, node families, dynamic distance matrix, data loading
├── static/                  # exact CVRP baseline (CPLEX / docplex)
├── experiments/             # the three experiments, their result sets and figures
├── examples/                # runnable demo and the references that informed the design
├── errors/                  # domain-specific exceptions
├── docs/images/             # figures used in this README
└── tests/                   # 46 unit and integration tests
```

`inputs/instance.py` resolves the bundled data set relative to the repository, so every script and test finds it regardless of the directory it is run from. Raw result sets are committed as `.xlsx` files alongside each experiment.

## Tests

```bash
python -m pytest
```

46 tests in under 10 seconds, covering the annealing engine, the feasibility constraints, node family and distance matrix bookkeeping, the demand reconciliation paths, and end-to-end runs on both the toy instance and the 48-node case study. The integration tests assert the invariants the approach depends on: every revealed visit served exactly once, every tour starting and ending at the depot, and no vehicle loaded beyond capacity.

## Scope and limitations

Stated plainly, because a result is only worth as much as its boundaries:

- **One case study.** A single 48-node instance from one district. The findings are indicative, not general.
- **The baseline is time-limited, not optimal.** Experiment 2 reports the best CPLEX incumbent after 120 seconds.
- **Heavy-tailed sampling is noisy.** With undefined moments, 50 trials per volatility level leaves wide confidence intervals. Results at γf ≥ 0.125 are not monotone for exactly this reason, and the thesis notes that assessing rare high-impact events properly would need orders of magnitude more repetitions.
- **Vehicles are assumed available on demand.** Fleet size is uncapped, so a new tour can always be dispatched. Real relief operations are fleet-constrained.
- **Travel cost is distance, not time.** No road damage, congestion or time windows, all of which matter after an earthquake.
- **Service level is measured against a fixed node basis.** When the solver splits a node mid-tour, demand moves onto a new child node while the metric still scores against the omniscient view. The two never diverged on this instance, but the metric would understate service level where they did.
- **The committed results predate several bug fixes.** The published `.xlsx` files were produced by the thesis-era implementation. Defects have since been corrected, most consequentially in how the search tracks its best solution, so a rerun will not reproduce these figures exactly. The qualitative findings are unaffected.

## Contributing

Contributions are welcome. If you have found a bug, have an idea, or want to suggest an enhancement, please open an issue.

To contribute code, fork the repository, work on a branch, and open a pull request. Please match the conventions already in the codebase and include tests for new behaviour. If you are planning research based on this project, or want to discuss a larger change, please get in touch first.

## Licence

Licensed under the **GNU General Public License v3.0**, see [LICENSE.txt](./LICENSE.txt).

You may use, modify and distribute this work, including commercially, provided derivative works carry the same licence so recipients keep the same freedoms. Provided without warranty. Full text: <https://www.gnu.org/licenses/gpl-3.0.html>.

## Contact

**Kilian Xhen Schwarz**, [sdcvrp@gmail.com](mailto:sdcvrp@gmail.com)

Developed as a bachelor thesis at the **Technical University of Munich**, Chair of Operations Management, supervised by Prof. Dr. Rainer Kolisch and M.Sc. Baturhan Bayraktar. Submitted 3 August 2023.

*Dynamic and Stochastic Vehicle Routing for Post-Disaster Relief Aid Distribution: A Case Study on the 2023 Turkish-Syrian Earthquake.*

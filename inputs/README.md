# Inputs

Data structures and loaders that feed the SDCVRP solver. Everything here is consumed by
[`dynamic_behaviour.py`](../dynamic_behaviour.py) and [`simulated_annealing.py`](../simulated_annealing.py).

## Contents

| Module | Purpose |
| --- | --- |
| [`node.py`](./node.py) | `Node` and `InputNode` types, plus loaders that build node lists from the bundled spreadsheet with deterministic or Cauchy-distributed actual demand |
| [`node_family.py`](./node_family.py) | `NodeFamily` — maps one physical settlement to the set of vehicle-load visits it requires, and recomputes that set when actual demand is revealed |
| [`dynamic_nodes_list.py`](./dynamic_nodes_list.py) | `DynamicNodeList` — the registry of node families; flattens them into the node list the solver operates on |
| [`dynamic_distance_matrix.py`](./dynamic_distance_matrix.py) | `DynamicDistanceMatrix` — a distance matrix that grows and shrinks as node families gain and lose child nodes |
| [`distances.py`](./distances.py) | Geographic helpers: reads coordinates, queries the Google Maps Distance Matrix API, and writes a symmetric road-distance matrix to Excel |
| `distances.xlsx` | The bundled Nurdağı instance: 48 settlements with coordinates, expected demand, and the 49 × 49 road-distance matrix |

## Node identifiers

Identifiers encode the family-to-child relationship:

- `'0'` — the depot, which is always a single node and never decomposes.
- `'7'` — node family 7, one physical settlement.
- `'7.1'`, `'7.2'`, … — child nodes of family 7, one per vehicle-load visit.

A family with demand `d` and vehicle capacity `c` decomposes into `⌈d / c⌉` child nodes: `⌊d / c⌋` at full capacity
plus one carrying the remainder. Because actual demand is only revealed on arrival, this count changes during
execution, which is what both the `DynamicNodeList` and the `DynamicDistanceMatrix` exist to absorb.

## The bundled instance

`distances.xlsx` holds two sheets used by the solver:

- `Sheet1` — settlement names, coordinates and expected demand (column C).
- `Distance matrix (districts)` — the symmetric road-distance matrix, range `B2:AX50`, in metres.

Distances were retrieved from the Google Maps Distance Matrix API in driving mode and made symmetric by mirroring
the upper triangle. Regenerating them requires an API key; the committed matrix means no API access is needed to
reproduce any of the reported results.

Refer to the module docstrings for per-function detail.

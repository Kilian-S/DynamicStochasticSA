# Readme

Welcome to the Dynamic Simulated Annealing (DSA) solution method solving the Stochastic and Dynamic Capacitated Vehicle Routing Problem (SDCVRP). Please find the problem 
definition and a description of the solution method below.

## Table of Contents

- [Introduction](#introduction)
- [Simulated Annealing](#Simulated Annealing)
- [Dynamism](#Dynamism)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Simulated Annealing (SA) is a metaheuristic optimisation technique using stochastic changes to problem instances to find an optimal or near-optimal solution. SA has been used 
effectively to solve the CVRP, however, only little research exists on how SA fares at solving the SDCVRP. Here, a case study using data from the 2023 Turkey-Syria earthquake 
is used to plug the gap in research. Specifically, data from the ravaged Nurdağı province is used. 

## Simulated Annealing

The simulated annealing engine of the DSA algorithm is implemented in [simulated_annealing.py](./simulated_annealing.py). Here, an input tour varied stochastically and can 
improve or worsen over iterations. Accordingly, tours can be accepted in two ways: Firstly, tours which are shorter in distance (more optimal) are always accepted. By contrast, 
tours which are less optimal may still be selected accoridng to the Metropolis acceptance criterion that depends on the current iteration, and the distance to the best-found 
solution.

## Dynamism

The dynamic part of the DSA is implemented in [dynamic_behaviour.py](./dynamic_behaviour.py). Here, after a SA reoptimisation has taken place, tours are traversed (they move on 
to their next stop). However, this is not always possible. Supply shocks might require a vehicle to resupply at the depot, or fit in another stop before resupplying. These 
changes are also reconciled in this method. To see how the DSA works on a simple example, please see the method test_simple() 
[test_dynamic_behaviour.py](./tests/test_dynamic_behaviour.py)

## Contributing

I warmly welcome any contributions! Your insights can make a significant impact and help improve this project. If you have ideas for new features, suggestions for enhancements or have found a bug, I would be glad to hear from you.

For those interested in contributing directly with code, I invite you to create a fork of the repository, make your changes, and then submit a pull request. I can then review and potentially merge your changes into the main code base. Please ensure your code aligns with the same standards and conventions used in the current code.

If you're planning to contribute to research based on this project or wish to discuss larger changes, please reach out to me first. This allows me to guide your efforts, prevent duplication, and help you understand the design and implementation decisions that may impact your work.

Feel free to communicate regarding this project at [sdcvrp@gmail.com](mailto:sdcvrp@gmail.com). I greatly appreciate your interest and am excited to hear from you!

-Kilian Xhen Schwarz

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE.txt) file for details.

This means that you're free to do almost anything with this project, like distributing, modifying, or selling it, under the condition that when you distribute the project, the 
same license is applied, so that any recipients also have these freedoms. Please note that this project is distributed WITHOUT ANY WARRANTY.

For more information on the GNU General Public License v3.0, please visit https://www.gnu.org/licenses/gpl-3.0.html.


# Experiments of the paper

Code of the numerical experiments of `paper.tex` (Section "Numerical experiments"), in Julia, with
JuMP and Gurobi. The environment is the `Project.toml` at the root of the repository, and every
command below is run from the root of the repository.

## Where to find what

| Paper | Code |
|---|---|
| Instances of the full model (Table `tab:component-parameters`) and of the simplified model (Table `tab:mtbf-simplified`) | `src/instances.jl` |
| MDP: costs of a period, separable approximation c̃_m, transitions, quota dynamics | `src/model.jl` |
| Enumeration of the joint actions | `src/joint_actions.jl` |
| 2SB, two-step baseline | `src/two_step_baseline.jl` |
| CFLP, Compact fluid LP (eq:compact-fluid) | `src/cflp.jl` |
| JFLP (enum.), Joint fluid LP with every joint action, and JFLP-Q (eq:fluid-quota-subsystem) | `src/jflp_enum.jl` |
| JFLP (CG), column generation (eq:fluid, eq:pricing-first, eq:pricing), JFLP+cuts (lift-and-project) and JFLP-2 (order-2 fluid LP) | `src/jflp_cg.jl` |
| Fluid policies (Section "Fluid policies") | `make_fluid_policy` in `src/model.jl`, and the `make_policy_*` function of every method |
| Best anticipative policy (lower bound of every scenario) | `src/anticipative.jl` |
| Monte Carlo simulation (paragraph "Policy evaluation") | `src/simulation.jl` |

`src/setup.jl` loads all the code, and `make_policy(method, fd)` returns the policy of a method,
named as in the tables: `2SB`, `CFLP`, `JFLP-enum`, `JFLP-CG`, `JFLP-cuts`, `JFLP-2`, `JFLP-Q`.

In the code, a substation is called a *pole*, and the maintenance decision of a component is its
*mode* (1 = not maintained, 2 = maintained).

## Instances

| Name | Model | Substations |
|---|---|---|
| `one_pole`, `two_poles`, `three_poles` | full model, 9 components per substation, quota K0 = 30 | 1, 2, 3 |
| `one_pole_simple`, `two_poles_simple`, `three_poles_simple` | simplified model, 7 components per substation, no quota | 1, 2, 3 |

The quota, the numbers of declared days and the start days are discretised in steps of 15 days
(`STEP`), and the start days are in {16, 31, 46} (lead time L = 15). The fluid horizon is 6 periods.

## Reproducing the results

1. **γ of 2SB.** For every instance, choose γ by simulation:

       julia --project=. paper_experiment/scripts/tune_gamma.jl one_pole_simple 200

2. **Simulations.** For every method and instance, simulate the 1000 scenarios, possibly in several
   ranges run in parallel in the same folder:

       GUROBI_THREADS=2 julia --project=. paper_experiment/scripts/simulate.jl JFLP-CG one_pole_simple 1 1000
       GUROBI_THREADS=2 julia --project=. paper_experiment/scripts/simulate.jl 2SB one_pole_simple 1 1000 --gamma=<γ of step 1>

   The results are written in `paper_experiment/results`: one row per simulated period
   (`*_periods.csv`), one row per new decision with its computing time (`*_solves.csv`), and the
   decisions computed (`*_policy.csv`). A simulation can be stopped and resumed. The computing times
   are only comparable between methods with the same number of Gurobi threads (`GUROBI_THREADS`).

3. **Anticipative costs.** For every instance:

       GUROBI_THREADS=2 julia --project=. paper_experiment/scripts/anticipative.jl one_pole_simple 1 1000

4. **Tables and figures.** For every instance, the rows of the tables (mean, standard deviation and
   CVaR_0.95 of the cost, grouped maintenances, computing time and its split) and the quantiles of the
   ratio to the anticipative cost (box plots), over the scenarios simulated by every method:

       julia --project=. paper_experiment/scripts/analysis.jl one_pole_simple

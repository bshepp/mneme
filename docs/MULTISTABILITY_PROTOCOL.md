# Multistability Experiment Protocol

**Status:** protocol written, runs not started.

## Question

Does a simulated tissue have more than one stable voltage pattern under the same parameters?

This is the prediction that separates "tissue stores pattern memory as attractors" from "tissue relaxes to a single resting state".

## Why the existing runs do not answer it

| Problem | Detail |
|---|---|
| Different tissues | `sim_1` has 153 cells and `sim_2` has 156. They are different worlds, so different end states are expected regardless. |
| Not at steady state | At the final frame both runs were still moving at 10% to 17% of their starting speed. |
| One run per condition | Nothing was replicated, so no variation can be attributed to a cause. |

## Design

### Fixed across all runs

- One cell cluster, generated once from one seed and reused.
- One parameter set, apart from the variable under test.
- One export schedule.

### Step 1: steady state

Run the tissue until the largest per-cell voltage change between exported frames stays below a tolerance for a sustained period.

| Setting | Value |
|---|---|
| Tolerance | 0.01 mV per frame |
| Sustained for | 20 consecutive frames |

A run that does not meet this is extended or excluded. It is not analysed as if it had settled.

### Step 2: initial conditions

Start the same tissue from at least 20 initial voltage patterns drawn from a stated distribution. Record the end state of each.

### Step 3: count the end states

Two end states are the same if the largest per-cell difference between them is below a threshold set from replicate noise.

| Quantity | How it is set |
|---|---|
| Replicate noise | Largest per-cell difference between end states of runs started from the same initial condition |
| Same-state threshold | Five times the replicate noise |

The number of distinct end states is the result. One distinct state means no multistability was found.

### Step 4: perturbation

Take a settled state, displace the voltages by a stated amount, and continue the run. Record whether it returns to the same state, and how long it takes. Repeat at increasing displacements to find the size at which it stops returning.

### Step 5: parameter sweep

Repeat steps 2 and 3 at several values of gap-junction conductance. Report the number of distinct end states at each value.

## Analysis

Analysis uses voltages at the cells, loaded with `load_betse_cells()`. No interpolation.

| Measure | Purpose |
|---|---|
| Convergence curve | Confirms steady state |
| Number of distinct end states | The primary result |
| Return time after perturbation | Stability of each state |
| Persistence diagram of each end state | Whether topology separates the states |

## Baselines

| Claim | Baseline it must beat |
|---|---|
| "There are k distinct end states" | Replicate noise from identical initial conditions |
| "Topology separates the states" | Separation by mean voltage alone |
| "The state count changes with conductance" | Variation in the count across repeated draws of initial conditions at fixed conductance |

## What would count as a negative result

- One distinct end state at every conductance.
- End states that differ by no more than replicate noise.

A negative result is reported as such.

## Requirements

| Requirement | Notes |
|---|---|
| BETSE | Not installed in the development environment. The earlier runs used BETSE 1.5.1 on Linux. |
| Simulation configs | Not in the repository. |
| Compute | The earlier attractor runs took about 2.5 hours each. 20 initial conditions at 5 conductances is 100 runs. |

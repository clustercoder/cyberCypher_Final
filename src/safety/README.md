# src/safety/

This folder contains formal safety verification logic.

## Files

- `z3_verifier.py`: safety constraints and proof generation using Z3

## What Is Verified

Before autonomous action execution, constraints are checked such as:
- max post-action link utilization
- no cascading overload on previously healthy links
- customer impact isolation limits
- rate-limit cap for automated actions
- blast radius cap
- rollback path existence

## Control Flow

1. Decider builds candidate action.
2. Causal + digital twin estimate post-action state.
3. `Z3SafetyVerifier.verify_action()` checks all registered constraints.
4. If unsafe, action is blocked and violations are reported.

## Data Flow

Input:
- proposed action
- current network state
- predicted post-action state

Output:
- verdict (`is_safe`)
- list of violated constraints
- human-readable proof summary

## Why Formal Verification

Heuristics and ML confidence are probabilistic.
Safety constraints encode non-negotiable invariants.
Z3 provides explicit satisfiable/unsatisfiable reasoning and makes behavior auditable.

## Extension Pattern

You can add custom constraints via `add_constraint(name, fn)`.
Keep constraints deterministic and explainable.

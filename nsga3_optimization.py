# ============================================================
# NSGA-III multi-objective optimization for preventive evacuation
# Updated version: 3 constraints (Rb, Ra, scenario validity)
# ============================================================
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# ==========================================================
# 1. Decision variables across nine factor dimensions
# ==========================================================
decision_names = (
    ["x1"]
    + [f"X2-{i}" for i in range(1, 14)]
    + [f"X3-{i}" for i in range(1, 4)]
    + [f"X4-{i}" for i in range(1, 4)]
    + [f"X5-{i}" for i in range(1, 4)]
    + [f"X6-{i}" for i in range(1, 3)]
    + [f"X7-{i}" for i in range(1, 14)]
    + [f"X8-{i}" for i in range(1, 14)]
    + [f"X9-{i}" for i in range(1, 9)]
)
n_var = len(decision_names)
assert n_var == 59, f"Expected 59 variables, got {n_var}"

# IMPORTANT:
# Replace these placeholders with the actual lower and upper
# bounds of all 59 variables, in decision_names order.
xl = np.array([
    # Enter 59 lower-bound values here
], dtype=float)
xu = np.array([
    # Enter 59 upper-bound values here
], dtype=float)

# Check that the bounds have been specified completely.
if len(xl) != n_var or len(xu) != n_var:
    raise ValueError(
        "xl and xu must each contain exactly 59 values."
    )
if np.any(xl > xu):
    raise ValueError("Every lower bound must be <= its upper bound.")

# ==========================================================
# 2. Baseline risk constraints
# ==========================================================
# These values must correspond to the initial building state.
Rb_baseline = 11.94
Ra_baseline = 6.82

# ==========================================================
# 3. Decode decision variables
# ==========================================================
def decode_individual(x):
    """Decode a candidate solution into named decision variables."""
    return {
        name: int(round(x[i]))
        for i, name in enumerate(decision_names)
    }

# ==========================================================
# 4. Multi-objective optimization problem
# ==========================================================
class EvacuationOptimization(Problem):
    def __init__(self, scenario):
        self.scenario = scenario
        super().__init__(
            n_var=n_var,
            n_obj=2,       # Rp and fire protection cost
            n_ieq_constr=3,  # Rb, Ra, and scenario validity
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        objectives = []
        constraints = []
        for individual_x in x:
            individual = decode_individual(individual_x)

            # --------------------------------------------------
            # Constraint 3: fire-scenario validity
            # Convention: G <= 0 means feasible.
            # --------------------------------------------------
            scenario_valid = check_scenario(
                individual, self.scenario
            )
            if not scenario_valid:
                # Penalize invalid scenarios.
                Rp = 999.0
                Rb = 999.0
                Ra = 999.0
                Cost = 999.0
                # Explicit scenario infeasibility constraint.
                G_scenario = 1.0
            else:
                G_scenario = -1.0
                # Construct model inputs in their training order.
                input_Rp = build_model_input(
                    individual, features["Rp"]
                )
                input_Rb = build_model_input(
                    individual, features["Rb"]
                )
                input_Ra = build_model_input(
                    individual, features["Ra"]
                )
                input_Cost = build_model_input(
                    individual, features["Cost"]
                )
                # Predict risk indicators and protection cost.
                Rp = float(
                    models["Rp"].predict(input_Rp)[0]
                )
                Rb = float(
                    models["Rb"].predict(input_Rb)[0]
                )
                Ra = float(
                    models["Ra"].predict(input_Ra)[0]
                )
                Cost = float(
                    models["Cost"].predict(input_Cost)[0]
                )

            # --------------------------------------------------
            # Two primary objectives:
            # Minimize occupant risk and protection cost.
            # --------------------------------------------------
            objectives.append([Rp, Cost])

            # --------------------------------------------------
            # Two secondary safety constraints:
            # Rb <= Rb_baseline
            # Ra <= Ra_baseline
            # --------------------------------------------------
            constraints.append([
                Rb - Rb_baseline,
                Ra - Ra_baseline,
                G_scenario
            ])

        out["F"] = np.asarray(objectives, dtype=float)
        out["G"] = np.asarray(constraints, dtype=float)

# ==========================================================
# 5. NSGA-III optimization
# ==========================================================
def run_nsga3(scenario):
    problem = EvacuationOptimization(scenario)

    # Das-Dennis reference directions for two objectives.
    ref_dirs = get_reference_directions(
        "das-dennis",
        2,
        n_partitions=164
    )

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=180,
        sampling=ScenarioSampling(scenario),
        crossover=SBX(),
        mutation=PM()
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", 300),
        seed=1,
        verbose=True
    )

    return result

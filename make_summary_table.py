import numpy as np
import pandas as pd

from optimizer import Method, Option, optSolver_OptimizationHunter_with_stats
from testing.objectives import Quadratic, Quadratic2, Rosenbrock, DataFit, Exp, Genhumps


def make_exp_problem(name, base_x0, target_n=None):
    base_x0 = np.asarray(base_x0)
    dim0 = base_x0.size

    if target_n is None:
        x0 = base_x0.copy()
        n = dim0
    else:
        reps = int(np.ceil(target_n / dim0))
        x_full = np.tile(base_x0, reps)
        x0 = x_full[:target_n]
        n = target_n

    return Exp(name, x0=x0, n=n)


def build_problems():
    np.random.seed(0)

    problems = []

    # ---------- Problem 1–4: Quadratic ----------
    problems.append(
        Quadratic("P1_quad_10_10", x0=20 * np.random.rand(10) - 10, kappa=10, n=10)
    )
    problems.append(
        Quadratic("P2_quad_10_1000", x0=20 * np.random.rand(10) - 10, kappa=1000, n=10)
    )
    problems.append(
        Quadratic("P3_quad_1000_10", x0=20 * np.random.rand(1000) - 10, kappa=10, n=1000)
    )
    problems.append(
        Quadratic(
            "P4_quad_1000_1000",
            x0=20 * np.random.rand(1000) - 10,
            kappa=1000,
            n=1000,
        )
    )

    # ---------- Problem 5–6: Quartic (Quadratic2) ----------
    Q1 = np.array(
        [
            [5, 1, 0, 0.5],
            [1, 4, 0.5, 0],
            [0, 0.5, 3, 0],
            [0.5, 0, 0, 2],
        ],
        dtype=float,
    )

    x0_q = np.array(
        [np.cos(70), np.sin(70), np.cos(70), np.sin(70)],
        dtype=float,
    )

    problems.append(
        Quadratic2("P5_quartic_1", x0=x0_q, Q=Q1, sigma=1e-4, n=4)
    )
    problems.append(
        Quadratic2("P6_quartic_2", x0=x0_q, Q=Q1, sigma=1e4, n=4)
    )

    # ---------- Problem 7–8: Rosenbrock ----------
    problems.append(
        Rosenbrock("P7_Rosenbrock_2", x0=np.array([-1.2, 1.0]), n=2)
    )

    x0_rosen_100 = np.ones(100)
    x0_rosen_100[0] = -1.2
    problems.append(
        Rosenbrock("P8_Rosenbrock_100", x0=x0_rosen_100, n=100)
    )

    # ---------- Problem 9: DataFit ----------
    problems.append(
        DataFit("P9_DataFit_2", x0=np.array([1.0, 1.0]), n=2)
    )

    # ---------- Problem 10–11: Exponential ----------
    base_x0_exp10 = np.zeros(10)
    base_x0_exp10[0] = 1.0
    problems.append(
        make_exp_problem("P10_Exponential_10", base_x0_exp10, target_n=10)
    )

    # Problem 11: n = 100
    base_x0_exp100 = base_x0_exp10  
    problems.append(
        make_exp_problem("P11_Exponential_100", base_x0_exp100, target_n=100)
    )

    # ---------- Problem 12: Genhumps ----------
    x0_gen = np.array([506.2] * 5)
    x0_gen[0] = -x0_gen[0]
    problems.append(Genhumps("P12_Genhumps_5", x0=x0_gen))

    return problems


def build_methods():
    methods = []

    method_name_list = [
        "GD",
        "ModifiedNewton",
        "NewtonCG",
        "BFGS",
        "DFP",
        "LBFGS",
    ]

    for name in method_name_list:
        methods.append(Method(name=name, line_search="armijo"))
        methods.append(Method(name=name, line_search="wolf"))

    return methods


def main():
    print(">>> make_summary_table.py main() started")

    problems = build_problems()
    methods = build_methods()

    rows = []

    for prob in problems:
        for meth in methods:
            option = Option()
            output_path = (
                f"./output_summary/{prob.name}/{meth.method_name}_{meth.line_search}.txt"
            )
            option.set_option_params(output=output_path)

            print(f"Running {prob.name} with {meth.method_name}_{meth.line_search} ...")

            try:
                res = optSolver_OptimizationHunter_with_stats(prob, meth, option)

                rows.append(
                    {
                        "problem": prob.name,
                        "method": f"{meth.method_name}_{meth.line_search}",
                        "iter": res.get("iter"),
                        "nfev": res.get("nfev"),
                        "ngev": res.get("ngev"),
                        "cpu": res.get("cpu"),
                        "f_final": res.get("f"),
                        "grad_norm_final": res.get("grad_norm"),
                    }
                )

            except Exception as e:
                print(f"{prob.name} with {meth.method_name}_{meth.line_search} fails: {e}")

                rows.append(
                    {
                        "problem": prob.name,
                        "method": f"{meth.method_name}_{meth.line_search}",
                        "iter": None,
                        "nfev": None,
                        "ngev": None,
                        "cpu": None,
                        "f_final": None,
                        "grad_norm_final": None,
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv("summary_table.csv", index=False)

    print("\nSummary table written to summary_table.csv")
    print(df)


if __name__ == "__main__":
    main()



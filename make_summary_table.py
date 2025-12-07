import numpy as np
import pandas as pd

from optimizer import Method, Option, optSolver_OptimizationHunter_with_stats
from testing.objectives import Quadratic, Quadratic2, Rosenbrock, DataFit, Exp, Genhumps


def make_exp_problem(name, base_x0, target_n=None):
    base_x0 = np.asarray(base_x0)
    dim0 = base_x0.size

    if target_n is None:
        x0 = base_x0.copy()
    else:
        reps = int(np.ceil(target_n / dim0))
        x_full = np.tile(base_x0, reps)
        x0 = x_full[:target_n]

    n = x0.size
    return Exp(name, x0=x0, n=n)


def build_problems():
    np.random.seed(0)

    problems = []

    # ---------- problem 1-4: Quadratic ----------
    x0_10_a = 20 * np.random.rand(10) - 10
    problems.append(
        Quadratic("P1_quad_10_10", x0=x0_10_a, kappa=10, n=len(x0_10_a))
    )

    x0_10_b = 20 * np.random.rand(10) - 10
    problems.append(
        Quadratic("P2_quad_10_1000", x0=x0_10_b, kappa=1000, n=len(x0_10_b))
    )

    x0_1000_a = 20 * np.random.rand(1000) - 10
    problems.append(
        Quadratic("P3_quad_1000_10", x0=x0_1000_a, kappa=10, n=len(x0_1000_a))
    )

    x0_1000_b = 20 * np.random.rand(1000) - 10
    problems.append(
        Quadratic("P4_quad_1000_1000", x0=x0_1000_b, kappa=1000, n=len(x0_1000_b))
    )

    # ---------- problem 5-6: Quartic (Quadratic2) ----------
    Q1 = np.array([[2, 1, 0, 0],
                   [1, 2, 1, 0],
                   [0, 1, 2, 1],
                   [0, 0, 1, 2]])
    x0_q = np.array([3, 1, 3, 1])

    problems.append(Quadratic2("P5_quartic_1", x0=x0_q, Q=Q1, sigma=1e-4, n=4))
    problems.append(Quadratic2("P6_quartic_2", x0=x0_q, Q=Q1, sigma=1e4,  n=4))

    # ---------- problem 7-8: Rosenbrock ----------
    problems.append(Rosenbrock("P7_Rosenbrock_2",
                               x0=np.array([-1.2, 1.0]),
                               n=2))

    x0_rosen_100 = -np.ones(100)
    x0_rosen_100[::2] = -1.2
    problems.append(Rosenbrock("P8_Rosenbrock_100",
                               x0=x0_rosen_100,
                               n=100))

    # ---------- problem 9: DataFit ----------
    problems.append(DataFit("P9_DataFit_2",
                            x0=np.array([1.0, 1.0]),
                            n=2))

    # ---------- problem 10-11: Exponential ----------
    base_x0 = np.array([5, 5, 5, 5, 5, 1, 1, 1, 1, 1]) 

    problems.append(
        make_exp_problem("P10_Exponential_10", base_x0, target_n=10)
    )

    problems.append(
        make_exp_problem("P11_Exponential_100", base_x0, target_n=100)
    )

    # ---------- problem 12: Genhumps ----------
    x0_gen = np.array([-506.2, -506.2, -506.2, -506.2, -506.2])
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
            opt = Option()
            output_path = f"./output_summary/{prob.name}/{meth.method_name}_{meth.line_search}.txt"
            opt.set_option_params(output=output_path)

            print(f"Running {prob.name} with {meth.method_name}_{meth.line_search} ...")

            try:
                res = optSolver_OptimizationHunter_with_stats(prob, meth, opt)
                rows.append({
                    "problem": prob.name,
                    "method": f"{meth.method_name}_{meth.line_search}",
                    "iter": res["iter"],
                    "nfev": res["nfev"],
                    "ngev": res["ngev"],
                    "cpu":  res["cpu"],
                    "f_final": res["f"],
                    "grad_norm_final": res["grad_norm"],
                })
            except Exception as e:
                print(f"  FAILED on {prob.name} / {meth.method_name}_{meth.line_search}: {e}")
                rows.append({
                    "problem": prob.name,
                    "method": f"{meth.method_name}_{meth.line_search}",
                    "iter": None,
                    "nfev": None,
                    "ngev": None,
                    "cpu":  None,
                    "f_final": None,
                    "grad_norm_final": None,
                })

            df = pd.DataFrame(rows)
            df.to_csv("summary_table.csv", index=False)

    print("\nSummary table written to summary_table.csv")
    print(df)


if __name__ == "__main__":
    main()


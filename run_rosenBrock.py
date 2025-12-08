import numpy as np
import pandas as pd

from optimizer import Method, Option, optSolver_OptimizationHunter_with_stats
from testing.objectives import Rosenbrock



def build_problems():
    np.random.seed(0)

    problems = []

    # ---------- Rosenbrock ----------
    problems.append(
        Rosenbrock("P7_Rosenbrock_2", x0=np.array([-1.2, 1.0]), n=2)
    )

    x0_rosen_100 = np.ones(100)
    x0_rosen_100[0] = -1.2
    problems.append(
        Rosenbrock("P8_Rosenbrock_100", x0=x0_rosen_100, n=100)
    )

    return problems



def build_methods():
    methods = []
    
    # use LBFGS methods
    method_name_list = [
        "LBFGS",
    ]
    
    # use Wolf line search
    for name in method_name_list:
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

            # use default parameters for LBFGS
            output_path = (
                f"./output_rosenBrock/{prob.name}_{meth.method_name}_{meth.line_search}.txt"
            )
            option.set_option_params(output=output_path)

            print(f"Running {prob.name} with {meth.method_name}_{meth.line_search} ...")

            try:

                # run solver with statistical output
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
                
                # append result for each problem
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


            # save result table
            df = pd.DataFrame(rows)
            df.to_csv("summary_table_rosenbrock.csv", index=False)

    print("\nSummary table written to summary_table.csv")
    print(df)


if __name__ == "__main__":
    main()


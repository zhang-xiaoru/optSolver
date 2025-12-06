from optimizer import Method, Option, optSolver_OptimizationHunter_with_stats
from testing import objectives
import pandas as pd

problems = [
    objectives.P1_quad_10_10(),
    objectives.P2_quad_10_1000(),
    # ...
]
methods = [
    Method("GD", "armijo"),
    Method("GD", "wolf"),
    Method("NewtonCG", "armijo"),
    Method("ModifiedNewton", "armijo"),
    Method("BFGS", "armijo"),
    Method("BFGS", "wolf"),
    Method("DFP", "armijo"),
    Method("DFP", "wolf"),
    Method("LBFGS", "armijo"),
    Method("LBFGS", "wolf"),
]

rows = []

for prob in problems:
    for meth in methods:
        opt = Option()
        opt.set_option_params(output="")

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

df = pd.DataFrame(rows)
df.to_csv("summary_table.csv", index=False)
print(df)

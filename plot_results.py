import os
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Directory where log files are stored, organized as:
# output/
#   P1_quad_10_10/
#       GD_armijo.txt
#       GD_wolf.txt
#       BFGS_armijo.txt
#       ...
OUTPUT_DIR = "output"

# Directory where figures will be saved
FIG_DIR = "figs"


def parse_log_file(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

    iters = []
    fvals = []
    gnorms = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip possible header / separator lines
            if line.startswith("Iter") or line.startswith("-"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                k = int(parts[0])
                fk = float(parts[1])
                gk = float(parts[2])
            except ValueError:
                continue

            iters.append(k)
            fvals.append(fk)
            gnorms.append(gk)

    if len(iters) == 0:
        return None, None, None

    return np.array(iters), np.array(fvals), np.array(gnorms)


METHOD_COLOR_MAP = {
    "GD": "#1f77b4",               
    "ModifiedNewton": "#2ca02c",   
    "NewtonCG": "#ff7f0e",         
    "BFGS": "#d62728",             
    "LBFGS": "#8d5dba",            
    "DFP": "#8c564b",              
}


def get_color(method_name: str) -> str:
    lower = method_name.lower()

    if "lbfgs" in lower:
        return "#9467bd"  
    if "bfgs" in lower:
        return "#d62728"   
    if "newtoncg" in lower:
        return "#ff7f0e"   
    if "modifiednewton" in lower:
        return "#2ca02c"   
    if "dfp" in lower:
        return "#8c564b"   
    if "gd" in lower:
        return "#1f77b4"  

    return "black"  



def get_linestyle(method_name: str) -> str:
    """Armijo → solid line, Wolfe/Wolf → dashed line, otherwise solid."""
    lower = method_name.lower()
    if "armijo" in lower:
        return "-"
    if "wolf" in lower or "wolfe" in lower:
        return "--"
    return "-"  # default


def main() -> None:
    # Ensure figure directory exists
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR, exist_ok=True)

    # Check output directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"Directory '{OUTPUT_DIR}' not found.")
        return

    # One subdirectory per problem
    problem_dirs = sorted(
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    )

    if not problem_dirs:
        print(f"No problem subdirectories found in '{OUTPUT_DIR}'.")
        return

    for prob in problem_dirs:
        prob_path = os.path.join(OUTPUT_DIR, prob)
        txt_files = sorted(
            f for f in os.listdir(prob_path)
            if f.endswith(".txt")
        )

        if not txt_files:
            print(f"No .txt files in '{prob_path}', skip.")
            continue

        print(f"Processing {prob} ...")

        # ---------- 1) f(x_k) vs iteration (log scale) ----------
        plt.figure()
        for txt in txt_files:
            method_name = os.path.splitext(txt)[0]  # e.g., "BFGS_armijo"
            path = os.path.join(prob_path, txt)
            iters, fvals, gnorms = parse_log_file(path)
            if iters is None:
                continue

            color = get_color(method_name)
            linestyle = get_linestyle(method_name)

            # Function values on log scale (as requested)
            plt.semilogy(
                iters,
                np.maximum(np.abs(fvals), 1e-20),  # guard against non-positive values
                label=method_name,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )

        plt.xlabel("Iteration")
        plt.ylabel("Function value $f(x_k)$ (log scale)")
        plt.title(f"{prob}: Function value vs. iteration")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{prob}_f_vs_iter.png"), dpi=200)
        plt.close()

        # ---------- 2) ||grad f(x_k)||_2 vs iteration (log scale) ----------
        plt.figure()
        for txt in txt_files:
            method_name = os.path.splitext(txt)[0]
            path = os.path.join(prob_path, txt)
            iters, fvals, gnorms = parse_log_file(path)
            if iters is None:
                continue

            color = get_color(method_name)
            linestyle = get_linestyle(method_name)

            plt.semilogy(
                iters,
                np.maximum(gnorms, 1e-20),
                label=method_name,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )

        plt.xlabel("Iteration")
        plt.ylabel(r"$\|\nabla f(x_k)\|_2$ (log scale)")
        plt.title(f"{prob}: Gradient norm vs. iteration")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{prob}_gradnorm_vs_iter.png"), dpi=200)
        plt.close()

        print(f"  Saved {prob}_f_vs_iter.png and {prob}_gradnorm_vs_iter.png")

    print("All plots generated into 'figs/' directory.")


if __name__ == "__main__":
    main()


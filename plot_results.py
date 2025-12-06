import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figs"


def parse_log_file(path: str):

    iters = []
    fvals = []
    gnorms = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Iter") or line.startswith("-"):
                continue
            if line.startswith("Terminated") or line.startswith("Terminated"):
                break

            parts = line.split()
            if len(parts) < 4:
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


def main():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(OUTPUT_DIR):
        print(f"Directory '{OUTPUT_DIR}' not found.")
        return

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

        # ---------- f(x_k) vs iteration ----------
        plt.figure()
        for txt in txt_files:
            method_name = os.path.splitext(txt)[0]  # e.g., "BFGS_armijo"
            path = os.path.join(prob_path, txt)
            iters, fvals, gnorms = parse_log_file(path)
            if iters is None:
                continue
            plt.semilogy(iters, fvals, label=method_name)

        plt.xlabel("Iteration")
        plt.ylabel("f(x_k)")
        plt.title(f"{prob}: f(x_k) vs iteration")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{prob}_f.png"), dpi=200)
        plt.close()

        # ---------- ||grad f(x_k)||^2 vs iteration ----------
        plt.figure()
        for txt in txt_files:
            method_name = os.path.splitext(txt)[0]
            path = os.path.join(prob_path, txt)
            iters, fvals, gnorms = parse_log_file(path)
            if iters is None:
                continue
            plt.semilogy(iters, gnorms ** 2, label=method_name)

        plt.xlabel("Iteration")
        plt.ylabel("||grad f(x_k)||^2")
        plt.title(f"{prob}: ||grad f(x_k)||^2 vs iteration")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{prob}_grad2.png"), dpi=200)
        plt.close()

        print(f"  Saved {prob}_f.png and {prob}_grad2.png")

    print("All plots generated into 'figs/' directory.")


if __name__ == "__main__":
    main()

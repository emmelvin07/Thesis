# training/abc_algorithm.py
import numpy as np

class ABCAlgorithm:
    """
    Stable Artificial Bee Colony (ABC) optimizer
    for expensive objective functions (CNN training).
    """

    def __init__(self, num_bees=12, limit=6, max_iter=6, bounds=None, rng_seed=None):
        self.num_bees = num_bees
        self.limit = limit
        self.max_iter = max_iter
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.rng = np.random.default_rng(rng_seed)

    def random_solution(self):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return self.rng.uniform(low, high)

    def apply_bounds(self, solution):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return np.clip(solution, low, high)

    def optimize(self, objective_fn, patience=3):
        foods = np.array([self.random_solution() for _ in range(self.num_bees)])
        fitness = np.array([objective_fn(sol) for sol in foods])
        trials = np.zeros(self.num_bees)

        best_idx = np.argmin(fitness)
        best_solution = foods[best_idx].copy()
        best_score = fitness[best_idx]

        no_improve = 0
        print(f"[ABC] Init best_score={best_score:.6f}")

        for cycle in range(1, self.max_iter + 1):

            # ---------------- EMPLOYED BEES ----------------
            for i in range(self.num_bees):
                k = self.rng.integers(self.num_bees)
                while k == i:
                    k = self.rng.integers(self.num_bees)

                phi = self.rng.uniform(-1, 1, self.dim)
                candidate = foods[i] + phi * (foods[i] - foods[k])
                candidate = self.apply_bounds(candidate)

                cand_fit = objective_fn(candidate)

                if cand_fit < fitness[i]:
                    foods[i] = candidate
                    fitness[i] = cand_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # ---------------- ONLOOKER BEES ----------------
            scores = np.exp(-fitness)
            probs = scores / (scores.sum() + 1e-12)

            for _ in range(self.num_bees):
                i = self.rng.choice(self.num_bees, p=probs)

                k = self.rng.integers(self.num_bees)
                while k == i:
                    k = self.rng.integers(self.num_bees)

                phi = self.rng.uniform(-1, 1, self.dim)
                candidate = foods[i] + phi * (foods[i] - foods[k])
                candidate = self.apply_bounds(candidate)

                cand_fit = objective_fn(candidate)

                if cand_fit < fitness[i]:
                    foods[i] = candidate
                    fitness[i] = cand_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # ---------------- SCOUT BEES ----------------
            for i in range(self.num_bees):
                if trials[i] >= self.limit:
                    foods[i] = self.random_solution()
                    fitness[i] = objective_fn(foods[i])
                    trials[i] = 0

            # ---------------- BEST TRACKING ----------------
            idx = np.argmin(fitness)
            if fitness[idx] < best_score:
                best_score = fitness[idx]
                best_solution = foods[idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            print(f"[ABC] Cycle {cycle}/{self.max_iter} best_score={best_score:.6f}")

            if no_improve >= patience:
                print("[ABC] Early stopping triggered.")
                break

        print(f"[ABC] Finished. Best score = {best_score:.6f}")
        return best_solution, best_score

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tqdm import tqdm

from esm.sdk import batch_executor
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.sdk.experimental.guided_generation import (
    ESM3GuidedDecoding,
    GuidedDecodingScoringFunction,
)


class ConstraintType(Enum):
    GREATER_EQUAL = "greater_equal"  #  f(x) ≥ threshold
    LESS_EQUAL = "less_equal"  #  f(x) ≤ threshold
    EQUAL = "equal"  #  f(x) = threshold


@dataclass(slots=True)
class GenerationConstraint:
    """
    A single inequality or equality constraint.

    Parameters
    ----------
    scoring_function
        Maps a protein ➜ real value (e.g. pTM, length, …).
    value
        Target value for inequality or equality constraint.
    constraint_type
        Type of constraint to apply.
        - GREATER_EQUAL: f(x) ≥ value
        - LESS_EQUAL: f(x) ≤ value
        - EQUAL: f(x) = value (equality)
    """

    scoring_function: GuidedDecodingScoringFunction
    value: float
    constraint_type: ConstraintType = ConstraintType.GREATER_EQUAL
    lambda_: float = field(default=0.0, init=False)  # dual variable (MDMM)

    def g(self, x: float) -> float:
        """
        Canonical form:

          • inequalities →  g(x) ≤ 0
          • equalities   →  h(x)  (we still return a scalar, no ≤)
        """
        if self.constraint_type is ConstraintType.GREATER_EQUAL:
            return self.value - x
        if self.constraint_type is ConstraintType.LESS_EQUAL:
            return x - self.value
        # equality: h(x) = x - value  (we will *not* project λ)
        return x - self.value

    def update_lambda(self, g: float, eta: float, gamma: float) -> None:
        """
        Update the dual variable λ according to the MDMM update rule.
        """
        if self.constraint_type is ConstraintType.EQUAL:
            self.lambda_ += eta * g  # no projection for equality constraints
        else:
            self.lambda_ = max(0.0, self.lambda_ + eta * g)

    def copy(self) -> GenerationConstraint:
        """
        Create a copy of this constraint.
        """
        c = GenerationConstraint(
            scoring_function=self.scoring_function,
            value=self.value,
            constraint_type=self.constraint_type,
        )
        c.lambda_ = self.lambda_  # copy the dual variable
        return c


class ESM3GuidedDecodingWithConstraints(ESM3GuidedDecoding):
    """
    Derivative-free guided decoding with constraints.

    Uses the Modified Differential Method of Multipliers (MDMM) to
    guarantee convergence to the constrained optimum without
    hand-tuning penalty weights.

    References:
    [1] Platt, John, and Alan Barr. "Constrained differential optimization." Neural Information Processing Systems. 1987.
    [2] https://www.engraved.blog/how-we-can-make-machine-learning-algorithms-tunable/
    """

    def __init__(
        self,
        client: ESM3InferenceClient,
        scoring_function: GuidedDecodingScoringFunction,
        constraints: GenerationConstraint | list[GenerationConstraint],
        *,
        damping: float = 10.0,
        learning_rate: float = 1.0,
    ):
        super().__init__(client, scoring_function)

        if isinstance(constraints, GenerationConstraint):
            constraints = [constraints]

        self.constraints = [c.copy() for c in constraints]
        self.gamma = float(damping)
        self.eta = float(learning_rate)

        self.recorder: TrajectoryRecorder | None = None

    def guided_generate(
        self,
        protein: ESMProtein,
        num_decoding_steps: int,
        num_samples_per_step: int,
        denoised_prediction_temperature: float = 0.0,
        track: str = "sequence",
        verbose: bool = True,
    ) -> ESMProtein:
        # Reset the trajectory recorder
        self.recorder = TrajectoryRecorder()

        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)

        if track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)

        n_masked = self.get_number_of_masked_positions(protein_tensor, track=track)
        n_unmask = n_masked // num_decoding_steps

        best_reward = float("-inf")

        if verbose:
            pbar = tqdm(range(num_decoding_steps), desc="S=-inf  λ=0.00")
        else:
            pbar = range(num_decoding_steps)

        for step in pbar:
            # Last iteration: unmask whatever is left
            if step == num_decoding_steps - 1:
                n_unmask = self.get_number_of_masked_positions(
                    protein_tensor, track=track
                )

            # ---------- propose & evaluate M samples (parallel-safe) ---- #
            def _propose_and_eval(pt: ESMProteinTensor):
                new_pt = self.randomly_unmask_positions(pt, n_unmask, track=track)
                reward, g_val, raw_vals = self._score_and_constraints(
                    new_pt, denoised_prediction_temperature
                )
                return new_pt, reward, g_val, raw_vals

            if self._use_batch_executor:
                with batch_executor(show_progress=False) as ex:
                    results = ex.execute_batch(
                        user_func=_propose_and_eval,
                        pt=[protein_tensor] * num_samples_per_step,
                    )

                if isinstance(results, Exception):
                    raise results
                samples, rewards, gh_lists, val_lists = zip(*results)
            else:
                samples, rewards, gh_lists, val_lists = [], [], [], []
                for _ in range(num_samples_per_step):
                    s, r, g, c = _propose_and_eval(protein_tensor)
                    samples.append(s)
                    rewards.append(r)
                    gh_lists.append(g)
                    val_lists.append(c)

            # -------- compute MDMM lagrangian for each sample -----------
            lags = [self._lagrangian(r, ghs) for r, ghs in zip(rewards, gh_lists)]

            best_idx = int(torch.tensor(lags).argmin())
            protein_tensor = samples[best_idx]
            best_reward = rewards[best_idx]
            best_g_vals = gh_lists[best_idx]

            # -------- dual updates (MDMM) -----------------
            for g, c in zip(best_g_vals, self.constraints):
                c.update_lambda(g, self.eta, self.gamma)

            self.recorder.log(
                step=step,
                reward=best_reward,
                g_list=best_g_vals,
                lambda_list=[c.lambda_ for c in self.constraints],
            )

            if verbose and isinstance(pbar, tqdm):
                lam_str = ", ".join(
                    f"λ_{i}={c.lambda_:.2f}" for i, c in enumerate(self.constraints)
                )
                pbar.set_description(f"S={best_reward:+.3f}  {lam_str}")

        final = self.client.forward_and_sample(
            protein_tensor,
            sampling_configuration=SamplingConfig(
                sequence=SamplingTrackConfig(temperature=0.0),
                structure=SamplingTrackConfig(temperature=0.0),
            ),
        )
        assert not isinstance(final, ESMProteinError)
        decoded = self.client.decode(final.protein_tensor)
        assert not isinstance(decoded, ESMProteinError)
        return decoded

    def visualize_latest_trajectory(
        self, constraint_idx: int = 0, cmap: str = "viridis"
    ) -> None:
        """
        Visualise the trajectory of the latest optimisation run.
        If you optimise multiple constraints, pick which one to plot via `constraint_idx`.
        """
        if not self.recorder:
            raise ValueError("No trajectory recorder available.")

        steps, g_vals, rewards = self.recorder.as_arrays(constraint_idx)
        self.recorder.plot_line(constraint_idx=constraint_idx, cmap=cmap)

    def _score_and_constraints(
        self, pt: ESMProteinTensor, temp: float
    ) -> tuple[float, list[float], list[float]]:
        protein = self.predict_denoised(pt, temperature=temp)
        reward = self.scoring_function(protein)
        vals, ghs = [], []
        for c in self.constraints:
            val = c.scoring_function(protein)
            vals.append(val)
            ghs.append(c.g(val))
        return reward, ghs, vals

    def _lagrangian(self, reward: float, g_vals: list[float]) -> float:
        """
        MDMM L(x, λ)  =  -reward  + Σ_i (λ_i - γ g_i) * g_i
        (reward is to be maximised ⇒ we minimise -reward)
        """
        lag = -reward
        for g, c in zip(g_vals, self.constraints):
            lag += (c.lambda_ - self.gamma * g) * g
        return lag


@dataclass
class TrajectoryRecorder:
    steps: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    g_vals: List[List[float]] = field(
        default_factory=list
    )  # each step → list of constraints
    lambdas: List[List[float]] = field(default_factory=list)  # each step → list of λ  s

    def log(
        self, step: int, reward: float, g_list: list[float], lambda_list: list[float]
    ) -> None:
        """Append one optimisation step to the trajectory."""
        self.steps.append(step)
        self.rewards.append(reward)
        self.g_vals.append(list(g_list))
        self.lambdas.append(list(lambda_list))

    def as_arrays(self, constraint_idx: int = 0):
        """
        Return numpy arrays suitable for plotting.
        If you optimise multiple constraints, pick which one to plot via `constraint_idx`.
        """
        return (
            np.asarray(self.steps),
            np.asarray([g[constraint_idx] for g in self.g_vals]),
            np.asarray(self.rewards),
        )

    def plot_line(self, constraint_idx: int = 0, cmap: str = "viridis"):
        """
        Continuous line with markers and a colour-gradient that follows the optimisation step.
        """
        steps, x_vals, y_vals = self.as_arrays(constraint_idx)

        # build coloured line segments
        points = np.column_stack([x_vals, y_vals])
        segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)

        norm = Normalize(vmin=steps.min(), vmax=steps.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)  # type: ignore
        lc.set_array(steps)

        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.scatter(
            x_vals,
            y_vals,
            c=steps,
            cmap=cmap,
            norm=norm,
            marker="o",
            edgecolor="k",
            zorder=3,
        )

        ax.axvline(0, linestyle="--", color="grey")
        ax.set_xlabel("constraint value  g(x)   (≤ 0 is feasible)")
        ax.set_ylabel("reward  R(x)")
        ax.set_title("Trajectory in constraint–reward space")
        plt.colorbar(lc, label="optimisation step")
        plt.tight_layout()
        plt.show()

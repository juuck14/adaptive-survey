"""Common population MNL analysis, independent of policy-specific embeddings.

The design matches the public handoff's 34-column mean-utility specification.
Under MMNL generation these fits are population MNL projections, not recovery
of random-coefficient parameters. Adaptive sampling can change that projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import logsumexp

from .schema import ATTRIBUTES, CONTEXTS, PERSON_FIELDS, Profile, Task

EDUCATION_FIELDS = ("EDU_1", "EDU_2", "EDU_3", "EDU_5", "EDU_6")
DEFAULT_FEATURE_NAMES = ("ASC_A", "ASC_B", "ASC_C") + ATTRIBUTES + (
    "NONE_LAYOVER", "NONE_STOPOVER", "MSA_PxGENZ", "MSA_FxGENZ",
    "MSA_PxGENDf", "MSA_FxGENDf", "MSA_PxGENDunmapped", "MSA_FxGENDunmapped",
) + tuple(f"{attribute}x{education}" for education in EDUCATION_FIELDS
          for attribute in ("ENV_2", "ENV_3"))


def design_matrix(task: Task, profile: Profile) -> np.ndarray:
    """Return A/B/C/NONE x 34 mean-utility features in handoff order."""
    attributes = np.asarray(task.attributes, dtype=np.float64)
    person_values = np.asarray(profile.features, dtype=np.float64)
    if attributes.shape != (4, len(ATTRIBUTES)) or person_values.shape != (len(PERSON_FIELDS),):
        raise ValueError("Task/profile have incompatible feature dimensions")
    if not np.all(np.isfinite(attributes)) or not np.all(np.isfinite(person_values)):
        raise ValueError("Nonfinite design inputs")
    if task.context not in CONTEXTS or np.any(attributes[3] != 0):
        raise ValueError("Invalid context or nonzero NONE airport attributes")
    person = dict(zip(PERSON_FIELDS, person_values))
    x = np.zeros((4, len(DEFAULT_FEATURE_NAMES)), dtype=np.float64)
    x[:3, :3] = np.eye(3)
    x[:3, 3:16] = attributes[:3]
    x[3, 16] = task.context == "layover"
    x[3, 17] = task.context == "stopover"
    p = attributes[:3, ATTRIBUTES.index("MSA_P")]
    f = attributes[:3, ATTRIBUTES.index("MSA_F")]
    unmapped = float(bool(person["GEND_O"] or person["GEND_PNO"]))
    x[:3, 18:24] = np.stack((p * person["GEN_Z"], f * person["GEN_Z"],
                            p * person["GEND_F"], f * person["GEND_F"],
                            p * unmapped, f * unmapped), axis=1)
    x[:3, 24:] = np.stack([attributes[:3, ATTRIBUTES.index(attribute)] * person[education]
                          for education in EDUCATION_FIELDS
                          for attribute in ("ENV_2", "ENV_3")], axis=1)
    return x


@dataclass(frozen=True, slots=True)
class ChoiceRecord:
    profile: Profile
    task: Task
    choice: int


@dataclass(frozen=True, slots=True)
class MNLFit:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    active_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    rank: int
    converged: bool
    log_likelihood: float
    n_observations: int
    n_respondents: int
    l2: float
    gradient_inf_norm: float
    iterations: int
    standard_errors: np.ndarray | None
    se_type: str | None

    def predict(self, profile: Profile, tasks: Sequence[Task]) -> np.ndarray:
        if not tasks:
            return np.empty((0, 4), dtype=float)
        x = np.stack([design_matrix(t, profile) for t in tasks])
        utility = x @ self.coefficients
        return np.exp(utility - logsumexp(utility, axis=1, keepdims=True))

    def as_dict(self) -> dict:
        return {
            "coefficients": {name: float(value) for name, value in zip(self.feature_names, self.coefficients)
                             if name in self.active_features},
            "active_features": list(self.active_features), "dropped_features": list(self.dropped_features),
            "rank": self.rank, "converged": self.converged,
            "log_likelihood": self.log_likelihood, "n_observations": self.n_observations,
            "n_respondents": self.n_respondents, "l2": self.l2,
            "gradient_inf_norm": self.gradient_inf_norm, "iterations": self.iterations,
            "standard_errors": None if self.standard_errors is None else {
                name: float(value) for name, value in zip(self.feature_names, self.standard_errors)
                if name in self.active_features},
            "se_type": self.se_type,
            "interpretation": "common population MNL; MMNL parameters are not recovered",
        }


def fit_pooled_mnl(
    records: Iterable[ChoiceRecord], l2: float = 1e-4,
    drop_unidentified: bool = False, cluster_se: bool = False,
    max_iter: int = 500, active_features: Sequence[str] | None = None,
) -> MNLFit:
    """Fit a common specification; structural rank failure is explicit.

    If ``drop_unidentified`` is explicitly enabled, pivoted QR chooses an
    estimable basis. Dropped names are returned, and their prediction-only
    coefficient placeholders are zero; they are not estimates of zero.
    Cluster standard errors use centered scores, UUID groups and the penalized
    Hessian; they describe the regularized fit rather than unpenalized MLE.
    """
    data = tuple(records)
    if not data or not np.isfinite(l2) or l2 < 0 or max_iter < 1:
        raise ValueError("Need observations, nonnegative finite regularization and iterations")
    y = np.asarray([r.choice for r in data])
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 3)):
        raise ValueError("Choices must be integers in [0, 3]")
    x = np.stack([design_matrix(r.task, r.profile) for r in data])
    # Only alternative utility contrasts are identifiable.
    contrasts = (x[:, :3] - x[:, 3:4]).reshape(-1, x.shape[-1])
    n_features = len(DEFAULT_FEATURE_NAMES)
    if active_features is None:
        requested = np.arange(n_features)
    else:
        names = tuple(active_features)
        if not names or len(set(names)) != len(names) or set(names) - set(DEFAULT_FEATURE_NAMES):
            raise ValueError("Active feature names must be unique known columns")
        requested = np.asarray([i for i, name in enumerate(DEFAULT_FEATURE_NAMES) if name in names])
    _, diagonal, pivots = qr(contrasts[:, requested], mode="economic", pivoting=True)
    diag = np.abs(np.diag(diagonal))
    threshold = max(contrasts.shape) * np.finfo(float).eps * (diag.max() if len(diag) else 0)
    rank = int(np.sum(diag > threshold))
    if rank == 0:
        raise ValueError("No identifiable utility contrasts")
    if rank < len(requested) and not drop_unidentified:
        dropped = [DEFAULT_FEATURE_NAMES[requested[i]] for i in sorted(pivots[rank:])]
        raise ValueError(f"Rank-deficient MNL design ({rank}/{len(requested)}); explicit reduction required: {dropped}")
    active = np.sort(requested[pivots[:rank]])
    inactive = np.asarray(sorted(set(range(n_features)) - set(active)), dtype=int)
    xa = x[:, :, active]
    count = len(data)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        utility = xa @ beta
        log_z = logsumexp(utility, axis=1)
        p = np.exp(utility - log_z[:, None])
        loss = np.mean(log_z - utility[np.arange(count), y]) + 0.5 * l2 * (beta @ beta)
        residual = p.copy()
        residual[np.arange(count), y] -= 1
        gradient = np.einsum("na,naf->f", residual, xa) / count + l2 * beta
        return float(loss), gradient

    result = minimize(objective, np.zeros(rank), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(f"MNL optimization failed: {result.message}")
    coefficients = np.zeros(n_features, dtype=float)
    coefficients[active] = result.x
    utility = xa @ result.x
    log_z = logsumexp(utility, axis=1)
    p = np.exp(utility - log_z[:, None])
    log_likelihood = float(np.sum(utility[np.arange(count), y] - log_z))
    standard_errors = None
    se_type = None
    n_people = len({r.profile.uuid for r in data})
    if cluster_se:
        if n_people < 2:
            raise ValueError("Respondent-cluster standard errors require at least two UUIDs")
        mean_x = np.einsum("na,naf->nf", p, xa)
        hessian = (np.einsum("na,naf,nag->fg", p, xa, xa)
                   - mean_x.T @ mean_x + count * l2 * np.eye(rank))
        if not np.all(np.isfinite(hessian)) or np.linalg.matrix_rank(hessian) < rank:
            raise RuntimeError("MNL Hessian is singular; standard errors unavailable")
        bread = np.linalg.inv(hessian)
        individual_scores = xa[np.arange(count), y] - mean_x
        # A ridge optimum has nonzero mean data score. The penalty is fixed
        # across samples, so remove that mean before the cluster meat.
        individual_scores -= individual_scores.mean(axis=0)
        cluster_scores: dict[str, np.ndarray] = {}
        for record, score in zip(data, individual_scores):
            cluster_scores.setdefault(record.profile.uuid, np.zeros(rank))
            cluster_scores[record.profile.uuid] += score
        score_matrix = np.stack(list(cluster_scores.values()))
        correction = n_people / (n_people - 1)
        if count > rank:
            correction *= (count - 1) / (count - rank)
        covariance = correction * bread @ (score_matrix.T @ score_matrix) @ bread
        variances = np.diag(covariance)
        if np.any(variances < -1e-10) or not np.all(np.isfinite(variances)):
            raise RuntimeError("Invalid clustered covariance")
        standard_errors = np.full(n_features, np.nan)
        standard_errors[active] = np.sqrt(np.maximum(variances, 0))
        se_type = "UUID-cluster sandwich with centered scores and penalized Hessian"
    return MNLFit(coefficients, DEFAULT_FEATURE_NAMES,
                  tuple(DEFAULT_FEATURE_NAMES[i] for i in active),
                  tuple(DEFAULT_FEATURE_NAMES[i] for i in inactive), rank, True,
                  log_likelihood, count, n_people, float(l2),
                  float(np.max(np.abs(objective(result.x)[1]))), int(result.nit),
                  standard_errors, se_type)


def coefficient_agreement(selected: MNLFit, reference: MNLFit) -> dict:
    """Report common identified coefficients, never latent-MMNL recovery."""
    if selected.active_features != reference.active_features:
        raise ValueError("Coefficient agreement requires the same identifiable feature basis")
    active = set(selected.active_features) & set(reference.active_features)
    names = tuple(n for n in selected.feature_names if n in active)
    if not names:
        raise ValueError("No common identifiable coefficients")
    a = np.asarray([selected.coefficients[selected.feature_names.index(n)] for n in names])
    b = np.asarray([reference.coefficients[reference.feature_names.index(n)] for n in names])
    norm = float(np.linalg.norm(b))
    informative = np.abs(b) > 1e-8
    return dict(feature_names=list(names), rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                l2_difference=float(np.linalg.norm(a - b)),
                relative_l2=None if norm <= 1e-12 else float(np.linalg.norm(a - b) / norm),
                sign_agreement=None if not informative.any() else float(np.mean(np.sign(a[informative]) == np.sign(b[informative]))),
                interpretation="reference agreement; adaptive sampling can change a misspecified MNL projection")


def fit_selected_vs_full(
    selected_records: Iterable[ChoiceRecord], full_records: Iterable[ChoiceRecord], **fit_options,
) -> dict:
    """Compare subsets only against the same UUID/context population."""
    selected = tuple(selected_records)
    full = tuple(full_records)
    if {(r.profile.uuid, r.task.context) for r in selected} != {(r.profile.uuid, r.task.context) for r in full}:
        raise ValueError("Subset and reference need identical UUID/context populations")
    lookup = {(r.profile.uuid, r.task.task_id): r for r in full}
    if len(lookup) != len(full):
        raise ValueError("Duplicate full-survey observations")
    seen = set()
    for r in selected:
        key = (r.profile.uuid, r.task.task_id)
        if key in seen or key not in lookup or lookup[key] != r:
            raise ValueError("Subset observations are duplicated or disagree with full reference")
        seen.add(key)
    reference_fit = fit_pooled_mnl(full, **fit_options)
    subset_options = dict(fit_options)
    subset_options["active_features"] = reference_fit.active_features
    # A reference basis is part of the comparison contract. Changing it for a
    # smaller subset would turn coefficient differences into a coding artifact.
    subset_options["drop_unidentified"] = False
    subset_fit = fit_pooled_mnl(selected, **subset_options)
    return dict(selected=subset_fit.as_dict(), reference=reference_fit.as_dict(),
                agreement=coefficient_agreement(subset_fit, reference_fit))

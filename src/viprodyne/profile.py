"""Profile-likelihood helpers for structured inference workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np

from viprodyne.fit import CAVIConfig
from viprodyne.model import (
    MS2Dataset,
    ModelConfig,
    ModelInferenceResult,
    ViprodyneModel,
)

FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class ContactThresholdProfileResult:
    """Result from fitting a contact-threshold grid."""

    candidate_values: np.ndarray
    elbos: np.ndarray
    fits: tuple[ModelInferenceResult, ...]

    @property
    def best_index(self) -> int:
        """Index of the highest-ELBO candidate."""
        return int(np.argmax(self.elbos))

    @property
    def best_value(self) -> np.float32:
        """Highest-ELBO threshold value."""
        return np.float32(self.candidate_values[self.best_index])

    @property
    def best_fit(self) -> ModelInferenceResult:
        """Inference result for the highest-ELBO threshold."""
        return self.fits[self.best_index]


def profile_contact_threshold(
    datasets: tuple[MS2Dataset, ...],
    config: ModelConfig,
    candidate_values: np.ndarray | None = None,
    *,
    fit_config: CAVIConfig | None = None,
    verbose: bool = False,
    **kwargs,
) -> ContactThresholdProfileResult:
    """Fit one model per contact-threshold candidate.

    Contact drives are read from `config.contact_drives`. Each candidate
    produces a fresh `ViprodyneModel` with `rc` pinned to that candidate through
    the same model-construction path used by MAP rc fitting.
    """
    if candidate_values is None:
        if config.rc_candidate_values is None:
            raise ValueError("candidate_values or config.rc_candidate_values must be set.")
        candidate_values = config.rc_candidate_values
    candidate_values = np.asarray(candidate_values, dtype=FLOAT_DTYPE)
    if candidate_values.ndim != 1 or candidate_values.size == 0:
        raise ValueError("candidate_values must be a non-empty one-dimensional array.")
    if not config.driven_transition_indices:
        raise ValueError(
            "config.driven_transition_indices must be set for contact profiling."
        )
    fit_config = (
        CAVIConfig(compute_elbo=True, **kwargs) if fit_config is None else fit_config
    )
    if not fit_config.compute_elbo:
        fit_config = replace(fit_config, compute_elbo=True)

    fits: list[ModelInferenceResult] = []
    elbos: list[np.float32] = []
    for candidate in candidate_values:
        if verbose:
            print(
                f"Profiling contact threshold candidate {candidate}...({fits.__len__() + 1}/{candidate_values.size})"
            )
        candidate_config = replace(
            config,
            rc_initial=np.asarray(candidate, dtype=FLOAT_DTYPE),
            rc_candidate_values=np.asarray([candidate], dtype=FLOAT_DTYPE),
        )
        model = ViprodyneModel(datasets, candidate_config)
        contact_mass = _model_contact_mass(model)
        if contact_mass <= 0.0:
            warnings.warn(
                "contact threshold candidate "
                f"{float(candidate):.6g} produced zero contact probability across "
                "all datasets; driven transitions are disabled and driven rates are "
                "unidentifiable for this candidate.",
                UserWarning,
                stacklevel=2,
            )
        fit = model.run_inference(config=fit_config)
        if fit.cavi is None or fit.cavi.elbo is None:
            raise RuntimeError("contact-threshold profiling requires ELBO computation.")
        fits.append(fit)
        elbos.append(np.asarray(fit.cavi.elbo, dtype=FLOAT_DTYPE))

    return ContactThresholdProfileResult(
        candidate_values=candidate_values.astype(FLOAT_DTYPE),
        elbos=np.asarray(elbos, dtype=FLOAT_DTYPE),
        fits=tuple(fits),
    )


def _model_contact_mass(model: ViprodyneModel) -> float:
    total = 0.0
    for nodes in model.dataset_nodes.values():
        contact_name = nodes["contact_drive"]
        if contact_name is None:
            continue
        moments = model.graph.moments.get(str(contact_name))
        if "p_contact" in moments:
            total += float(np.sum(np.asarray(moments["p_contact"], dtype=FLOAT_DTYPE)))
    return total

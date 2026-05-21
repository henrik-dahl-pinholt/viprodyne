"""Profile-likelihood helpers for structured inference workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import warnings

import numpy as np

from viprodyne.fit import CAVIConfig
from viprodyne.model import (
    ContactDrive,
    MS2Dataset,
    ModelConfig,
    ModelInferenceResult,
    ViprodyneModel,
    _assign_dataset_names,
    _interval_contact_probability,
    _validate_time_grid,
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
    contact_scores: Mapping[str, np.ndarray | Callable] | np.ndarray | Callable,
    candidate_values: np.ndarray,
    *,
    fit_config: CAVIConfig | None = None,
    less_than: bool = True,
    verbose: bool = False,
    **kwargs,
) -> ContactThresholdProfileResult:
    """Fit one model per contact-threshold candidate.

    This supports workflows where an external score `z(t)` is converted into a
    contact probability by thresholding, for example `p_contact(t) = z(t) < rc`.
    `contact_scores` may also be a function `fn(rc)` or `fn(time_grid, rc)`
    returning contact probabilities directly; mappings can provide arrays or
    functions per dataset.
    Each candidate produces a fresh `ViprodyneModel` with a model-level fixed
    `ContactDrive`; the input datasets are not modified.
    """
    candidate_values = np.asarray(candidate_values, dtype=FLOAT_DTYPE)
    if candidate_values.ndim != 1 or candidate_values.size == 0:
        raise ValueError("candidate_values must be a non-empty one-dimensional array.")
    if not config.driven_transition_indices:
        raise ValueError(
            "config.driven_transition_indices must be set for contact profiling."
        )
    datasets = _assign_dataset_names(tuple(datasets))
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
        contact_drive = {
            dataset.name: ContactDrive.fixed(
                _profile_contact_probability(
                    contact_scores,
                    dataset=dataset,
                    config=config,
                    candidate=candidate,
                    less_than=less_than,
                )
            )
            for dataset in datasets
        }
        contact_mass = sum(
            float(np.sum(np.asarray(drive.probability, dtype=FLOAT_DTYPE)))
            for drive in contact_drive.values()
            if drive.probability is not None
        )
        if contact_mass <= 0.0:
            warnings.warn(
                "contact threshold candidate "
                f"{float(candidate):.6g} produced zero contact probability across "
                "all datasets; driven transitions are disabled and driven rates are "
                "unidentifiable for this candidate.",
                UserWarning,
                stacklevel=2,
            )
        model = ViprodyneModel(datasets, config, contact_drive=contact_drive)
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


def _profile_contact_probability(
    contact_scores: Mapping[str, np.ndarray | Callable] | np.ndarray | Callable,
    *,
    dataset: MS2Dataset,
    config: ModelConfig,
    candidate: np.ndarray,
    less_than: bool,
) -> np.ndarray:
    source = _contact_source_for_dataset(contact_scores, dataset.name)
    if callable(source):
        time_grid = _profile_time_grid(dataset, config)
        try:
            values = source(time_grid, candidate)
        except TypeError as first_error:
            try:
                values = source(candidate)
            except TypeError:
                raise first_error
        return _interval_contact_probability(
            np.asarray(values, dtype=FLOAT_DTYPE),
            n_intervals=time_grid.size - 1,
        )
    return _threshold_contact_score(
        np.asarray(source, dtype=FLOAT_DTYPE),
        candidate,
        less_than=less_than,
    )


def _contact_source_for_dataset(
    contact_scores: Mapping[str, np.ndarray | Callable] | np.ndarray | Callable,
    dataset_name: str,
) -> np.ndarray | Callable:
    if isinstance(contact_scores, Mapping):
        try:
            return contact_scores[dataset_name]
        except KeyError as exc:
            raise KeyError(
                f"missing contact score for dataset {dataset_name!r}."
            ) from exc
    return contact_scores


def _profile_time_grid(dataset: MS2Dataset, config: ModelConfig) -> np.ndarray:
    if dataset.time_grid is not None:
        return _validate_time_grid(dataset.time_grid, f"{dataset.name}.time_grid")
    if config.time_grid is not None:
        return _validate_time_grid(config.time_grid, "time_grid")
    raise ValueError("each dataset needs time_grid, or ModelConfig.time_grid must be set.")


def _threshold_contact_score(
    score: np.ndarray,
    candidate: np.ndarray,
    *,
    less_than: bool,
) -> np.ndarray:
    score = np.asarray(score, dtype=FLOAT_DTYPE)
    candidate = np.asarray(candidate, dtype=FLOAT_DTYPE)
    contact = score < candidate if less_than else score > candidate
    return np.asarray(contact, dtype=FLOAT_DTYPE)

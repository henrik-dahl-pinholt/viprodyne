import jax.numpy as jnp
import numpy as np

from viprodyne.core.ms2_kernels import (
    ProximalKernel,
    build_ms2_observation_model,
    proximal_kernel,
)


def test_proximal_kernel_is_float32_ramp_then_plateau():
    values = proximal_kernel(
        jnp.asarray([-0.5, 0.0, 0.5, 1.0, 2.5, 3.1], dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
        jnp.asarray(4.0, dtype=jnp.float32),
    )

    assert values.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(values),
        np.array([0.0, 0.0, 2.0, 4.0, 4.0, 0.0], dtype=np.float32),
    )


def test_observation_model_builds_transfer_windows_from_kernel():
    model = build_ms2_observation_model(
        time_grid=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        n_observations=2,
        kernel=ProximalKernel(
            t_rise=np.float32(0.5),
            t_plateau=np.float32(1.5),
            rna_intensity=np.float32(2.0),
        ),
        mode="transfer",
    )

    assert model.design_matrix is None
    np.testing.assert_array_equal(model.observation_starts, np.array([0, 0], dtype=np.int32))
    np.testing.assert_allclose(
        model.window_weights,
        np.array([[2.0, 0.0], [2.0, 2.0]], dtype=np.float32),
    )


def test_observation_model_builds_dense_inputs_for_mean_field_mode():
    model = build_ms2_observation_model(
        time_grid=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        n_observations=2,
        kernel=ProximalKernel(
            t_rise=np.float32(0.5),
            t_plateau=np.float32(0.5),
            rna_intensity=np.float32(3.0),
        ),
        mode="mean_field",
    )

    assert model.window_weights is None
    np.testing.assert_allclose(
        model.design_matrix,
        np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )

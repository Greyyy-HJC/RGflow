import numpy as np

from rgflow.phi4.blocking import (
    KERNEL_SUM,
    apply_kernel,
    block_field,
    identity_kernel,
    inverse_kernel,
    kernel_from_parameters,
    kernel_transform,
    sos_filters_from_parameters,
    sos_kernel_from_parameters,
    spectrum_diagnostics,
)


def test_kernel_has_d4_symmetry_and_fixed_sum() -> None:
    parameters = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    kernel = kernel_from_parameters(parameters)

    np.testing.assert_allclose(kernel, np.rot90(kernel))
    np.testing.assert_allclose(kernel, np.flip(kernel, axis=0))
    np.testing.assert_allclose(np.sum(kernel), KERNEL_SUM)
    np.testing.assert_allclose(
        kernel[2, 2],
        KERNEL_SUM - np.array([4, 4, 4, 8, 4]) @ parameters,
    )


def test_identity_kernel_scales_and_even_even_downsamples() -> None:
    field = np.arange(64, dtype=np.float64).reshape(8, 8)
    kernel = identity_kernel()

    np.testing.assert_allclose(apply_kernel(field, kernel), KERNEL_SUM * field)
    np.testing.assert_allclose(
        block_field(field, kernel),
        KERNEL_SUM * field[0::2, 0::2],
    )


def test_periodic_kernel_wraps_across_boundaries() -> None:
    field = np.zeros((8, 8), dtype=np.float64)
    field[0, 0] = 1.0
    parameters = np.zeros(5, dtype=np.float64)
    parameters[0] = 0.1
    kernel = kernel_from_parameters(parameters)
    transformed = apply_kernel(field, kernel)

    assert transformed[0, 1] == 0.1
    assert transformed[0, -1] == 0.1
    assert transformed[1, 0] == 0.1
    assert transformed[-1, 0] == 0.1


def test_fourier_inverse_round_trip() -> None:
    rng = np.random.default_rng(41)
    field = rng.normal(size=(3, 16, 16))
    kernel = kernel_from_parameters(
        np.array([-0.025, -0.008, 0.03, 0.02, -0.001])
    )
    transformed = apply_kernel(field, kernel)

    np.testing.assert_allclose(
        inverse_kernel(transformed, kernel),
        field,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    diagnostics = spectrum_diagnostics(kernel, 16)
    assert diagnostics["min_abs_K"] > 0.0
    assert diagnostics["condition_number"] >= 1.0


def test_sos_kernel_is_invertible_by_construction() -> None:
    rng = np.random.default_rng(43)
    channels = 2
    floor_fraction = 0.05
    parameters = rng.normal(size=8 * channels)
    filters = sos_filters_from_parameters(parameters, channels)
    kernel = sos_kernel_from_parameters(
        parameters,
        channels=channels,
        floor_fraction=floor_fraction,
    )

    np.testing.assert_allclose(
        np.sum(filters, axis=(1, 2)),
        [1.0, 0.0],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(kernel, np.rot90(kernel))
    np.testing.assert_allclose(kernel, np.flip(kernel, axis=0))
    np.testing.assert_allclose(np.sum(kernel), KERNEL_SUM)
    spectrum = kernel_transform(kernel, 128)
    np.testing.assert_allclose(spectrum.imag, 0.0, atol=1.0e-12)
    assert np.min(spectrum.real) >= KERNEL_SUM * floor_fraction - 1.0e-12

    field = rng.normal(size=(3, 16, 16))
    np.testing.assert_allclose(
        inverse_kernel(apply_kernel(field, kernel), kernel),
        field,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_zero_sos_parameters_give_scaled_identity() -> None:
    kernel = sos_kernel_from_parameters(np.zeros(16), channels=2)
    np.testing.assert_allclose(kernel, identity_kernel())

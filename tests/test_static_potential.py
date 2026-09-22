import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "heatbath" / "static_potential.py"
SPEC = importlib.util.spec_from_file_location("static_potential", SCRIPT)
static_potential = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(static_potential)


def test_extract_fit_recovers_cornell_parameters():
    radii = np.arange(1, 7, dtype=float)
    times = np.arange(1, 7, dtype=float)
    expected = np.array([0.4, 0.06, 0.25])
    potential = expected[0] + expected[1] * radii - expected[2] / radii
    loops = np.exp(-potential[:, None] * times[None, :])

    fitted_potential, fitted = static_potential.extract_fit(loops, 2, 5, 2, 6)

    np.testing.assert_allclose(fitted_potential, potential, atol=1e-12)
    np.testing.assert_allclose(fitted[:3], expected, atol=1e-12)
    np.testing.assert_allclose(fitted[3], np.sqrt((1.65 - expected[2]) / expected[1]), atol=1e-12)


def test_rectangular_loop_count_and_closure():
    loops = static_potential.rectangular_loops(3, 4)

    assert len(loops) == 3 * 3 * 4
    for loop in loops:
        displacement = np.zeros(4, dtype=int)
        for direction in loop:
            if direction < 4:
                displacement[direction] += 1
            else:
                displacement[direction - 4] -= 1
        np.testing.assert_array_equal(displacement, 0)

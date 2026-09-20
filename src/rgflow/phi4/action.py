"""Two-dimensional scalar phi-four lattice action."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class Phi4Action:
    """Kappa-lambda phi-four action with periodic boundary conditions."""

    kappa: float = 0.340301
    lam: float = 1.0

    def potential(self, field: FloatArray) -> FloatArray:
        field_sq = field * field
        return field_sq + self.lam * (field_sq - 1.0) ** 2

    def __call__(self, field: FloatArray) -> FloatArray:
        """Return one action value per configuration."""
        axes = (-2, -1)
        hopping = field * np.roll(field, -1, axis=-2)
        hopping += field * np.roll(field, -1, axis=-1)
        return np.sum(self.potential(field) - 2.0 * self.kappa * hopping, axis=axes)

    def neighbor_sum(self, field: FloatArray) -> FloatArray:
        """Sum the four nearest neighbors at every lattice site."""
        return (
            np.roll(field, 1, axis=-2)
            + np.roll(field, -1, axis=-2)
            + np.roll(field, 1, axis=-1)
            + np.roll(field, -1, axis=-1)
        )

    def local_delta(self, field: FloatArray, proposal: FloatArray) -> FloatArray:
        """Action change for independently replacing each site by ``proposal``."""
        delta_field = proposal - field
        return (
            self.potential(proposal)
            - self.potential(field)
            - 2.0 * self.kappa * delta_field * self.neighbor_sum(field)
        )

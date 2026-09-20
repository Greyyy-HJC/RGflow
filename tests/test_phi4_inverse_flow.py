import json

import numpy as np
import torch

from rgflow.cli import main
from rgflow.phi4.affine_flow import (
    AffineFlowConfig,
    ConditionalAffineFlow,
    model_from_checkpoint,
    rational_quadratic_spline,
)
from rgflow.phi4.blocking import apply_kernel, identity_kernel
from rgflow.phi4.inverse_blocking import (
    assemble_details,
    extract_details,
    reblocking_error,
    reconstruct_fine,
)


def test_detail_coordinates_and_inverse_kernel_round_trip() -> None:
    rng = np.random.default_rng(101)
    coarse = rng.normal(size=(3, 4, 4))
    details = rng.normal(size=(3, 3, 4, 4))
    kernel = identity_kernel()

    psi = assemble_details(coarse, details)
    extracted_coarse, extracted_details = extract_details(psi)
    fine, reconstructed_psi = reconstruct_fine(coarse, details, kernel)

    np.testing.assert_allclose(extracted_coarse, coarse)
    np.testing.assert_allclose(extracted_details, details)
    np.testing.assert_allclose(reconstructed_psi, psi)
    np.testing.assert_allclose(apply_kernel(fine, kernel), psi)
    np.testing.assert_allclose(
        reblocking_error(fine, coarse, kernel), 0.0, atol=1.0e-12
    )


def test_affine_flow_inverse_logdet_checkpoint_and_seed() -> None:
    torch.manual_seed(103)
    config = AffineFlowConfig(lattice_size=4, coupling_layers=2, hidden_channels=8)
    model = ConditionalAffineFlow(config)
    coarse = torch.randn(5, 4, 4)
    latent = torch.randn(5, 3, 4, 4)
    details, forward_logdet = model(coarse, latent)
    reconstructed, inverse_logdet = model.inverse(coarse, details)

    torch.testing.assert_close(reconstructed, latent, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    checkpoint = {
        "config": {"model": model.configuration()},
        "model_state": model.state_dict(),
    }
    loaded = model_from_checkpoint(checkpoint)
    torch.testing.assert_close(loaded(coarse, latent)[0], details)

    old_model_config = model.configuration()
    for name in ("model_type", "spline_layers", "spline_bins", "spline_tail_bound"):
        old_model_config.pop(name)
    old_checkpoint = {
        "config": {"model": old_model_config},
        "model_state": model.state_dict(),
    }
    old_loaded = model_from_checkpoint(old_checkpoint)
    torch.testing.assert_close(old_loaded(coarse, latent)[0], details)

    generator_a = torch.Generator().manual_seed(107)
    generator_b = torch.Generator().manual_seed(107)
    torch.testing.assert_close(
        model.sample(coarse, generator=generator_a)[0],
        model.sample(coarse, generator=generator_b)[0],
    )


def _save_ensemble(path, configurations) -> None:
    np.savez_compressed(
        path,
        configurations=configurations,
        size=configurations.shape[-1],
        kappa=0.340301,
        lam=1.0,
        seed=109,
    )


def test_flow_and_upscale_cli_smoke(tmp_path) -> None:
    rng = np.random.default_rng(109)
    fine = rng.normal(scale=0.5, size=(4, 3, 8, 8))
    coarse = rng.normal(scale=0.5, size=(4, 3, 4, 4))
    fine_path = tmp_path / "fine.npz"
    coarse_path = tmp_path / "coarse.npz"
    kernel_path = tmp_path / "kernel.json"
    flow_root = tmp_path / "flow"
    _save_ensemble(fine_path, fine)
    _save_ensemble(coarse_path, coarse)
    kernel_path.write_text(json.dumps({"matrix": identity_kernel().tolist()}))

    main(
        [
            "train",
            "flow",
            "phi4",
            "--pair",
            str(fine_path),
            str(coarse_path),
            "--kernel",
            str(kernel_path),
            "--output",
            str(flow_root),
            "--epochs",
            "2",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "2",
            "--patience",
            "1",
            "--coupling-layers",
            "2",
            "--hidden-channels",
            "8",
            "--device",
            "cpu",
        ]
    )
    flow_output = flow_root / "L4_to_L8_affine"
    assert (flow_output / "checkpoint.pt").stat().st_size > 0
    flow_metrics = json.loads((flow_output / "metrics.json").read_text())
    saved_checkpoint = torch.load(
        flow_output / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert saved_checkpoint["format_version"] == 2
    assert saved_checkpoint["model_type"] == "affine"
    assert flow_metrics["best_validation_importance_ess_ratio"] > 0.0

    mh_output = tmp_path / "mh"
    main(
        [
            "upscale",
            "phi4",
            "--checkpoint",
            str(flow_output / "checkpoint.pt"),
            "--coarse",
            str(coarse_path),
            "--native-fine",
            str(fine_path),
            "--output",
            str(mh_output),
            "--chains",
            "2",
            "--sweeps",
            "1",
            "--calibration-chains",
            "2",
            "--calibration-sweeps",
            "1",
            "--batch-size",
            "2",
            "--save-sweeps",
            "0,1",
            "--device",
            "cpu",
        ]
    )
    mh_metrics = json.loads((mh_output / "metrics.json").read_text())
    assert mh_metrics["run_config"]["correction"] == "coordinate"
    assert mh_metrics["initial"]["one_shot_has_time_series_ess"] is False
    assert (mh_output / "observable_history.npz").stat().st_size > 0
    assert (mh_output / "checkpoints/checkpoint_sweep_0001.npz").stat().st_size > 0



def test_rational_quadratic_spline_and_residual_flow_round_trip() -> None:
    torch.manual_seed(81)
    values = torch.linspace(-5.0, 5.0, 41).reshape(1, 41)
    parameters = torch.randn(1, 41, 3 * 6 - 1)
    transformed, logdet = rational_quadratic_spline(
        values, parameters, bins=6, tail_bound=4.0
    )
    reconstructed, inverse_logdet = rational_quadratic_spline(
        transformed, parameters, bins=6, tail_bound=4.0, inverse=True
    )
    torch.testing.assert_close(reconstructed, values, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        logdet + inverse_logdet, torch.zeros_like(logdet), atol=3.0e-5, rtol=3.0e-5
    )

    config = AffineFlowConfig(
        lattice_size=4,
        coupling_layers=2,
        hidden_channels=4,
        model_type="residual-spline",
        spline_layers=2,
        spline_bins=4,
    )
    model = ConditionalAffineFlow(config)
    coarse = torch.randn(2, 4, 4)
    latent = torch.randn(2, 3, 4, 4)
    details, forward_logdet = model(coarse, latent)
    latent_back, backward_logdet = model.inverse(coarse, details)
    torch.testing.assert_close(latent_back, latent, atol=3.0e-5, rtol=3.0e-5)
    torch.testing.assert_close(
        forward_logdet + backward_logdet,
        torch.zeros_like(forward_logdet),
        atol=3.0e-5,
        rtol=3.0e-5,
    )

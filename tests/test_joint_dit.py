import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.joint_dit import JointTrajectoryDiT, NoiseExpertDiT, load_joint_dit, sample_trajectory
from scripts.train_joint_dit import flow_step


def make_model(temporal_patch=1):
    return JointTrajectoryDiT(
        latent_channels=4,
        action_dim=3,
        grid=(4, 4, 4),
        n_history=2,
        n_future=2,
        patch_size=(2, 2, 2),
        temporal_patch=temporal_patch,
        dim=32,
        depth=2,
        heads=4,
        ffn_dim=64,
        freq_dim=32,
    )


def make_batch(batch_size=3):
    return dict(
        z_seq=torch.randn(batch_size, 4, 4, 4, 4, 4),
        actions=torch.randn(batch_size, 4, 3),
        obj_occ=torch.rand(batch_size, 4, 4, 4, 4),
    )


def test_forward_flow_loss_and_backward():
    torch.manual_seed(0)
    model = make_model()
    batch = make_batch()
    tau = torch.tensor([0.2, 0.5, 0.8])
    vz, va = model(
        batch["z_seq"][:, :2],
        batch["actions"][:, :2],
        batch["z_seq"][:, 2:],
        batch["actions"][:, 2:],
        tau,
        tau,
    )
    assert vz.shape == (3, 2, 4, 4, 4, 4)
    assert va.shape == (3, 2, 3)
    metrics = flow_step(model, batch, 2, torch.tensor([0, 1, 2]), tau)
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics["loss"].backward()
    assert model.latent_head.proj.weight.grad is not None
    assert model.action_head.proj.weight.grad is not None


def test_temporal_patch_and_all_sampling_modes():
    torch.manual_seed(1)
    model = make_model(temporal_patch=2)
    batch = make_batch(batch_size=2)
    z_hist, a_hist = batch["z_seq"][:, :2], batch["actions"][:, :2]
    z_future, a_future = batch["z_seq"][:, 2:], batch["actions"][:, 2:]
    for mode in ("joint", "world", "inverse"):
        z, a = sample_trajectory(model, z_hist, a_hist, mode=mode, actions=a_future, latents=z_future, steps=2)
        assert z.shape == z_future.shape
        assert a.shape == a_future.shape
        assert torch.isfinite(z).all() and torch.isfinite(a).all()
        if mode == "world":
            torch.testing.assert_close(a, a_future)
        if mode == "inverse":
            torch.testing.assert_close(z, z_future)


def test_goal_conditioning_and_dual_experts():
    torch.manual_seed(3)
    model = make_model()
    with torch.no_grad():
        model.latent_head.proj.weight.normal_(0, 0.02)
        model.action_head.proj.weight.normal_(0, 0.02)
    batch = make_batch(batch_size=1)
    args = (
        batch["z_seq"][:, :2],
        batch["actions"][:, :2],
        batch["z_seq"][:, 2:],
        batch["actions"][:, 2:],
        torch.tensor([0.4]),
        torch.tensor([0.4]),
    )
    goal = batch["z_seq"][:, -1]
    model.eval()
    with torch.no_grad():
        unconditioned = model(*args)
        masked = model(*args, goal=goal, goal_mask=torch.zeros(1))
        conditioned = model(*args, goal=goal, goal_mask=torch.ones(1))
    torch.testing.assert_close(unconditioned[0], masked[0])
    torch.testing.assert_close(unconditioned[1], masked[1])
    assert not torch.allclose(unconditioned[0], conditioned[0])
    assert not torch.allclose(unconditioned[1], conditioned[1])

    dual = NoiseExpertDiT(model.config, switch_tau=0.5)
    batch = make_batch(batch_size=3)
    tau = torch.tensor([0.2, 0.5, 0.8])
    metrics = flow_step(dual, batch, 2, torch.zeros(3, dtype=torch.long), tau)
    metrics["loss"].backward()
    assert dual.high.latent_head.proj.weight.grad is not None
    assert dual.low.latent_head.proj.weight.grad is not None
    z, a = sample_trajectory(
        dual,
        batch["z_seq"][:, :2],
        batch["actions"][:, :2],
        goal=batch["z_seq"][:, -1],
        steps=3,
    )
    assert torch.isfinite(z).all() and torch.isfinite(a).all()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    single = make_model()
    for name, model in (("single", single), ("dual", NoiseExpertDiT(single.config))):
        path = tmp_path / f"{name}.pth"
        torch.save(dict(model_config=model.config, model_state_dict=model.state_dict()), path)
        loaded, _ = load_joint_dit(path, device="cpu")
        batch = make_batch(batch_size=1)
        args = (
            batch["z_seq"][:, :2],
            batch["actions"][:, :2],
            batch["z_seq"][:, 2:],
            batch["actions"][:, 2:],
            torch.tensor([0.4]),
            torch.tensor([0.4]),
        )
        model.eval()
        with torch.no_grad():
            expected = model(*args)
            actual = loaded(*args)
        torch.testing.assert_close(expected[0], actual[0])
        torch.testing.assert_close(expected[1], actual[1])

from __future__ import annotations

import torch

from src.losses_v2 import gaussian_heatmaps, object_masks_from_rgb, soft_object_segmentation_loss
from src.model_v2 import SemanticLatentRenderer, SemanticStateDynamics, StreamingLatentDynamics, V2RepresentationCodec
from scripts.train_v2_dynamics import decode_normalized, rollout_segment


def test_codec_shapes_and_semantic_projection() -> None:
    codec = V2RepresentationCodec(latent_channels=64, semantic_dim=384)
    frames = torch.rand(2, 3, 128, 128)
    reconstruction, latent = codec(frames)
    assert reconstruction.shape == frames.shape
    assert latent.shape == (2, 64, 16, 16)
    assert codec.project_semantic(latent).shape == (2, 384, 16, 16)


def test_streaming_dynamics_consumes_context_actions_and_backpropagates() -> None:
    model = StreamingLatentDynamics()
    context = torch.randn(2, 4, 64, 16, 16)
    context_actions = torch.randint(0, 6, (2, 3))
    current, hidden = model.prefill(context, context_actions)
    prediction, hidden = model.step(current, torch.tensor([1, 4]), hidden)
    prediction.mean().backward()
    assert prediction.shape == current.shape
    assert hidden[0].shape == (2, 128, 16, 16)
    assert model.action_embedding.weight.grad is not None


def test_masks_and_heatmaps_have_expected_channels() -> None:
    frames = torch.zeros(2, 3, 128, 128)
    frames[:, 2, 20:30, 20:30] = 1.0
    masks = object_masks_from_rgb(frames)
    assert masks.shape == (2, 3, 128, 128)
    heatmaps = gaussian_heatmaps(
        torch.tensor([[20.0, 20.0], [40.0, 40.0]]),
        torch.tensor([[60.0, 60.0], [80.0, 80.0]]),
        torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]] * 2),
    )
    assert heatmaps.shape == (2, 3, 32, 32)
    assert torch.all(heatmaps.amax(dim=(2, 3)) > 0.95)


def test_soft_segmentation_penalizes_missing_object_colors() -> None:
    target = torch.zeros(1, 3, 32, 32)
    target[:, :, 8:16, 8:16] = torch.tensor([0.27, 0.56, 0.98]).view(1, 3, 1, 1)
    matching = soft_object_segmentation_loss(target, target)
    missing = soft_object_segmentation_loss(torch.zeros_like(target), target)
    assert matching < missing


def test_rollout_segment_can_mix_teacher_and_predicted_latents() -> None:
    model = StreamingLatentDynamics()
    context = torch.randn(2, 3, 64, 16, 16)
    current, hidden = model.prefill(context, torch.randint(0, 6, (2, 2)))
    targets = torch.randn(2, 4, 64, 16, 16)
    predictions, final, hidden = rollout_segment(
        model, current, hidden, torch.randint(0, 6, (2, 4)), targets, 0.5
    )
    assert predictions.shape == targets.shape
    assert final.shape == current.shape
    codec = V2RepresentationCodec()
    decoded = decode_normalized(codec, predictions, torch.zeros(1, 64, 1, 1), torch.ones(1, 64, 1, 1))
    assert decoded.shape == (2, 4, 3, 128, 128)


def test_semantic_dynamics_and_renderer_shapes() -> None:
    dynamics = SemanticStateDynamics()
    context = torch.randn(2, 4, 19)
    current, hidden = dynamics.prefill(context, torch.randint(0, 6, (2, 3)))
    prediction, hidden = dynamics.step(current, torch.tensor([1, 4]), hidden)
    renderer = SemanticLatentRenderer()
    latent = renderer(prediction)
    latent.mean().backward()
    assert prediction.shape == (2, 19)
    assert hidden[0].shape == (2, 256)
    assert latent.shape == (2, 64, 16, 16)

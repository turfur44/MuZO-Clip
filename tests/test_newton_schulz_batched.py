from __future__ import annotations

import torch

from muzo_clip.newton_schulz import batched_zeropower_via_newtonschulz5, zeropower_via_newtonschulz5


def test_batched_newton_schulz_matches_per_item_wide() -> None:
    torch.manual_seed(1)
    batch = torch.randn(3, 4, 7)
    out = batched_zeropower_via_newtonschulz5(batch, steps=3)
    expected = torch.stack([zeropower_via_newtonschulz5(item, steps=3) for item in batch])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_batched_newton_schulz_matches_per_item_tall() -> None:
    torch.manual_seed(2)
    batch = torch.randn(2, 7, 4)
    out = batched_zeropower_via_newtonschulz5(batch, steps=3)
    expected = torch.stack([zeropower_via_newtonschulz5(item, steps=3) for item in batch])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_batched_newton_schulz_zero_and_nonfinite_return_zero() -> None:
    batch = torch.randn(3, 4, 4)
    batch[0].zero_()
    batch[1, 0, 0] = float("nan")
    out = batched_zeropower_via_newtonschulz5(batch)
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isfinite(out[2]).all().item()

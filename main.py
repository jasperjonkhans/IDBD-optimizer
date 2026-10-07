import torch
from torch import nn
from idbd_exercise import IDBD


def main():
    lr = 1e-3

    x = torch.tensor([1.0, 3.0, 2.0])
    weights = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    optimizer = IDBD([weights], lr, lr)

    before = weights.detach().clone()

    optimizer.zero_grad()

    output = (weights * x).sum()
    target = torch.zeros_like(output)
    loss = 0.5 * (output - target).square()

    loss.backward()

    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()

    optimizer.step(x=x)

    assert torch.isfinite(weights).all()
    assert not torch.equal(before, weights.detach())

    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient: {weights.grad}")
    print(f"Gewichte vorher: {before}")
    print(f"Gewichte danach: {weights.detach()}")


if __name__ == "__main__":
    main()

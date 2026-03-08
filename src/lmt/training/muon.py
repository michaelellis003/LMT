r"""Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

An optimizer for hidden weight matrices in neural networks. Applies
SGD+momentum followed by orthogonalization via Newton-Schulz iteration,
producing updates that are approximately orthogonal matrices.

The key insight: orthogonal updates preserve the scale of activations
across layers, preventing the gradient explosion/vanishing problem
that plagues deep networks. Muon achieves ~2x computational efficiency
vs AdamW on language model pretraining at scale.

**Which parameters should use Muon?**

- 2D hidden weights (Linear layers): **yes** -- use Muon
- Embeddings, output/LM heads: **no** -- use AdamW
- Biases, gains, norms: **no** -- use AdamW

For a combined optimizer, pair Muon with AdamW in separate param groups.

References:
    Jordan, K. (2024). "Muon: An optimizer for hidden layers."
    (GitHub: KellerJordan/Muon)

    Karpathy, A. (2025). "autoresearch" -- uses Muon+AdamW hybrid
    for single-GPU research iteration.
"""

import torch
from torch import Tensor
from torch.optim import Optimizer


def newton_schulz(g: Tensor, steps: int = 5) -> Tensor:
    r"""Orthogonalize a matrix via Newton-Schulz iteration.

    Computes an approximate orthogonal matrix from ``G`` using a
    quintic Newton-Schulz iteration. The coefficients are chosen to
    maximize the slope at zero, which empirically works well even
    though the iteration doesn't fully converge.

    The result is approximately :math:`U S' V^T` where :math:`S'_{ii}`
    is roughly uniform in [0.5, 1.5], not exactly 1. This does not
    hurt model performance vs exact SVD.

    Runs internally in bfloat16 for efficiency on modern GPUs.

    Args:
        g: Input matrix ``[rows, cols]`` (or batched ``[..., rows, cols]``).
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Approximately orthogonal matrix with same shape as ``g``.
    """
    assert g.ndim >= 2

    # Quintic coefficients maximizing slope at zero
    a, b, c = (3.4445, -4.7750, 2.0315)

    orig_dtype = g.dtype
    x = g.bfloat16()

    # Work with the shorter dimension for efficiency
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT

    # Normalize spectral norm to at most 1
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.mT

    return x.to(orig_dtype)


class Muon(Optimizer):
    r"""Muon optimizer for hidden weight matrices.

    Runs SGD with momentum, then orthogonalizes each 2D parameter's
    update via Newton-Schulz iteration. The result is an update whose
    rows (or columns) are approximately orthonormal.

    .. note::

        Only use Muon for hidden weight matrices (2D parameters).
        Embeddings, output heads, biases, and normalization parameters
        should use standard AdamW.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate in units of spectral norm per update.
            Default 0.02 (Keller Jordan's recommended default).
        momentum: Momentum coefficient. Default 0.95.
        weight_decay: AdamW-style weight decay. Default 0.
        ns_steps: Number of Newton-Schulz iterations. Default 5.
        nesterov: Use Nesterov momentum. Default True.

    Example::

        # Separate param groups for Muon and AdamW
        hidden_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim >= 2 and 'embed' not in n and 'head' not in n
        ]
        other_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim < 2 or 'embed' in n or 'head' in n
        ]

        muon_opt = Muon(hidden_params, lr=0.02)
        adam_opt = AdamW(other_params, lr=3e-4)

        # Step both each iteration
        muon_opt.step()
        adam_opt.step()
    """

    def __init__(
        self,
        params: ...,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        """Initialize Muon optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate (spectral norm units).
            momentum: Momentum coefficient.
            weight_decay: AdamW-style weight decay.
            ns_steps: Newton-Schulz iteration count.
            nesterov: Whether to use Nesterov momentum.
        """
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'ns_steps': ns_steps,
            'nesterov': nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: ... = None) -> torch.Tensor | None:  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: Optional closure for re-evaluating the loss.

        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Momentum update: buf = beta * buf + (1 - beta) * grad
                buf.lerp_(grad, 1 - beta)

                # Nesterov: blend grad toward buf (non-in-place to
                # avoid corrupting p.grad for other consumers)
                update = grad.lerp(buf, beta) if nesterov else buf

                # Reshape 4D (conv) to 2D for orthogonalization
                original_shape = update.shape
                if update.ndim == 4:
                    update = update.view(len(update), -1)
                elif update.ndim < 2:
                    # 1D params shouldn't use Muon, but handle gracefully
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                    continue

                # Newton-Schulz orthogonalization
                update = newton_schulz(update, steps=ns_steps)

                # Aspect ratio scaling: scale by sqrt(max(1, rows/cols))
                update = (
                    update * max(1, update.size(-2) / update.size(-1)) ** 0.5
                )

                # Reshape back if needed
                update = update.reshape(original_shape)

                # AdamW-style weight decay (before update)
                p.mul_(1 - lr * wd)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss

# Normalization

Normalization layers stabilize training by keeping activations in a
well-behaved range. Without them, deep transformers often diverge.

## RMSNorm

From Zhang and Sennrich (2019). Used in LLaMA, Mixtral, Gemma.

**Key idea**: Normalize by the root mean square of activations, without
centering (no mean subtraction). Simpler and faster than LayerNorm:

\[
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
\]

```python
from lmt.layers.normalization import RMSNorm

norm = RMSNorm(d_model=256, eps=1e-6)
```

### Why Not LayerNorm?

LayerNorm computes both mean and variance:

\[
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
\]

RMSNorm drops the mean centering and bias term. Empirically, the
re-centering doesn't help much -- the scaling is what matters. This
saves compute and parameters with no quality loss.

### FP32 Upcast

LMT upcasts to FP32 before computing the norm, then casts back to the
input dtype. This is critical for numerical stability in mixed-precision
training -- without it, the `rsqrt` operation can overflow in FP16/BF16.

```python
def forward(self, x: Tensor) -> Tensor:
    input_dtype = x.dtype
    x = x.float()  # upcast to FP32
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return (x * rms * self.weight).to(input_dtype)  # back to original
```

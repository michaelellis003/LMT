# Feed-Forward Networks

The FFN layer processes each token independently through a nonlinear
transform. It's where most of the model's parameters (and "knowledge")
live.

## SwiGLU

From Shazeer (2020). Used in LLaMA, Mixtral, PaLM.

**Key idea**: A gated linear unit where the gate uses the Swish
(\(\text{SiLU}\)) activation:

\[
\text{SwiGLU}(x) = (\text{Swish}(xW_g) \odot xW_1) W_2
\]

Three weight matrices instead of two, but the gating mechanism
consistently outperforms standard FFN with ReLU or GELU.

```python
from lmt.layers.ffn import SwiGLU

ffn = SwiGLU(d_model=256, hidden_dim=683)
# hidden_dim defaults to int(2/3 * 4 * d_model) if not specified
```

**Why 2/3 scaling?** The third matrix means more parameters per hidden
unit, so the hidden dimension is reduced by 2/3 to keep total parameter
count comparable to a standard 4x FFN.

**No bias**: All three linear layers are bias-free, following modern
practice (LLaMA, PaLM).

---

## Mixture of Experts (MoE)

From Shazeer et al. (2017), refined in Switch Transformer and Mixtral.

**Key idea**: Replace one FFN with N expert FFNs. A learned router
selects the top-k experts per token. Only those experts compute --
the rest are skipped entirely.

```python
from lmt.layers.ffn import MoEFeedForward

moe = MoEFeedForward(
    d_model=256,
    num_experts=8,
    top_k=2,              # each token uses 2 of 8 experts
    num_shared_experts=1,  # always-active expert (optional)
)

out = moe(x)
aux_loss = moe.aux_loss  # load balancing loss
```

### The Router

The `TopKRouter` computes a softmax over expert logits, selects the
top-k, and renormalizes their gates to sum to 1:

\[
g_i = \frac{\exp(h_i)}{\sum_{j \in \text{top-k}} \exp(h_j)}
\]

### Load Balancing

Without regularization, the router tends to collapse -- sending all
tokens to the same 1-2 experts while the rest go unused. The auxiliary
loss encourages uniform expert utilization:

\[
\mathcal{L}_{\text{aux}} = N \sum_{i=1}^{N} f_i \cdot P_i
\]

where \(f_i\) is the fraction of tokens routed to expert \(i\) and
\(P_i\) is the average gate probability for expert \(i\).

### Shared Experts

Mixtral-style MoE can include **shared experts** that process every
token regardless of routing. This ensures a baseline capacity while
the routed experts specialize.

### Sparse Activation

The key benefit: a model with 8 experts and top-k=2 has 8x the
parameters of a dense model but only 2x the compute per token.
This is how Mixtral-8x7B achieves 47B total parameters but runs
at the cost of a 13B model.

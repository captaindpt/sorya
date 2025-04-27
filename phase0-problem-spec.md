# Entangled LLM: Problem Statement, Notation, and Research Questions

## 1. Background & Motivation
Large Language Models (LLMs) achieve remarkable fluency but often contradict themselves or abandon prior commitments across extended interactions. Existing remedies (larger context windows, external retrieval, RL for factuality) *externalise* consistency, rather than making it an **intrinsic property** of the model's own computation.

## 2. Literature Gap
A survey of 11 representative works shows that prior approaches typically provide **either**
* local latent control (VAE-style models, PLATO),
* temporary commitment within an episode (Option-Critic, DIAYN),
* post-hoc reward for correctness (RLFC), **or**
* separate memory buffers (LongMem, Generative Agents).

None combines **(i)** a *sampled, irreversible* latent commitment, **(ii)** an internal mechanism binding future generation to that commitment, **and** **(iii)** a memory of past commitments that biases future ones.

## 3. Formal Problem Statement
Design an LLM architecture that, during an interaction of length $T$, must
1. **Sample** a latent commitment $z^\star \in \mathbb{R}^k$ at a decisive timestep $t_0$.  
2. **Irreversibly condition** all subsequent hidden states $h_t$ ($t\ge t_0$) on $z^\star$.  
3. **Receive reward** for staying coherent with its own $z^\star$ *and* with similar commitments stored from earlier sessions.

Goal: maximise self-consistency and usefulness while minimising additional compute and memory.

## 4. Proposed Solution ("Entangled LLM")
| Component | Role | Minimal Instantiation |
|-----------|------|-----------------------|
| Base LM $p_\theta$ | Generates tokens | Pre-trained transformer (e.g. Llama-2-7B) |
| Posterior $q_\phi(z\mid h)$ | "Superposition of selves" | 2-layer MLP producing $\mu,\sigma$ |
| Commit operator $\mathcal M_{z^\star}$ | Injects $z^\star$ | Additive projection $h \leftarrow h + W_c z^\star$ |
| Memory $M$ | Store past $(z^\star,\text{ctx})$ | FAISS or Chroma DB |
| Rewards | $R_{task}$, $R_{ent}$ | GPT-4o prefs, cosine sim. |

### Key Equations
1. **Sampling**  \[ z^\star \sim q_\phi(z\mid h_{t_0}) \]
2. **Conditioning**  \[ h_t \leftarrow h_t + W_c z^\star, \; t \ge t_0 \]
3. **Entanglement reward**  \[ R_{ent}=\lambda \; \tfrac1K \sum_{i=1}^K \cos(z^\star, z_i^\star) \]
4. **Objective**
\[
\mathcal L = \mathbb E_{z^\star}\bigl[-\log p_\theta(x_{t_0:T}\mid z^\star)\bigr]
\;+\;\beta \operatorname{KL}\bigl(q_\phi \| p\bigr)
\;-\;\gamma R_{task}\;-\;\eta R_{ent}.
\]

## 5. Notation Quick-Reference
| Symbol | Meaning |
|--------|---------|
| $h_t \in \mathbb{R}^d$ | LM hidden state after token $t$ |
| $z, z^\star \in \mathbb{R}^k$ | Latent policy code; $z^\star$ is the sampled commitment |
| $q_\phi$, $p(z)$ | Variational posterior, prior (usually $\mathcal N(0,I)$) |
| $W_c \in \mathbb{R}^{d\times k}$ | Projection from latent to hidden space |
| $M$ | Memory of past commitments |
| $R_{task}$ | External task reward / human preference |
| $R_{ent}$ | Self-consistency (entanglement) reward |
| $\beta,\gamma,\eta$ | Loss hyper-parameters |

## 6. Research Questions / Hypotheses
1. **Commitment Effectiveness:** Does adding a fixed $z^\star$ improve intra-dialogue self-consistency versus a baseline LLM?
2. **Learnability & Trade-off:** Can $q_\phi$ and $W_c$ be trained via QLoRA + RLHF to balance consistency and utility with <5 % hit to perplexity?
3. **Entanglement Transfer:** Does memory-based bias toward previous $z^\star$ values double cross-session loyalty compared with baseline?

**Hypothesis:** An irreversible latent commitment plus entanglement memory yields >15 pp human-preference gain on "Which assistant contradicts itself less?" while adding <1 % parameters and <2 % inference FLOPs.

---
*Phase 0 deliverable prepared — ready for reviewer feedback or direct move to Phase 1 prototype.* 
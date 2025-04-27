Lean-Budget Road-map to an "Entangled LLM" Paper

(everything chosen so a solo researcher with < $2 000 in cloud credits or a single 24 GB GPU can get publishable evidence)

Phase	Goal	What you actually build / test	Cost ceiling	Proof you'll collect	Deliverable section
0. Survey & spec (1 week)	Nail the related-work gap and pin down notation.	• 2-page gap table contrasting latent-commitment vs ordinary RLHF, VAE-style latent planners, etc.	$0	Literature matrix + clear problem statement	Intro & Related Work
1. "Latent commitment" without training (2 days)	Show that freezing a random latent already forces coherence.	• Pick Llama-2-7B-chat/Q5 in llama.cpp.  • At turn N, sample one hidden-state projection z^\star=W_c h_N and add it as a fixed prefix token ID for the rest of dialogue.	$0 (CPU)	• Measure self-similarity of embeddings in later turns vs baseline.  • Quick human eval ("does the assistant contradict itself?").	Exploratory Evidence
1b. Sanity-tune commitment strength (1 day)	Find injection λ/layers that preserve fluency.	• Add `λ·Wc_z_star` in `llama-model.cpp` with λ ∈ {1e-3..1e-1}. • Test injecting in last N layers only. 	$0 (CPU)	• Perplexity on short text. • Qualitative eval on 3-4 prompts. Find settings that don't break generation.	Calibration
2. Tiny latent-head fine-tune with QLoRA (1-2 weeks)	Train the variational head & projection matrix on 100 M tokens.	• Clone trl repo; attach 64-d latent head.  • QLoRA: fits in 1× A10G (24 GB) → $0.40 hr on RunPod.  • 2 epochs = 20 GPU-hours ≈ $8.	≤ $50	• Drop in perplexity.  • Automatic self-agreement metric ↑ vs baseline.	Method & Supervised Results
3. Cheap RLHF surrogate (1 week)	Encourage the model to stay "loyal" to its chosen latent.	• Use GPT-4o (API) to judge pairs: "Which answer stays more consistent with its own earlier claim?"  • 5 k comparisons → $150.  • PPO on 8 A100 spot for 10 h → $100.	≤ $300	• Win-rate on consistency preferences.  • Plot reward vs divergence from latent.	RLHF Section
4. Retrieval-memory prototype (3 days)	Show entanglement carries across sessions.	• Store z^\star vectors + dialogue ID in Chroma / FAISS-CPU.  • At new session, top-k retrieve and bias prior of q_\phi.	$0	• A/B: probability of repeating earlier opinions ↑.	Memory & Entanglement
5. Ablations & scaling law fit (1 week)	Understand cost/benefit curves.	• Sweep latent dim {0,16,32,64}.  • Sweep KL temperature τ.  • Run on 3-B Mistral and 13-B Llama-2 with PEFT.	≤ $200	• Two log-log plots (dim ↔ consistency, τ ↔ helpfulness).	Analysis
6. Write-up & open artefacts (2 weeks)	Clean repo + reproducibility kit.	• Dockerfile, Colab, w-and-b dashboards.  • 8-page paper in ACL style.	$0	• Public code + weights (LoRA adapters).	Paper & Appendix

Running compute tally

Item	$	Notes
Tiny QLoRA pre-fine-tune	8	20 h × $0.40
PPO consistency RL	100	spot A100
GPT-4o preference labels	150	5 k judgements
Ablations	200	misc.
Total hard spend	≈ $460	

(Keep a 30 % buffer → $600 all-in.)

⸻

Practical Tips to Stay Under Budget
	1.	Start on consumer hardware: phases 0–2 run on a single 16 GB RTX-4060Ti with QLoRA if you quantise to 4-bit and cut batch = 1.
	2.	Spot rentals only: Use Lambda or RunPod spot queues (≈ $0.90 hr) for PPO; checkpoint every 30 min.
	3.	Scale tokens, not parameters: The latent-head adds < 0.1 % params; the supervised corpus size drives signal.
	4.	Synthetic preference labels first: Use your own heuristic ("contradicts earlier claim? yes/no") to pre-train the reward model before paying for GPT-4o.
	5.	Publish adapters, not full weights: keeps legal and bandwidth costs low.

⸻

What counts as "proof" before the full grand model?
	•	Statistically significant rise in self-consistency metrics (Entailment-based or embedding cosine).
	•	Human preference delta (> +15 pp) on "Which assistant contradicts itself less?"
	•	Cross-session loyalty: given a second chat a day later, does the model remember its stance?  Memory-augmented version should ≥ 2× baseline.
	•	Cost/performance curve showing latent-head delivers +X coherence at 1 % extra FLOPs.

Each of those is a stand-alone figure for your paper; together they triangulate the thesis without needing a $7 k full run.

⸻

Suggested section skeleton for the paper
	1.	Introduction – Vulnerability & commitment as missing inductive bias.
	2.	Related Work – VAEs, policy-switching LLMs, memory-augmented transformers.
	3.	Method – Latent posterior, commitment operator, entanglement reward.
	4.	Experiments
	•	4.1 Zero-train commitment (Phase 1)
	•	4.2 Supervised QLoRA (Phase 2)
	•	4.3 RLHF consistency (Phase 3)
	•	4.4 Memory transfer (Phase 4)
	•	4.5 Ablations & scaling (Phase 5)
	5.	Discussion – Cost analysis, ethical risks, future scaling path.
	6.	Conclusion – Commitment as the pathway to "chosen identity" in LLMs.
	7.	Appendix – Hyper-params, exact dollar budget, open-source links.

Follow this roadmap and, even solo, you'll have incremental proofs, tight cost control, and a clear narrative for reviewers that this isn't just philosophy—it's measurable engineering.
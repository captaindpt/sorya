## 2024-07-27: Phase 1 - Zero-Shot Commitment Setup

1.  **Environment Reset:** Reverted local changes in the `llama.cpp` submodule to ensure a clean state (`git checkout -- .` within `llama.cpp`).
2.  **Model Acquisition:** Downloaded the required `Llama-2-7B-Chat-GGUF` model (`llama-2-7b-chat.Q5_K_M.gguf`) necessary for Phase 1 experiments.
3.  **Code Analysis:**
    *   Identified `llama_context::decode` as the main entry point for token generation.
    *   Traced graph building to `llama_context::graph_build`, which delegates to `llama_model::build_graph`.
    *   Located the architecture-specific graph building logic for Llama models within the `llm_build_llama` constructor in `llama-model.cpp`.
    *   Pinpointed the injection site for the commitment vector (`z_star`) addition within the transformer layer loop (specifically after the second residual connection, around line 4605 in `llama-model.cpp`).
4.  **Implementation (Commitment Mechanism):**
    *   Added static variables (`z_star_static`, `Wc_z_star_static`, flags) to `llama-model.cpp` to hold the commitment state.
    *   Implemented `initialize_commitment` function to sample `z_star` (from N(0, 1/k)) and compute the projected vector `Wc_z_star` (using Xavier init for implicit `W_c`) exactly once using `std::call_once`.
    *   Created a `ggml_tensor` (`Wc_z_star_tensor`) within the `llm_build_llama` constructor, allocating its buffer on the CPU backend using `ggml_backend_alloc_ctx_tensors` and copying the static data via `memcpy`.
    *   Injected `ggml_add(ctx0, cur, Wc_z_star_tensor)` into the layer loop after the second residual connection to add the commitment to the hidden state.
    *   Iteratively debugged compilation errors related to buffer types (`ggml_backend_buffer_t` vs `ggml_backend_buffer_type_t`), tensor allocation (`ggml_allocr_alloc` vs `ggml_backend_alloc_ctx_tensors`), and `ggml` API usage.
5.  **Initial Test:** Successfully compiled the modified code using CMake and ran `./bin/llama-cli`. Output confirms:
    *   Commitment initialization logs appear once.
    *   No allocation errors or crashes occurred.
    *   Model generates output, indicating the modified code path is executed.

**Status:** Phase 1 core implementation complete and minimally tested. Ready for coherence evaluation. 

6.  **Phase 1 Evaluation (Consistency Test):**
    *   **Baseline:** Ran original `llama-cli` with a prompt testing self-consistency (`What is your favorite language?... A little later... what did you say your favorite language was again?`). Baseline model contradicted its initial statement.
    *   **Modified:** Ran the modified `llama-cli` (with fixed random commitment vector `Wc_z_star` added in each layer) on the same prompt.
    *   **Result:** The modified model produced incoherent, nonsensical output after the second question. Adding a fixed *random* vector severely degraded generation quality.

**Conclusion:** The Phase 1 mechanism successfully injects the commitment, but a *random* commitment is too disruptive. This motivates Phase 2/3: training the commitment vector (`z*`) and projection (`W_c`).

7.  **Phase 1b Calibration (Tunable Commitment):**
    *   Implemented tunable `lambda_commit` scaling factor and `commit_layer_start`/`commit_layer_end` parameters in `llama-model.cpp`.
    *   **Experiment 1 (λ=1.0, all layers):** Replicated incoherent output from Phase 1 Evaluation.
    *   **Experiment 2 (λ=0.1, all layers):** Output remained incoherent.
    *   **Experiment 3 (λ=0.01, all layers):** Output became coherent but repetitive/stuck.
    *   **Experiment 4 (λ=0.01, last 8 layers):** Output became incoherent again.
    *   **Conclusion:** Simple scaling or layer masking of a fixed *random* commitment vector is insufficient. The vector likely needs to be contextually generated and integrated via training.

**Next Step:** Proceed to Phase 2 - setting up QLoRA fine-tuning for the variational head and projection matrix.

## 2024-07-28: Phase 2 - QLoRA Training Setup

1.  **Training Script (`train_entangled.py`):**
    *   Created a Python script using `transformers`, `peft`, `bitsandbytes`, and `torch`.
    *   Configured the script to load a base model (e.g., `meta-llama/Llama-2-7b-chat-hf`).
    *   Implemented QLoRA configuration (`BitsAndBytesConfig`, `LoraConfig`).
    *   Set up the model for training using `get_peft_model`.
    *   Included basic training arguments (`TrainingArguments`) and initialized the `Trainer`.
    *   Added a placeholder dataset (needs replacement with actual data).
2.  **Dependencies (`requirements.txt`):** Created a requirements file listing necessary packages (`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`).
3.  **Local Testing & Environment Issues:**
    *   Attempted to run the script locally on macOS.
    *   Encountered errors with `bitsandbytes` as the pre-compiled wheel lacked CUDA support (macOS doesn't have NVIDIA GPUs).
    *   Confirmed that `bitsandbytes` requires a Linux environment with CUDA installed for GPU acceleration, which is essential for efficient QLoRA training.
4.  **Cloud Execution Plan:**
    *   Decided to run the training on a cloud GPU platform (e.g., Vast.ai) to leverage CUDA.
    *   Selected the **`PyTorch (Vast)`** template as the most suitable starting point, providing PyTorch and CUDA drivers.
    *   Outlined the setup process on the cloud instance:
        *   Clone the repository.
        *   Create and activate a Python virtual environment.
        *   Install dependencies using `pip install -r phase2_training/requirements.txt`.
        *   Run the training script `python phase2_training/train_entangled.py`.

**Status:** Phase 2 training script and environment requirements are defined. Local execution is blocked by hardware limitations. Cloud environment strategy is determined.

**Next Step:** Provision a cloud GPU instance using the `PyTorch (Vast)` template, set up the environment, and execute the `train_entangled.py` script to begin QLoRA fine-tuning. 
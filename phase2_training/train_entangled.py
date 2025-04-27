import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    LlamaForCausalLM,
    AutoConfig,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast

# TODO: Define EntangledModel class inheriting from AutoModelForCausalLM
# TODO: Add variational head (q_phi)
# TODO: Add commitment projection (W_c) using LoRA or similar
# TODO: Modify forward pass to sample z_star and add commitment
# TODO: Define training loop / use SFTTrainer with custom logic

# --- Define Custom Output Dataclass ---
@dataclass
class EntangledCausalLMOutput(CausalLMOutputWithPast):
    """
    Output class for Entangled Llama model, inheriting from CausalLMOutputWithPast
    and adding entanglement-specific tensors.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # --- Entanglement specific outputs ---
    z_star: Optional[torch.FloatTensor] = None
    mu: Optional[torch.FloatTensor] = None
    logvar: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None


# --- Modified EntangledLlama Class ---
class EntangledLlama(LlamaForCausalLM):
    def __init__(self, config, commitment_dim=64):
        super().__init__(config)
        self.commitment_dim = commitment_dim
        hidden_size = config.hidden_size

        # Define the Variational Head (q_phi)
        self.variational_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2 * self.commitment_dim)
        )

        # --- Add Commitment Projection Layer (W_c) ---
        # This layer projects the commitment vector z* to the model's hidden dimension
        self.Wc_projection = nn.Linear(self.commitment_dim, hidden_size, bias=False)

    # TODO: Add commitment projection (W_c) using LoRA or similar (affects forward pass)

    # Modified forward signature and return type
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        calculate_entanglement: bool = False, # Flag to enable z* calculation
        commitment_lambda: float = 1.0, # Strength of commitment addition
        kl_beta: float = 0.1, # Hyperparameter for KL divergence weight (example value)
    ) -> Union[Tuple, EntangledCausalLMOutput]: # Return either tuple or custom output

        # Ensure hidden states are output if calculating entanglement
        # Also force return_dict for easier access to named outputs
        original_output_hidden_states = output_hidden_states
        if calculate_entanglement or (commitment_lambda > 0):
            output_hidden_states = True
            if commitment_lambda > 0 or calculate_entanglement: # Need dict if calculating KL loss too
                 return_dict = True

        # --- Call Base Model Forward Pass ---
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, # Use potentially modified flag
            return_dict=return_dict, # Use potentially modified flag
        )

        # --- Calculate Entanglement Variables (if requested) ---
        z_star, mu, logvar = None, None, None
        projected_commitment = None # To store Wc * z_star
        sequence_lengths = None # Store sequence lengths for commitment addition
        kl_loss = None # Initialize KL loss
        if calculate_entanglement and outputs.hidden_states is not None:
            # 1. Get hidden state at decisive timestep (use last token of input sequence)
            if input_ids is not None:
                # Determine sequence lengths (index of last non-padding token)
                if attention_mask is not None:
                    # Use attention mask to find the length reliably
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                else:
                    # Fallback: assume no padding if no mask provided
                    # Warning: This might be inaccurate if padding exists without mask
                    sequence_lengths = (input_ids.shape[1] - 1)
                    if torch.any(input_ids == 0): # Basic check for padding token
                         logging.warning("Input IDs seem to contain padding, but no attention mask provided. Assuming full sequence length.")

                # Get hidden state from the last layer
                last_hidden_state = outputs.hidden_states[-1] # Shape: (batch_size, seq_len, hidden_size)

                # Gather hidden states at the end of each sequence in the batch
                batch_size = last_hidden_state.shape[0]
                # Create indices tensor for gathering: (batch_size, 1, hidden_size)
                # Clamp indices to be within valid range just in case
                indices = sequence_lengths.clamp(0, last_hidden_state.shape[1] - 1).view(batch_size, 1, 1).expand(-1, -1, last_hidden_state.shape[-1])
                decisive_h = last_hidden_state.gather(1, indices).squeeze(1) # Shape: (batch_size, hidden_size)

                # 2. Pass through variational head
                mu_logvar = self.variational_head(decisive_h) # Shape: (batch_size, 2 * commitment_dim)
                mu = mu_logvar[:, :self.commitment_dim]
                logvar = mu_logvar[:, self.commitment_dim:]

                # 3. Reparameterization Trick to sample z_star
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_star = mu + eps * std

                # --- Project z_star using W_c ---
                projected_commitment = self.Wc_projection(z_star) # Shape: (batch_size, hidden_size)

                # --- Calculate KL Divergence ---
                # KL divergence between q_phi(z|h) and p(z) = N(0, I)
                # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                # Here, logvar = log(sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # Sum over commitment dim
                kl_loss = kl_loss.mean() # Average over batch

            else:
                logging.warning("Cannot calculate entanglement: input_ids not provided.")


        # --- Commitment Addition Logic ---
        final_logits = outputs.logits # Start with original logits
        if commitment_lambda > 0 and projected_commitment is not None and outputs.hidden_states is not None and sequence_lengths is not None:
            last_hidden_state = outputs.hidden_states[-1] # Shape: (batch_size, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = last_hidden_state.shape

            # Create a mask for positions *after* the decisive turn
            # Mask should be 1 for positions > sequence_length, 0 otherwise
            pos_indices = torch.arange(seq_len, device=last_hidden_state.device)[None, :] # Shape: (1, seq_len)
            # sequence_lengths[:, None] gives shape (batch_size, 1)
            commitment_mask = (pos_indices > sequence_lengths[:, None]).float() # Shape: (batch_size, seq_len)
            commitment_mask = commitment_mask.unsqueeze(-1) # Shape: (batch_size, seq_len, 1)

            # Add scaled projected commitment to the last hidden state, applying the mask
            # projected_commitment shape: (batch_size, hidden_size) -> unsqueeze to (batch_size, 1, hidden_size) for broadcasting
            modified_last_hidden_state = last_hidden_state + (
                projected_commitment.unsqueeze(1) * commitment_lambda * commitment_mask
            )

            # Recalculate logits using the modified hidden state
            final_logits = self.lm_head(modified_last_hidden_state)


        # --- Handle Return Value ---
        if not return_dict:
             # If tuple was requested originally, try to reconstruct, otherwise return base output
             # This gets complicated, strongly recommend using return_dict=True when using entanglement
             logging.warning("Returning tuple output for EntangledLlama is complex; dict output recommended.")
             # Simplest fallback: return modified logits if calculated, else base logits/outputs
             if commitment_lambda > 0 and projected_commitment is not None:
                 # This won't match the standard tuple format perfectly
                 return (final_logits,) + outputs[1:] # Replace original logits
             else:
                 return outputs # Return original tuple/dict from base

        # If we calculated entanglement, return the custom dataclass
        if calculate_entanglement:
            return EntangledCausalLMOutput(
                loss=outputs.loss,
                logits=final_logits,
                past_key_values=outputs.past_key_values,
                # Only include hidden_states/attentions if originally requested OR calculated for entanglement
                hidden_states=outputs.hidden_states if (original_output_hidden_states or calculate_entanglement or (commitment_lambda > 0)) else None,
                attentions=outputs.attentions,
                z_star=z_star,
                mu=mu,
                logvar=logvar,
                kl_loss=kl_loss,
            )
        else:
            # If dict was requested, but entanglement wasn't calculated, return standard dict output
            return outputs


# --- PEFT Configuration ---
# Example LoRA configuration for general fine-tuning
# We target q_proj and v_proj, common for QLoRA
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=16, # Alpha scaling factor
    target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# --- Custom Trainer for Combined Loss ---
class EntangledTrainer(Trainer):
    def __init__(self, *args, kl_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = kl_beta # Store the KL divergence weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # Ensure the model calculates entanglement variables and returns the custom dict
        # Pass relevant args from init or state if needed inside the model's forward
        outputs = model(**inputs, calculate_entanglement=True, kl_beta=self.kl_beta)

        # Extract the standard LM loss (calculated by the base model on original/modified logits)
        lm_loss = outputs.loss

        # Extract the KL loss calculated in our custom forward pass
        kl_loss = outputs.kl_loss

        # Combine the losses
        # Ensure kl_loss is valid before adding
        if lm_loss is not None and kl_loss is not None:
            total_loss = lm_loss + self.kl_beta * kl_loss
        elif lm_loss is not None:
            total_loss = lm_loss # Fallback to only LM loss if KL wasn't calculated
            logging.warning("KL loss not found in model outputs, using only LM loss.")
        else:
            # This should generally not happen if labels are provided
            logging.error("LM loss not found in model outputs.")
            total_loss = None # Or handle error appropriately

        return (total_loss, outputs) if return_outputs else total_loss


if __name__ == "__main__":
    # --- Configuration ---
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # Use a smaller subset for faster testing initially
    # dataset_name = "mlabonne/guanaco-llama2-1k"
    dataset_name = "mlabonne/guanaco-llama2-1k" # Full dataset for example
    commitment_dim = 64
    output_dir = "./results_phase2"
    kl_beta_value = 0.1 # Example value for KL loss weight

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token for batched training
    tokenizer.padding_side = "right" # Llama requires right padding

    # --- Load Dataset ---
    print(f"Loading dataset: {dataset_name}")
    # Load a portion for testing, use full later
    # dataset = load_dataset(dataset_name, split="train[:100]") # Load first 100 samples
    dataset = load_dataset(dataset_name, split="train") # Load full training set
    # TODO: Add preprocessing if needed (e.g., formatting, tokenization)
    # TRL's SFTTrainer often handles formatting, but check requirements
    print(f"Dataset loaded. Size: {len(dataset)}")


    # --- Model Loading (QLoRA Example) ---
    print(f"Loading base model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    # Instantiate our custom model (example for QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for A100/H100
        bnb_4bit_use_double_quant=False,
    )
    entangled_model = EntangledLlama.from_pretrained(
        model_name,
        config=config,
        commitment_dim=commitment_dim,
        quantization_config=bnb_config, # Apply QLoRA quantization - RE-ENABLED
        device_map="auto", # Automatically distribute model layers
    )
    entangled_model.config.use_cache = False # Required for gradient checkpointing/training
    # Set pretraining_tp to 1 to avoid issues with device mapping / gradient checkpointing
    entangled_model.config.pretraining_tp = 1

    # --- PEFT Setup ---
    # Ensure custom layers are trainable
    entangled_model.variational_head.requires_grad_(True)
    entangled_model.Wc_projection.requires_grad_(True)
    # Apply LoRA adapters
    peft_model = get_peft_model(entangled_model, lora_config)
    peft_model.print_trainable_parameters()


    # --- Training Arguments ---
    print("Setting up Training Arguments...")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,              # Start with 1 epoch for testing
        per_device_train_batch_size=4,   # Adjust based on GPU memory
        gradient_accumulation_steps=1,   # Adjust based on GPU memory
        optim="paged_adamw_32bit",       # Optimizer suitable for QLoRA
        save_steps=50,                   # Save checkpoints periodically
        logging_steps=10,                # Log metrics periodically
        learning_rate=2e-4,              # Common learning rate for QLoRA
        weight_decay=0.001,
        fp16=False,                      # Use bf16 if available (A100/H100), else fp16
        bf16=True,                       # Set to True if using Ampere GPUs or newer
        max_grad_norm=0.3,
        max_steps=10,                    # MODIFIED: Run only 10 steps for quick test
        warmup_ratio=0.03,
        group_by_length=True,            # Faster training by grouping similar length sequences
        lr_scheduler_type="constant",    # Or "cosine"
        report_to="tensorboard",         # Or "wandb"
        # Additional arguments...
    )

    # --- Initialize Trainer ---
    print("Initializing Custom Entangled Trainer...")
    trainer = EntangledTrainer( # Use the custom trainer
        model=peft_model,
        train_dataset=dataset,
        # peft_config=lora_config, # Not needed for standard Trainer
        args=training_arguments,
        tokenizer=tokenizer,
        # Data collator might be needed depending on dataset format / padding
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        kl_beta=kl_beta_value, # Pass beta hyperparameter
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.train()

    # --- Save Model ---
    print("Saving final model...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Phase 2 Training Script Setup Complete")
    pass

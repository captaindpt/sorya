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


# --- Modified EntangledLlama Class (Becomes mostly standard Llama) ---
# We remove the custom head and projection from here. The wrapper will handle them.
class BaseLlamaForEntanglement(LlamaForCausalLM):
    # No custom __init__ needed anymore, inherits directly
    # Keep the modified forward signature ONLY to potentially force output_hidden_states
    # if the wrapper needs them, but remove all entanglement logic.
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
        # Remove entanglement-specific args from base model forward
    ) -> Union[Tuple, CausalLMOutputWithPast]: # Return standard output

        # --- Call Base Model Forward Pass (Standard Llama) ---
        # Force output_hidden_states=True if needed externally?
        # Or just let the caller request it.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, # Pass through request
            return_dict=return_dict, # Pass through request
        )
        return outputs

# --- EntangledModel Wrapper (Full Forward Pass Restored) --- 
class EntangledModel(nn.Module):
    def __init__(self, base_model, commitment_dim=64):
        super().__init__()
        self.base_model = base_model
        self.commitment_dim = commitment_dim
        # Use the config from the base model to get hidden_size
        hidden_size = base_model.config.hidden_size 

        # Define the Variational Head (q_phi)
        self.variational_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2 * self.commitment_dim)
        )

        # --- Add Commitment Projection Layer (W_c) ---
        self.Wc_projection = nn.Linear(self.commitment_dim, hidden_size, bias=False)

        # Ensure these custom layers use the same compute dtype as the base model if possible
        compute_dtype = getattr(base_model.config, "torch_dtype", torch.float32)
        if compute_dtype == torch.float16 or compute_dtype == torch.bfloat16:
            self.variational_head.to(dtype=compute_dtype)
            self.Wc_projection.to(dtype=compute_dtype)

    # --- FULL FORWARD PASS --- 
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Add other necessary args passed by Trainer (position_ids etc.)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        calculate_entanglement: bool = True, 
        commitment_lambda: float = 1.0, 
    ) -> EntangledCausalLMOutput: 
        # --- Remove most verbose prints --- 
        # print(f">>> [Step Start] ...", flush=True)

        # print(">>> [Step] Calling base_model forward...", flush=True)
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None, 
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, 
            return_dict=True, 
        )
        # print(">>> [Step] base_model forward complete.", flush=True)

        # --- Calculate Entanglement Variables --- 
        z_star, mu, logvar = None, None, None
        projected_commitment = None
        sequence_lengths = None
        kl_loss = None
        # --- Initialize decisive_h --- 
        decisive_h = None 
        # --- Calculate decisive_h only if input_ids are present --- 
        if input_ids is not None:
            # print(">>> [DEBUG] Calculating decisive_h from input_ids...")
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
            else:
                sequence_lengths = torch.tensor([input_ids.shape[1] - 1] * input_ids.shape[0], device=input_ids.device)
                if torch.any(input_ids == 0):
                     logging.warning("Input IDs seem to contain padding... Ensure attention_mask is correct.")
            
            last_hidden_state = base_outputs.hidden_states[-1]
            batch_size = last_hidden_state.shape[0]
            indices = sequence_lengths.clamp(0, last_hidden_state.shape[1] - 1).view(batch_size, 1, 1).expand(-1, -1, last_hidden_state.shape[-1])
            indices = indices.to(last_hidden_state.device)
            decisive_h = last_hidden_state.gather(1, indices).squeeze(1)
            # print(f">>> [DEBUG] decisive_h calculated, shape: {decisive_h.shape}")
        # --- End of block for calculating decisive_h ---

        # --- Calculate variational head outputs ONLY if decisive_h exists --- 
        if decisive_h is not None:
            # print(">>> [DEBUG] Calculating variational head outputs...")
            self.variational_head.to(decisive_h.device)
            mu_logvar = self.variational_head(decisive_h)
            # print(f">>> [DEBUG KL Calc] type(mu_logvar)={type(mu_logvar)}, mu_logvar.shape={mu_logvar.shape if torch.is_tensor(mu_logvar) else 'N/A'}", flush=True)
            mu = mu_logvar[:, :self.commitment_dim]
            logvar = mu_logvar[:, self.commitment_dim:]
            # print(f">>> [DEBUG KL Calc] type(mu)={type(mu)}, type(logvar)={type(logvar)}", flush=True)
            if torch.is_tensor(logvar):
                print(f">>> [DEBUG KL Calc] logvar.shape={logvar.shape}", flush=True)
            else:
                print(f">>> [DEBUG KL Calc] logvar is NOT a tensor (or is None)! Value: {logvar}", flush=True)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_star = mu + eps * std

            self.Wc_projection.to(z_star.device)
            projected_commitment = self.Wc_projection(z_star)

            # --- Calculate KL Divergence (only if mu/logvar calculated) --- 
            # print(">>> [DEBUG] Calculating KL loss...")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss.mean()
            # print(f">>> [Step] Entanglement variables calculated. KL Loss: {kl_loss.item():.4f}", flush=True)
        else:
            # This case occurs if input_ids were None (e.g., inputs_embeds used)
            # or if base_outputs.hidden_states was None
            print(">>> [DEBUG] Skipping variational head calculation (decisive_h is None).", flush=True)

        # --- Commitment Addition Logic ---
        final_logits = base_outputs.logits
        # --- ADD SHAPE PRINT --- 
        print(f">>> [DEBUG Shape Check] Before commitment: base_outputs.logits shape = {base_outputs.logits.shape}", flush=True)
        # --- 
        if commitment_lambda > 0 and projected_commitment is not None and base_outputs.hidden_states is not None and sequence_lengths is not None:
            # print(">>> [Step] Applying commitment vector...", flush=True)
            last_hidden_state = base_outputs.hidden_states[-1]
            batch_size, seq_len, hidden_size = last_hidden_state.shape
            pos_indices = torch.arange(seq_len, device=last_hidden_state.device)[None, :]
            commitment_mask = (pos_indices > sequence_lengths[:, None]).float()
            commitment_mask = commitment_mask.unsqueeze(-1)
            
            projected_commitment = projected_commitment.to(last_hidden_state.device)
            # --- Corrected multiplication --- 
            modified_last_hidden_state = last_hidden_state + (
                projected_commitment.unsqueeze(1) * commitment_lambda * commitment_mask # Use commitment_mask
            )
            # --- 
            # print(f">>> [DEBUG Shape Check] Before lm_head: modified_last_hidden_state shape = {modified_last_hidden_state.shape}", flush=True)
            
            final_logits = self.base_model.lm_head(modified_last_hidden_state)
            # print(f">>> [DEBUG Shape Check] After lm_head: final_logits shape = {final_logits.shape}", flush=True)
        # elif commitment_lambda > 0:
        #      print(">>> [Step] Skipping commitment addition (conditions not met).", flush=True)

        # --- Calculate Loss (if labels provided) ---
        loss = None
        if labels is not None:
            print(">>> [Step] Calculating loss...", flush=True)
            # --- ADD SHAPE PRINT for final_logits --- 
            print(f">>> [DEBUG Loss Calc] Entering loss calc: final_logits shape={final_logits.shape}", flush=True)
            # --- END SHAPE PRINT --- 
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # --- ADD SHAPE PRINTS --- 
            print(f">>> [DEBUG Loss Calc] Before view: shift_logits shape={shift_logits.shape}, shift_labels shape={shift_labels.shape}", flush=True)
            # --- END SHAPE PRINTS --- 
            # --- ENSURE loss_fct is defined HERE --- 
            loss_fct = nn.CrossEntropyLoss() # Ensure definition is present
            # --- 
            shift_logits = shift_logits.view(-1, self.base_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # --- ADD SHAPE PRINTS --- 
            print(f">>> [DEBUG Loss Calc] After view: shift_logits shape={shift_logits.shape}, shift_labels shape={shift_labels.shape}", flush=True)
            # --- END SHAPE PRINTS --- 
            # --- ADD DEBUG PRINTS --- 
            print(f">>> [DEBUG Loss Calc] Before .to(): type(shift_labels)={type(shift_labels)}, type(shift_logits)={type(shift_logits)}", flush=True)
            if torch.is_tensor(shift_logits):
                 print(f">>> [DEBUG Loss Calc] shift_logits.device = {shift_logits.device}", flush=True)
            else:
                 print(f">>> [DEBUG Loss Calc] shift_logits is NOT a tensor!", flush=True)
            if 'shift_labels' in locals():
                 print(f">>> [DEBUG Loss Calc] shift_labels exists.", flush=True)
            else:
                 print(f">>> [DEBUG Loss Calc] shift_labels does NOT exist!", flush=True)
            # --- END DEBUG PRINTS --- 
            shift_labels = shift_labels.to(shift_logits.device)
            # Calculate loss
            loss = loss_fct(shift_logits, shift_labels)
            # Print loss info
            print(f">>> [Step] Loss calculated. Value: {loss}, Type: {type(loss)}", flush=True)
            if torch.is_tensor(loss):
                try:
                    print(f">>> [Step] Loss item: {loss.item()}", flush=True)
                except Exception as e_item:
                    print(f">>> [Step] Error getting loss.item(): {e_item}", flush=True)
            elif loss is None:
                 print(">>> [Step] Calculated loss is None!", flush=True)
        else:
            print(">>> [Step] Skipping loss calculation (no labels).", flush=True)

        # --- Return Custom Output Dataclass --- 
        # print(">>> [Step End] EntangledModel forward pass complete.", flush=True)
        return EntangledCausalLMOutput(
            loss=loss, 
            logits=final_logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
            z_star=z_star,
            mu=mu,
            logvar=logvar,
            kl_loss=kl_loss,
        )
    # --- END FULL FORWARD --- 

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


# --- Custom Trainer (Keep COMMENTED OUT) ---
# class EntangledTrainer(Trainer):
#     # Keep __init__ as is, kl_beta won't be used in this test
#     def __init__(self, *args, kl_beta=0.1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.kl_beta = kl_beta 

#     # --- MODIFIED compute_loss for DEBUGGING --- 
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         """
#         DEBUG: Compute loss using ONLY the standard output from the minimal model.
#         """
#         # The model forward pass is the minimal one returning standard outputs
#         outputs = model(**inputs)

#         # Extract the standard LM loss ONLY
#         lm_loss = outputs.loss
#         
#         # IGNORE KL loss for this test
#         # kl_loss = outputs.kl_loss 

#         # Combine the losses (only lm_loss here)
#         if lm_loss is not None:
#             total_loss = lm_loss
#         else:
#             logging.error("LM loss not found in model outputs.")
#             total_loss = None
#         
#         # Print the loss being returned for debugging
#         print(f">>> [DEBUG EntangledTrainer] compute_loss returning: {total_loss.item() if total_loss is not None else 'None'}", flush=True)

#         return (total_loss, outputs) if return_outputs else total_loss
#     # --- END MODIFIED compute_loss ---

if __name__ == "__main__":
    # --- Configuration ---
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "databricks/databricks-dolly-15k"
    commitment_dim = 64
    output_dir = "./results_phase2_dolly"
    kl_beta_value = 0.1

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Dataset Loading and Preprocessing (Adapted for Dolly) --- 
    # Define a formatting function for the Dolly dataset
    def format_dolly(sample):
        instruction = sample["instruction"]
        context = sample["context"]
        response = sample["response"]
        # Create a prompt string similar to Alpaca format
        if context:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"""
        # Add EOS token needed for generation separation, but it might interfere with training if labels are just input_ids
        # Let's return just the formatted prompt for now. SFTTrainer or manual loss handles shifting.
        return prompt # + tokenizer.eos_token 

    def preprocess_function(examples, tokenizer):
        # examples is a dictionary where keys are column names ('instruction', 'context', 'response')
        # and values are lists of strings.
        
        formatted_texts = []
        # Iterate through the batch index
        for i in range(len(examples["instruction"])):
            # Reconstruct a sample dictionary for each item in the batch
            sample = {
                "instruction": examples["instruction"][i],
                "context": examples["context"][i],
                "response": examples["response"][i]
            }
            formatted_texts.append(format_dolly(sample))
            
        # Then tokenize the formatted texts
        tokenized_inputs = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=512) # Keep max_length for now
        # Set labels as input_ids
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    print(f"Loading dataset: {dataset_name}")
    # Dolly dataset doesn't have pre-defined splits, load all and potentially split later if needed
    dataset = load_dataset(dataset_name)["train"] # Dolly only has a 'train' split 
    print(f"Dataset loaded. Size: {len(dataset)}")
    print("Preprocessing dataset...") 
    tokenized_dataset = dataset.map( 
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["instruction", "context", "response", "category"] # Remove original Dolly columns
    )
    print("Dataset preprocessed.") 

    # --- Model Loading (QLoRA Base, try multi-GPU) --- 
    print(f"Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=False,
    )
    base_model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # device_map="auto", # REMOVED: Let Accelerator handle device placement
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # --- PEFT Setup ---
    peft_base_model = get_peft_model(base_model, lora_config)
    peft_base_model.print_trainable_parameters()

    # --- Instantiate Entangled Wrapper --- 
    print("Creating EntangledModel wrapper (Full Forward)...") 
    model = EntangledModel( 
        base_model=peft_base_model, 
        commitment_dim=commitment_dim
    )
    model.variational_head.requires_grad_(True)
    model.Wc_projection.requires_grad_(True)
    print("EntangledModel wrapper created.")

    # --- Training Arguments (Longer run) --- 
    print("Setting up Training Arguments...")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              # Set epochs instead of steps for full dataset run
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=4,   
        optim="paged_adamw_32bit",      
        save_strategy="epoch",           # Save checkpoints every epoch
        logging_steps=100,               # Log every 100 steps for longer run
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,                      
        bf16=True,                       
        max_grad_norm=0.3,
        max_steps=-1,                    # CHANGED: Use num_train_epochs instead
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        # report_to="tensorboard", # Keep commented out
        gradient_checkpointing=False,    
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
    )

    # --- Initialize Standard Trainer (for setup only) --- 
    print("Initializing Standard Trainer for setup...")
    from transformers import default_data_collator
    trainer = Trainer( 
        model=model, 
        train_dataset=tokenized_dataset, 
        args=training_arguments,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    print("Trainer initialized for setup.")

    # --- Get DataLoader FIRST --- 
    print("Getting DataLoader...", flush=True)
    train_dataloader = trainer.get_train_dataloader()
    print("DataLoader obtained.", flush=True)

    # --- Create Optimizer and Scheduler manually (using DataLoader info) --- 
    print("Creating optimizer and scheduler...", flush=True)
    optimizer = trainer.create_optimizer()
    # Calculate total training steps for scheduler
    # Use max_steps directly if provided, otherwise calculate based on dataloader length
    if training_arguments.max_steps > 0:
        num_training_steps = training_arguments.max_steps
    else:
        num_update_steps_per_epoch = len(train_dataloader) // training_arguments.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) # Ensure at least 1 step
        num_training_steps = num_update_steps_per_epoch * training_arguments.num_train_epochs
        
    scheduler = trainer.create_scheduler(num_training_steps=num_training_steps)
    print(f"Optimizer and scheduler created. Num training steps for scheduler: {num_training_steps}", flush=True) # Log num_steps

    # --- Manual Training Loop --- 
    print("Starting MANUAL training loop...")
    model.train() # Set model to training mode
    global_step = 0
    completed_steps = 0

    # --- Use Accelerate for proper device handling and autocast --- 
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision='bf16' if training_arguments.bf16 else 'fp16' if training_arguments.fp16 else 'no')
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    print("Accelerator prepared.", flush=True)

    for epoch in range(int(training_arguments.num_train_epochs)):
        print(f"--- Starting Epoch {epoch+1}/{int(training_arguments.num_train_epochs)} ---") # Keep epoch print
        for step, batch in enumerate(train_dataloader):
            
            # print(f"--- Epoch {epoch+1}, Step {completed_steps+1}/{num_training_steps} --- Batch {step+1} --- ", flush=True) # Remove inner step print
            
            with accelerator.accumulate(model):
                outputs = model(**batch) 
                lm_loss = outputs.loss
                kl_loss = outputs.kl_loss if outputs.kl_loss is not None else torch.tensor(0.0, device=accelerator.device)
                
                total_loss = lm_loss + kl_beta_value * kl_loss
                
                # Keep basic loss print 
                print(f">>> [Global Step {global_step+1}] Loss: {total_loss.item():.4f} (LM: {lm_loss.item():.4f}, KL: {kl_loss.item():.4f})", flush=True)

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # print(f">>> [Step {global_step+1}] Performing gradient sync & optimizer step ...", flush=True)
                    if training_arguments.max_grad_norm is not None:
                         accelerator.clip_grad_norm_(model.parameters(), training_arguments.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1 
                    # print(f">>> [Step {global_step}] Optimizer step complete.", flush=True)
            
            completed_steps += 1
            if training_arguments.max_steps > 0 and global_step >= training_arguments.max_steps:
                print(f">>> Reached max_steps ({training_arguments.max_steps}). Stopping manual loop.")
                break 
                
        if training_arguments.max_steps > 0 and global_step >= training_arguments.max_steps:
            break 

    print("--- Manual training loop finished ---")

    # --- Save Model (Use unwrapped model if using accelerate) --- 
    print("Saving final model components...", flush=True)
    unwrapped_model = accelerator.unwrap_model(model)
    # Save the LoRA adapter weights from the base model
    unwrapped_model.base_model.save_pretrained(output_dir) 
    # Save the custom head and projection weights
    torch.save(unwrapped_model.variational_head.state_dict(), f"{output_dir}/variational_head.pth")
    torch.save(unwrapped_model.Wc_projection.state_dict(), f"{output_dir}/Wc_projection.pth")
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print("Model components saved.", flush=True)

    print(f"Phase 2 Full Training Run ({dataset_name}) Complete")
    pass

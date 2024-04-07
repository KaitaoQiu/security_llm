import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

base_model = 'NousResearch/Llama-2-7b-chat-hf'
new_model = "/scratch/kqa3/security_llm/security_miles_model/security_stack_exchange"
new_model_config = os.path.join(new_model, 'config.json')
train_data_path = '/scratch/kqa3/security_llm/data/security_stack_exchange.parquet'
training_log_path = "/scratch/kqa3/security_llm/results/security_stack_exchange"


cache_dir = '/scratch/kqa3/security_llm/cache/'
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

print("huggingface cache is in {}".format(os.getenv('HF_HOME')))
print("cuda is available(): {}".format(torch.cuda.is_available()))

dataset = dataset = load_dataset("parquet", data_files={'train':train_data_path })
print("finish loading dataset")
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
    cache_dir = cache_dir
)
model.config.use_cache = False
model.config.pretraining_tp = 1



# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load LoRA configuration
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_params = TrainingArguments(
    output_dir=training_log_path,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=2500,
    logging_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    peft_config=peft_args,
    dataset_text_field="train",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train model
print("start training")
trainer.train()
print("finish training")

# Save trained model
# trainer.model.save_pretrained(new_model)
trainer.model.save_pretrained(new_model)
print("save model into {}".format(new_model))


model.config.to_json_file(new_model_config)
print("save model config ...")
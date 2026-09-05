import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
from datasets import ClassLabel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from collections import Counter
from datasets import load_dataset, DatasetDict

ds = load_dataset("metaeval/strategy-qa")
print(f"Dataset Loaded Successfully!")

def format_example(example):
    facts = "\n".join([f"- {fact}" for fact in example['facts']])
    steps = "\n".join([f"- {step}" for step in example['decomposition']])
    answer = "yes" if example['answer'] else "no"
    
    return {
        "prompt": f"""### Question:
{example['question']}
### Context:
{facts}
### Answer:
""",
        "completion": f"""Let me think step by step.
{steps}
Therefore: {answer}"""
    }

ds = ds.cast_column(
    "answer",
    ClassLabel(names=["False", "True"])
)
print(f"Splitting Dataset with test size 0.2")
split = ds['train'].train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="answer"
)

dataset = DatasetDict({
    "train": split['train'],
    "test": split['test']
})

print("Train distribution:", Counter(dataset['train']['answer']))
print("Test distribution:", Counter(dataset['test']['answer']))

train_dataset = dataset['train'].map(
    format_example,
    remove_columns=dataset['train'].column_names
)
eval_dataset = dataset['test'].map(
    format_example,
    remove_columns=dataset['test'].column_names
)

model_name = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # dtype not type
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

training_args = SFTConfig(
    output_dir="./strategyqa-qlora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    max_length=300,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    weight_decay = 0.01,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.model.print_trainable_parameters()

trainer.train()

trainer.save_model("./strategyqa-cot-qlora-final")
tokenizer.save_pretrained("./strategyqa-cot-qlora-final")
print("Saved.")


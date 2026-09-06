import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache\hub"

import torch
import gc
import json
import numpy as np

from datasets import load_dataset, ClassLabel, DatasetDict
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel


from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CHROMA_PATH = r"./chroma_db_strategy"
COLLECTION_NAME = "strategyqa"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
FINETUNED_PATH = "./strategyqa-cot-qlora-final"
DATASET_NAME = "metaeval/strategy-qa"
MAX_INPUT_LEN = 300
MAX_NEW_TOKEN = 150
TOP_K = 3
SEED = 42
TEST_SIZE = 0.2

def load_test_data():
    raw = load_dataset(DATASET_NAME)
    raw = raw.cast_column(
        "answer",
        ClassLabel(names=["False", "True"])
    )
    split = raw['train'].train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
        stratify_by_column="answer"
    )
    dataset = DatasetDict({
    "train": split['train'],
    "test": split['test']
    })
    test_data = dataset["test"]
    print(f"Test set has {len(test_data)} examples.")
    print(f"Distribution: {Counter(test_data['answer'])}")
    return test_data

def load_retriever():
    print("Loading EMBEDDER...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"ChromaDB loaded: {collection.count()} chunks")
    
    print("Loading reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Reranker ready.")
    
    return embedder, collection, reranker

def retrieve(question, collection, embedder, reranker=None, candidates=20):
    embedded = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=embedded,
        n_results=candidates if reranker else TOP_K
    )
    candidate_docs = results['documents'][0]
    
    if reranker is None:
        return "\n\n".join(candidate_docs[:TOP_K])
    
    pairs = [[question, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidate_docs), reverse=True)
    top_docs = [doc for _, doc in ranked[:TOP_K]]
    return "\n\n".join(top_docs)

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_base_model():
    print("Loading BASE model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer

def load_finetuned_model():
    print("Loading Finetuned Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
    base,
    FINETUNED_PATH
)
    model.eval()
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after unload: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def create_eval_prompt(question, context=None, finetuned=False):

    if finetuned:
        prompt = f"""### Question:
{question}
"""

        if context:
            prompt += f"""### Context:
{context}
"""

        prompt += """### Answer:
"""

    else:
        prompt = f"""You are answering a question that requires reasoning.

Determine whether the answer to the question is "yes" or "no".

Respond with ONLY one word: yes or no.

### Question:
{question}
"""

        if context:
            prompt += f"""### Context:
{context}
"""

        prompt += """### Answer:
"""

    return prompt

def predict(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKEN,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip().lower()

    if "therefore:" in response:
        conclusion = response.split("therefore:")[-1].strip()
        first_word = conclusion.split()[0].strip(".,\n") if conclusion.split() else ""
    else:
        # fallback to first word
        first_word = response.split()[0].strip(".,\n") if response.split() else ""
    
    if first_word in ["yes", "true"]:
        return "yes"
    elif first_word in ["no", "false"]:
        return "no"
    else:
        return "unknown"

def run_eval(config_name, test_data, tokenizer, model, collection=None, embedder=None, reranker = None, retrieval = False, finetuned = False):
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")
    predictions = []
    ground_truth = []

    for i, example in enumerate(test_data):
        if retrieval:
            context = retrieve(example["question"], collection, embedder, reranker)
            prompt = create_eval_prompt(example["question"], context, finetuned)
        else:
            prompt = create_eval_prompt(example['question'])
        response = predict(model, tokenizer, prompt)
        predictions.append(response)
        ground_truth.append("yes" if example["answer"] == 1 else "no")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_data)}] done")


    report = classification_report(
        ground_truth, predictions,
        labels=["yes", "no"],
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(
        ground_truth, predictions,
        labels=["yes", "no"]
    )

    print(f"\nClassification Report:")
    print(classification_report(
        ground_truth, predictions,
        labels=["yes", "no"],
        zero_division=0
    ))
    print(f"Confusion Matrix:\n{cm}")
    return {
        "config": config_name,
        "report": report,
        "confusion_matrix": cm.tolist()
    }
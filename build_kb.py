import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from datasets import load_dataset, ClassLabel
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db_strategy"
COLLECTION_NAME = "strategyqa"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# load train split only
raw = load_dataset("metaeval/strategy-qa")
raw = raw.cast_column(
    "answer",
    ClassLabel(names=["False", "True"])
)
split = raw['train'].train_test_split(test_size=0.2, seed=42, stratify_by_column="answer")
train_data = split['train']  # 1832 examples

print(f"Building KB from {len(train_data)} train examples")

# extract all facts as individual documents
documents = []
ids = []
metadatas = []

for i, example in enumerate(train_data):
    for j, fact in enumerate(example['facts']):
        doc_id = f"{example['qid']}_fact_{j}"
        documents.append(fact)
        ids.append(doc_id)
        metadatas.append({
            "qid": example['qid'],
            "question": example['question'][:200],
            "fact_idx": j
        })

print(f"Total facts: {len(documents)}")

# embed and store
embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

batch_size = 512
for start in range(0, len(documents), batch_size):
    end = min(start + batch_size, len(documents))
    batch_docs = documents[start:end]
    batch_ids = ids[start:end]
    batch_meta = metadatas[start:end]
    
    embeddings = embedder.encode(
        batch_docs,
        batch_size=64,
        show_progress_bar=False
    ).tolist()
    
    collection.add(
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"Indexed {end}/{len(documents)}")

print(f"\nDone. Collection count: {collection.count()}")
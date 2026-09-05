# config2.py
import json
from utils import load_test_data, load_base_model, load_retriever, unload_model, run_eval

test_data = load_test_data()
embedder, collection = load_retriever()
model, tokenizer = load_base_model()

results = run_eval(
    config_name="Config 2: Base LLM + RAG",
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        collection=collection,
        embedder=embedder,
        retrieval=True,
)

with open("results_config_2.json", "w") as f:
    json.dump(results, f, indent=2)

unload_model(model)
print("Config 2 complete. Results saved to results_config2.json")
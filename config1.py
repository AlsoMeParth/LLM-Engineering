import json
from utils import load_test_data, load_base_model, unload_model, run_eval

test_data = load_test_data()
model, tokenizer = load_base_model()

results = run_eval(
    config_name = "Config 1: BASE LLM (no RAG)",
    model=model,
    tokenizer=tokenizer,
    test_data=test_data,
)
with open("./results/results_config_1.json", "w") as f:
    json.dump(results, f, indent=2)

unload_model(model)
print("Config 1 complete. Results saved to results_config1.json")

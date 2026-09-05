# compare_results.py
import json

files = {
    "Config 1": "results_config_1.json",
    "Config 2": "results_config_2.json",
    "Config 3": "results_config_3.json"
}

print(f"\n{'='*75}")
print("FINAL COMPARISON TABLE")
print(f"{'='*75}")
print(f"{'Config':<35} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 75)

for name, path in files.items():
    with open(path) as f:
        r = json.load(f)
    
    report = r["report"]
    
    # handle both formats
    if "accuracy" in report:
        acc = report["accuracy"]
    elif "micro avg" in report:
        acc = report["micro avg"]["recall"]
    
    macro_f1 = report["macro avg"]["f1-score"]
    
    print(f"{r['config']:<40} {acc:>10.3f} {macro_f1:>10.3f}")
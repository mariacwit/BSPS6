import json

# File paths (update as needed)
ground_truth_file = "/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/data/harmful_output/gpt-3.5-turbo/AttackEval/evaluation_results_JB.json"  # File with "success" field
predictions_file = "/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/defense_output/gpt3.5-turbo/2Agent/attack-JB_0.json"  # File with "jailbreak_successful" field


# Load ground truth
with open(ground_truth_file, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

# Load predicted results
with open(predictions_file, "r", encoding="utf-8") as f:
    predicted_data = json.load(f)

# Build lookup dictionaries by "name"
ground_truth_lookup = {str(item["nr"]): item["result"] for item in ground_truth_data}
predictions_lookup = {item["name"]: item["valid"] for item in predicted_data}

TP = FP = TN = FN = 0
missing = 0

for name, actual in ground_truth_lookup.items():
    predicted = predictions_lookup.get(name)

    if predicted is None:
        missing += 1
        continue

    # Convert "True"/"False" (as strings) to actual booleans
    actual = actual.strip().lower() != "true"

    if predicted and actual:
        TP += 1
    elif predicted and not actual:
        FP += 1
    elif not predicted and not actual:
        TN += 1
    elif not predicted and actual:
        FN += 1


# Print the confusion matrix and stats
print("\n🔍 Comparison Results:")
print(f"✅ True Positives (TP): {TP}")
print(f"❌ False Positives (FP): {FP}")
print(f"✅ True Negatives (TN): {TN}")
print(f"❌ False Negatives (FN): {FN}")
print(f"⚠️  Missing Predictions: {missing}")

# Metrics
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"\n📊 Precision: {precision:.2f}")
print(f"📈 Recall: {recall:.2f}")
print(f"🏅 F1 Score: {f1:.2f}")


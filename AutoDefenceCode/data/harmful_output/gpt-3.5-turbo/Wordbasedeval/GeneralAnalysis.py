import json

# File paths (update as needed)
ground_truth_file = "AutoDefenceCode/data/harmful_output/gpt-3.5-turbo/processed_attack-OG_0.json"  # File with "success" field
predictions_file = "AutoDefenceCode/defense_output/gpt3.5-turbo/1Agent/attack-OG_0.json"  # File with "jailbreak_successful" field


# Load ground truth
with open(ground_truth_file, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

# Load predicted results
with open(predictions_file, "r", encoding="utf-8") as f:
    predicted_data = json.load(f)

# Build lookup dictionaries by "name"
ground_truth_lookup = {item["name"]: item["jailbreak_successful"] for item in ground_truth_data}
predictions_lookup = {item["name"]: item["success"] for item in predicted_data}

# Initialize confusion matrix counters
TP = FP = TN = FN = 0
missing = 0

# Loop over ground truth and compare with predictions
for name, actual in ground_truth_lookup.items():
    predicted = predictions_lookup.get(name)

    if predicted is None:
        missing += 1
        continue

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

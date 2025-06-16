import json
import pandas as pd

for i in ("JB", "BetterDan", "OG"):
    # File paths
    ground_truth_file = f"/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/data/harmful_output/gpt-3.5-turbo/AttackEval/evaluation_results_{i}.json"
    predictions_file = f"/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/defense_output/gpt3.5-turbo/3Agent/attack-{i}_0.json"
    output_csv_file = f"/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/data/harmful_output/gpt-3.5-turbo/AttackEval/3Agent_{i}.csv"

    # Load data
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    with open(predictions_file, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)

    ground_truth_lookup = {str(item["nr"]): item["result"] for item in ground_truth_data}
    predictions_lookup = {item["name"]: item["valid"] for item in predicted_data}

    false_negatives = []

    for name, actual in ground_truth_lookup.items():
        predicted = predictions_lookup.get(name)
        if predicted is None:
            continue

        actual = actual.strip().lower() != "true"

        if not predicted and actual:
            false_negatives.append(name)

    # Save to separate CSV file
    df_fp = pd.DataFrame(false_negatives, columns=["False Negatives Name"])
    df_fp.to_csv(output_csv_file, index=False)

    print(f"📁 False Negatives for {i} saved to: {output_csv_file}")

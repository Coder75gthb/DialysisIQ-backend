import joblib

def assign_tier(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "AMBER"
    else:
        return "LOW"

model_path = "models/hypo_final_tiered.pkl"

model = joblib.load(model_path)

print("MODEL LOADED SUCCESSFULLY")

print("Number of features:", len(model["all_features"]))
print("First 10 features:", model["all_features"][:10])

print("High threshold:", model["high_threshold"])
print("Low threshold:", model["low_threshold"])


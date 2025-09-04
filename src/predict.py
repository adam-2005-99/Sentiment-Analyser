
#!/usr/bin/env python3
"""CLI predictor for the saved TF-IDF + LinearSVC model."""
import argparse, sys, joblib
from train import basic_clean  # ensures pickled preprocessor resolves

def main():
    p = argparse.ArgumentParser(description="Predict sentiment for a text string.")
    p.add_argument("--text", required=True, help="Quote your text: --text \"i loved it\"")
    p.add_argument("--model_path", default="models/model.joblib")
    args = p.parse_args()

    try:
        pipe = joblib.load(args.model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model at {args.model_path}: {e}")
        sys.exit(1)

    text = args.text.strip()
    pred = int(pipe.predict([text])[0])
    label = {0: "negative", 1: "positive"}[pred]
    print(f"Prediction: {label} ({pred})")

    # probs if available (CalibratedClassifierCV gives this)
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba([text])[0]
            print(f"Probabilities [neg, pos]: [{proba[0]:.3f}, {proba[1]:.3f}]")
        except Exception:
            pass

if __name__ == "__main__":
    main()


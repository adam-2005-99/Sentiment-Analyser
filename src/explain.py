import joblib, numpy as np
from train import basic_clean
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_trained_linear_svc(clf):
    """
    Return the *trained* LinearSVC from a CalibratedClassifierCV or the clf itself.
    Prefer calibrated_classifiers_[].estimator (fitted), then fall back if needed.
    """
    # If the classifier is the calibrated wrapper
    if isinstance(clf, CalibratedClassifierCV):
        # Best: after fit, this list exists and holds the trained estimators
        if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
            est = getattr(clf.calibrated_classifiers_[0], "estimator", None)
            if isinstance(est, LinearSVC):
                return est
        # Fallbacks (may be *unfitted* templates)
        if hasattr(clf, "estimator") and isinstance(clf.estimator, LinearSVC):
            return clf.estimator
        if hasattr(clf, "base_estimator") and isinstance(clf.base_estimator, LinearSVC):
            return clf.base_estimator

    # If it's already a LinearSVC (and hopefully fitted)
    if isinstance(clf, LinearSVC):
        return clf

    return None

def main():
    pipe = joblib.load("models/model.joblib")
    vect = pipe.named_steps["tfidf"]
    clf_wrap = pipe.named_steps["clf"]

    svc = get_trained_linear_svc(clf_wrap)
    if svc is None:
        print("❌ Could not extract LinearSVC. Explainability requires LinearSVC.")
        return

    # Ensure it's fitted (coef_ exists only after fit)
    if not hasattr(svc, "coef_"):
        print("❌ Extracted LinearSVC doesn’t appear fitted (no coef_). "
              "Try retraining, then run explain again.")
        return

    features = np.array(vect.get_feature_names_out())
    coefs = svc.coef_[0]

    top_pos = np.argsort(coefs)[-15:][::-1]
    top_neg = np.argsort(coefs)[:15]

    print("\nTop positive features:")
    for i in top_pos:
        print(f"{features[i]:20s} {coefs[i]: .3f}")

    print("\nTop negative features:")
    for i in top_neg:
        print(f"{features[i]:20s} {coefs[i]: .3f}")

if __name__ == "__main__":
    main()

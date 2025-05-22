import joblib

# 1) Load the pipeline
path = "src/files/models/meta_action_model.pkl"
pipeline = joblib.load(path)

# 2) Inspect its structure
print("Pipeline type:", type(pipeline))
print("Top‐level keys:", list(pipeline.keys()))

# 3) Check base learners
base = pipeline.get("base", {})
print(" → Number of base learners:", len(base))
for name, clf in base.items():
    print(f"   • {name}: {clf.__class__.__name__}")

# 4) Check the meta‐classifier
meta = pipeline.get("meta")
print("Meta‐classifier object:", meta)
print("Meta classes:", getattr(meta, "classes_", "N/A"))

# 5) Try a dummy predict_proba to verify it runs
import numpy as np
if base and meta:
    # make up a single‐sample feature vector of the right dim
    # here we assume each base clf takes a single‐prob feature, so:
    dummy_feats = np.column_stack([clf.predict_proba(np.zeros((1, clf.n_features_in_)))[:,1]
                                   for clf in base.values()])
    print("Dummy meta prediction:", meta.predict(dummy_feats))

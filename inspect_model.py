import joblib

path = r"models/baseline_nb.joblib"
obj = joblib.load(path)

print("TYPE:", type(obj))

if isinstance(obj, dict):
    print("DICT KEYS:", list(obj.keys()))
    for k, v in obj.items():
        print(f" - {k}: {type(v)} | methods: predict={hasattr(v,'predict')}, transform={hasattr(v,'transform')}, predict_proba={hasattr(v,'predict_proba')}")
elif isinstance(obj, (list, tuple)):
    print("LEN:", len(obj))
    for i, v in enumerate(obj):
        print(f"[{i}] {type(v)} | methods: predict={hasattr(v,'predict')}, transform={hasattr(v,'transform')}, predict_proba={hasattr(v,'predict_proba')}")
else:
    print("ATTRS:",
          "predict", hasattr(obj,'predict'),
          "transform", hasattr(obj,'transform'),
          "predict_proba", hasattr(obj,'predict_proba'))

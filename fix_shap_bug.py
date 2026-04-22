import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'explainer = shap.TreeExplainer(xgb_fused)' in source:
            replacement = """# Fix SHAP compatibility with XGBoost 2.x
import json as js
mybooster = xgb_fused.get_booster()
config = js.loads(mybooster.save_config())
config["learner"]["learner_model_param"]["base_score"] = "0.5"
mybooster.load_config(js.dumps(config))
explainer = shap.TreeExplainer(mybooster)"""
            source = source.replace('explainer = shap.TreeExplainer(xgb_fused)', replacement)
            cell['source'] = [line + '\n' for line in source.split('\n')]

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

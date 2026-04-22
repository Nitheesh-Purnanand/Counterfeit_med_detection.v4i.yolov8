import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'explainer = shap.TreeExplainer' in source:
            # Replace the old SHAP fix with the monkey-patch
            replacement = """print('Computing SHAP values for the test set...')
visual_names = ([f'color_{i}' for i in range(512)] +
                [f'lbp_{i}' for i in range(26)] +
                [f'glcm_{i}' for i in range(5)] +
                [f'hu_{i}' for i in range(7)] +
                [f'hog_{i}' for i in range(X_train_visual.shape[1] - 550)])
all_feature_names = visual_names + feature_names

# --- FIX FOR SHAP + XGBOOST 2.x BUG ---
import shap.explainers._tree as shap_tree
old_init = shap_tree.XGBTreeModelLoader.__init__
def new_init(self, xgb_model):
    old_save_config = xgb_model.save_config
    def mock_save_config(*args, **kwargs):
        import json
        config = json.loads(old_save_config(*args, **kwargs))
        val = config.get("learner", {}).get("learner_model_param", {}).get("base_score", "0.5")
        if isinstance(val, str) and val.startswith('['):
            config["learner"]["learner_model_param"]["base_score"] = val.strip('[]')
        return json.dumps(config)
    xgb_model.save_config = mock_save_config
    old_init(self, xgb_model)
shap_tree.XGBTreeModelLoader.__init__ = new_init
# --------------------------------------

explainer = shap.TreeExplainer(xgb_fused)
# Evaluate on a subset to keep it fast
sample_size = min(200, len(X_test_fused))
X_shap = X_test_fused[:sample_size]
shap_values = explainer.shap_values(X_shap)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=all_feature_names, max_display=15, show=False)
plt.title('SHAP Summary - Top 15 Features')
plt.show()"""
            
            # Since there are two cells that use TreeExplainer (one for plot, one for ind_test),
            # Wait, my create_new_contribution.py actually defined TreeExplainer in the plot cell, and reused `explainer` in the ind_test cell!
            # Let's just completely replace the plot cell source.
            if 'plt.figure' in source and 'Computing SHAP values' in source:
                cell['source'] = [line + '\n' for line in replacement.split('\n')]
                # remove the trailing \n for the last element
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SHAP cell patched.")

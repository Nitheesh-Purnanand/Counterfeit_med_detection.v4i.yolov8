import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'xgb_fused = XGBClassifier' in source:
            source = source.replace("xgb_fused = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,",
                                    "xgb_fused = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, base_score=0.5,")
            cell['source'] = [line + '\n' for line in source.split('\n')]

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

import json
import re

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Look for the XGBoost instantiation
        if 'xgb_fused = XGBClassifier' in source:
            # 1. Cast to float32 to save memory
            old_scaler = "X_train_fused = scaler_fused.fit_transform(X_train_fused_raw)"
            new_scaler = "X_train_fused = scaler_fused.fit_transform(X_train_fused_raw).astype(np.float32)"
            source = source.replace(old_scaler, new_scaler)
            
            old_scaler_test = "X_test_fused  = scaler_fused.transform(X_test_fused_raw)"
            new_scaler_test = "X_test_fused  = scaler_fused.transform(X_test_fused_raw).astype(np.float32)"
            source = source.replace(old_scaler_test, new_scaler_test)
            
            # 2. Update XGBoost parameters for low memory (hist tree method, 1 thread)
            old_xgb = "xgb_fused = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,\n                          scale_pos_weight=scale_pos, eval_metric='logloss',\n                          random_state=42, n_jobs=-1)"
            new_xgb = "xgb_fused = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,\n                          scale_pos_weight=scale_pos, eval_metric='logloss',\n                          random_state=42, n_jobs=2, tree_method='hist')"
            
            # Since the user's stack trace has base_score=0.5 in there from my old fix, let's just do a regex replace to be safe
            import re
            source = re.sub(r'xgb_fused = XGBClassifier\([^)]+\)', 
                            r"xgb_fused = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=2, tree_method='hist')", 
                            source)
            
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Memory optimizations applied to XGBoost.")

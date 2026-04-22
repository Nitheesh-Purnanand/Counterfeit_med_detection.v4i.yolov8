import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'X_train_ocr = np.array([extract_ocr_features(p, reader) for p in paths_train])' in source:
            # Fix any weird indentation by standardizing it
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if 'X_train_ocr = np.array([extract_ocr_features(p, reader) for p in paths_train])' in line:
                    new_lines.append('    X_train_ocr = np.array([extract_ocr_features(p, reader) for p in paths_train])')
                elif 'X_test_ocr = np.array([extract_ocr_features(p, reader) for p in paths_test])' in line:
                    new_lines.append('    X_test_ocr = np.array([extract_ocr_features(p, reader) for p in paths_test])')
                elif "pickle.dump({'X_train_ocr': X_train_ocr" in line:
                    new_lines.append("    pickle.dump({'X_train_ocr': X_train_ocr, 'X_test_ocr': X_test_ocr}, open('ocr_features.pkl', 'wb'))")
                elif 'print("OCR features extracted and cached.")' in line:
                    new_lines.append('    print("OCR features extracted and cached.")')
                else:
                    new_lines.append(line)
            
            cell['source'] = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Indentation fixed.")

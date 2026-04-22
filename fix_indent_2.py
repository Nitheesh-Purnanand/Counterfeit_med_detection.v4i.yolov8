import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # We will look for the `else:` block and replace it correctly
        if 'else:' in source and 'X_train_ocr = np.array' in source:
            lines = source.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if line.startswith('else:'):
                    new_lines.append('else:')
                    new_lines.append('    print("Extracting OCR features (this may take a while)...")')
                    new_lines.append('    X_train_ocr = np.array([extract_ocr_features(p, reader) for p in paths_train])')
                    new_lines.append('    X_test_ocr = np.array([extract_ocr_features(p, reader) for p in paths_test])')
                    new_lines.append("    pickle.dump({'X_train_ocr': X_train_ocr, 'X_test_ocr': X_test_ocr}, open('ocr_features.pkl', 'wb'))")
                    new_lines.append('    print("OCR features extracted and cached.")')
                    skip = True
                elif skip and ('print' in line or 'X_train_ocr' in line or 'X_test_ocr' in line or 'pickle.dump' in line or 'OCR features extracted' in line or 'extracting' in line.lower()):
                    pass # skipping the bad lines
                else:
                    new_lines.append(line)
            cell['source'] = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

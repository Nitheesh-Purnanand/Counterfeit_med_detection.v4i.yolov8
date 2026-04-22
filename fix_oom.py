import json

with open('contribution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'import easyocr' in source:
            source = source.replace('import easyocr\n', '')
            cell['source'] = [line + '\n' for line in source.split('\n')]
            
        if 'reader = easyocr.Reader' in source and 'import easyocr' not in source:
            source = source.replace('reader = easyocr.Reader', 'import easyocr\n    reader = easyocr.Reader')
            cell['source'] = [line + '\n' for line in source.split('\n')]

with open('contribution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

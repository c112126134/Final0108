import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Final0108.final0108 import INDEX_TEMPLATE, DEFAULT_PARAMS
params = DEFAULT_PARAMS.copy()
params.update({'ticker':'AAPL','start':'2018-01-01','end':'2025-12-18','capital':100000})
html = INDEX_TEMPLATE.render(params=params)
with open('index_preview.html','w',encoding='utf-8') as f:
    f.write(html)
print('Wrote index_preview.html')
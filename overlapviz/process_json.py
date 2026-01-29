import json
import pandas as pd
import pickle

# 读取 JSON
with open('geometric_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为以 shape_id 为键的字典
venn_dict = {}
for item in data:
    sid = "shape" + str(item['shape_id']).strip("[]").strip("'\"")
    venn_dict[sid] = {
        'shape_id': sid,
        'type': str(item['type']).strip("[]").strip("'\""),
        'nsets': str(item['nsets']).strip("[]"),
        'set_edge': pd.DataFrame(item['set_edge']),
        'set_label': pd.DataFrame(item['set_label']),
        'region_edge': pd.DataFrame(item['region_edge']),
        'region_label': pd.DataFrame(item.get('region_label', []))
    }

venn_dict
print(venn_dict['shape404'])


# 保存
with open('geometric_data.pkl', 'wb') as f:
    pickle.dump(venn_dict, f)

#读取
with open('venn_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# 直接使用
print(loaded_dict['shape404']['set_edge'])

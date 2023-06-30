import json
import os

from src.pre_data import load_data

ls = load_data("./data/training_k.json")
# print(ls[1])
# random.shuffle(ls)
ls1 = []
ls2 = []
for item in ls:
    if "k" in item:
        ls1.append(item)
    else:
        ls2.append(item)
        item['k'] = "None"
        item['k_mask'] = "None"
ls3 = ls1 + ls2
print(f"k_data num:{len(ls1)}")
print(f"no_k_data num:{len(ls2)}")
fw=open(os.path.join("./data/train_k.json"),"w",encoding='utf-8')
a = json.dumps(ls1,sort_keys=True,indent=4,ensure_ascii=False)
# print(a)
fw.write(a)
fw.close()


fw=open(os.path.join("./data/train_no_k.json"),"w",encoding='utf-8')
a = json.dumps(ls2,sort_keys=True,indent=4,ensure_ascii=False)
# print(a)
fw.write(a)
fw.close()

fw=open(os.path.join("./data/train_k_merge_203.json"),"w",encoding='utf-8')
a = json.dumps(ls3,sort_keys=True,indent=4,ensure_ascii=False)
# print(a)
fw.write(a)
fw.close()
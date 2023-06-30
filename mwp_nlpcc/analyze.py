import json
import os
import re

from src.pre_data import load_data

data = load_data("./data/val1.json")
with open("./results/mengzi-bert-base_4e-5_DA/test.txt", "r", encoding="utf-8") as f:
    A1 = []
    lines = f.readlines()
    for line in lines:
        t_or_f, _ = line.strip().split(',', maxsplit=1)
        A1.append([t_or_f, _])

with open("./results/mengzi-bert-base_8e-5/test.txt", "r", encoding="utf-8") as f:
    A2 = []
    lines = f.readlines()
    for line in lines:
        t_or_f, _ = line.strip().split(',', maxsplit=1)
        A2.append([t_or_f, _])
A3 = []
A3_id = []
A3_json = []
k = 0
for q, i1, i2 in zip(data, A1, A2):
    if i1[0] == "True" and i2[0] == "False":
        k += 1
        A3.append(f"Q{k}:\nid:{q['id']}\n question:{q['segmented_text']} \n A1:  {i1} \n A2  :{i2}\n True Answer:{q['answer']}\n\n")
        A3_id.append(f"{q['id']}\n")
        A3_json.append(q)



with open("./results/new_DA.txt", "w", encoding="utf-8") as f:
    f.writelines(A3)

with open("./results/id_new_DA.txt", "w", encoding="utf-8") as f:
    f.writelines(A3_id)

with open("./results/id_new_DA.json", "w", encoding="utf-8") as f:
    a = json.dumps(A3_json,sort_keys=True,indent=4,ensure_ascii=False)
    f.write(a)



A4 = []
A4_id = []
k = 0
A4_json = []
for q, i1, i2 in zip(data, A1, A2):
    if i1[0] == "False" and i2[0] == "True":
        k += 1
        A4.append(f"Q{k}:\nid:{q['id']}\n question:{q['segmented_text']} \n A1:  {i1} \n A2  :{i2}\n True Answer:{q['answer']}\n\n")
        A4_id.append(f"{q['id']}\n")
        A4_json.append(q)


with open("./results/new.txt", "w", encoding="utf-8") as f:
    f.writelines(A4)

with open("./results/id_new.txt", "w", encoding="utf-8") as f:
    f.writelines(A4_id)

with open("./results/id_new.json", "w", encoding="utf-8") as f:
    a = json.dumps(A4_json,sort_keys=True,indent=4,ensure_ascii=False)
    f.write(a)




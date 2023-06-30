import random
import re

import numpy as np
import torch

from src.pre_data import transfer_num, load_data
from src.pre_data import load_raw_data
import json
import os
from src.expressions_transfer import compute_prefix_expression

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('PLMs/bert-base-chinese')
#
# b = '四、五、六年级捐款钱数的比是5：6：7， 共5+6+7份，那么六年级的捐款占7份'
# c = tokenizer.encode(b)
# print(c)
# print(tokenizer.decode(c))



# expression = ['/', '+', '-', '100', '*', '20', '2', '10', '2']
# print(compute_prefix_expression(expression))
# import os
# path = './results/mengzi-bert-base'
# if os.path.exists(path):
#     print()
# else:
#     os.makedirs(path)

# np.random.seed(42)
# prob = np.random.rand(1)
# print(prob.item())

# a = torch.LongTensor([[1,2], [1, 2, 3]])
# print(a)


# def load_data(filename):
#     print("Reading lines...")
#     f = open(filename, encoding="utf-8")
#     json_data = json.load(f)
#     data = []
#     for i, s in enumerate(json_data):
#         if "equation" in s:
#             if "千米/小时" in s["equation"]:
#                 s["equation"] = s["equation"][:-5]
#             if re.search("\d+\(\(\d+\)/\(\d+\)\)", s["ans"]):
#                 b = re.sub("\(\(", "+(", s["ans"])
#                 c = re.sub("\)\)", ")", b)
#                 s["ans"] = c
#             s["ans"] = re.sub("%", "/100", s["ans"])
#         data.append(s)
#     return data
#
#
# ls = load_data("./data/training_k.json")
# # print(ls[1])
# # random.shuffle(ls)
# ls1 = []
# ls2 = []
# for item in ls:
#     if "k" in item:
#         ls1.append(item)
#     else:
#         ls2.append(item)
#
# fw=open(os.path.join("./data/train_k.json"),"w",encoding='utf-8')
# a = json.dumps(ls1,sort_keys=True,indent=4,ensure_ascii=False)
# # print(a)
# fw.write(a)
# fw.close()
#
#
# fw=open(os.path.join("./data/train_no_k.json"),"w",encoding='utf-8')
# a = json.dumps(ls2,sort_keys=True,indent=4,ensure_ascii=False)
# # print(a)
# fw.write(a)
# fw.close()


from src.test import print_index
print_index(".")













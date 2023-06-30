import json
import os

f=open("../data/test_out1.csv", "r", encoding='gbk') #
ls=[]
for line in f:
        line = line.replace("\n", "")
        temp = line.split(",")
        # del temp[0]
        ls.append(temp)

f.close()
print(ls[:2])
os.path.join("../data/test_out1.json")
fw=open("../data/test_out1.json","w",encoding='utf-8')
for i in range(1,len(ls)):
    ls[i]=dict(zip(ls[0],ls[i]))
ls_w = ls[1:]
for item in ls_w:
    item['segmented_text'] = item['ori_text']
a = json.dumps(ls_w,sort_keys=True,indent=4,ensure_ascii=False)
# print(a)
fw.write(a)
fw.close()






import json

import torch
from torch import nn

from cluster import feature_extractor
from src.pre_data import load_data
from test_API import test_api
import pandas as pd
USE_CUDA = True

class CosineSimilarity(nn.Module):

    def forward(self, tensor_1, tensor_2):
        if USE_CUDA:
            tensor_1.cuda()
            tensor_2.cuda()
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        new_nt2 = normalized_tensor_2.transpose(1, 0)
        cos_sim_tensor = torch.mm(normalized_tensor_1, new_nt2)
        # max_tensor, _ = cos_sim_tensor.topk(3, dim=-1)
        # # print(max_tensor.shape)
        # return max_tensor.mean(dim=-1)
        max_tensor, _ = cos_sim_tensor.topk(3, dim=-1)
        # print(max_tensor.shape)
        return max_tensor.mean(dim=-1)



def select_model(test_data_path, test_flag=False):
    # model_path1 = "./models/mengzi-bert-base_7e-5"
    model_path2 = "./models/mengzi-bert-base_4e-5_DA"
    plm_lang_path = "./parameter/plm_lang.pkl"
    other_path = "./parameter/plm_other.pkl"
    pretrain_path = "./PLMs/mengzi-bert-base"
    data_path_62 = "./results/analyze_test_id_62.json"
    data_path_37 = "./results/analyze_test_id_37.json"
    # data_path_62 = "./results/id_new_DA.json"
    # data_path_37 = "./results/id_new.json"

    # test_data_path = "./results/analyze_test_id_37.json"

    test_data_feature = feature_extractor(test_data_path, pretrain_path, model_path2, plm_lang_path, other_path, test_flag=test_flag)
    data_da = feature_extractor(data_path_62, pretrain_path, model_path2, plm_lang_path, other_path)
    data_woda = feature_extractor(data_path_37, pretrain_path, model_path2, plm_lang_path, other_path)

    test_data_tensor = torch.tensor(test_data_feature)
    data_da = torch.tensor(data_da)
    data_woda = torch.tensor(data_woda)
    cos_sim = CosineSimilarity()
    score_mdoel_da = cos_sim(test_data_tensor, data_da)
    print(score_mdoel_da.shape)
    score_mdoel_woda = cos_sim(test_data_tensor, data_woda)
    print(score_mdoel_woda)
    # select_flag = torch.zeros((score_mdoel_da.shape[0],))
    select_flag = score_mdoel_da > score_mdoel_woda
    # print(select_flag)

    return select_flag


def split_test_data(test_data_path, test_flag=False):
    if test_flag:
        data2DACS_path = "./data/data2DACS_test.json"
        data2bs_path = "./data/data2bs_test.json"
    else:
        data2DACS_path = "./data/data2DACS.json"
        data2bs_path = "./data/data2bs.json"
    test_data = load_data(test_data_path)
    select_flag = select_model(test_data_path, test_flag=test_flag)
    select_flag = select_flag.tolist()
    print(select_flag)
    data2DACS = []
    data2bs = []
    for i in range(len(test_data)):
        if select_flag[i] is True:
            data2DACS.append(test_data[i])
        else:
            data2bs.append(test_data[i])
    with open(data2DACS_path, 'w', encoding='utf-8') as f:
        a = json.dumps(data2DACS, sort_keys=True, indent=4, ensure_ascii=False)
        f.write(a)
    with open(data2bs_path, 'w', encoding='utf-8') as f:
        a = json.dumps(data2bs, sort_keys=True, indent=4, ensure_ascii=False)
        f.write(a)


# split_test_data("./data/val1.json")
def test_on_DACS_model(data_path, test_flag=False):
    model_path = "./models/mengzi-bert-base_4e-5_DA"
    plm_lang_path = "./parameter/plm_lang.pkl"
    other_path = "./parameter/plm_other.pkl"
    pretrain_path = "./PLMs/mengzi-bert-base"
    # data_path = "./data/test.json"
    answer_list, val_ac_num = test_api(model_path, plm_lang_path, other_path, pretrain_path, data_path,
                                       encoder_type="plm", beam_size=3, test_flag=test_flag)

    return answer_list, val_ac_num


def test_on_bs_model(data_path, test_flag=False):
    model_path = "./models/mengzi-bert-base_7e-5"
    plm_lang_path = "./parameter/plm_lang.pkl"
    other_path = "./parameter/plm_other.pkl"
    pretrain_path = "./PLMs/mengzi-bert-base"
    # data_path = "./data/test.json"
    answer_list, val_ac_num = test_api(model_path, plm_lang_path, other_path, pretrain_path, data_path,
                                       encoder_type="plm", beam_size=3, test_flag=test_flag)

    return answer_list, val_ac_num

def get_data_id(path):
    data = load_data(path)
    id_list = []
    for item in data:
        id_list.append(item['id'])
    return id_list


def ensenable_test(DACS_data_path, bs_data_path, test_data_path, test_flag=False):
    DACS_data_id = get_data_id(DACS_data_path)
    bs_data_id = get_data_id(bs_data_path)
    assert len(DACS_data_id) + len(bs_data_id) == 1200

    DACS_answer_list, DACS_val_num = test_on_DACS_model(DACS_data_path, test_flag=test_flag)
    bs_answer_list, bs_val_num = test_on_bs_model(bs_data_path, test_flag=test_flag)
    acc = (DACS_val_num + bs_val_num)/1200
    print(f"final acc:{acc:.3f}")

    data_dict = {}
    for idx, answer in zip(DACS_data_id, DACS_answer_list):
        data_dict[idx] = answer
    for idx, answer in zip(bs_data_id, bs_answer_list):
        data_dict[idx] = answer
    assert len(data_dict) == 1200

    all_data_id = get_data_id(test_data_path)

    all_answer_list = []
    for idx in all_data_id:
        all_answer_list.append(data_dict[idx])

    csv_data = {'id': all_data_id, 'prediction': all_answer_list}
    df = pd.DataFrame(csv_data)
    if test_flag:
        df.to_csv("./data/submission_07.csv", index=False)
    else:
        df.to_csv("./data/submission_07_val.csv", index=False)


test_flag = True

if test_flag:
    split_test_data("./data/test_out1.json", test_flag=test_flag)
    ensenable_test("./data/data2DACS_test.json", "./data/data2bs_test.json", "./data/test_out1.json", test_flag=test_flag)
else:
    split_test_data("./data/val1.json", test_flag=test_flag)
    ensenable_test("./data/data2DACS.json", "./data/data2bs.json", "./data/val1.json", test_flag=test_flag)











import os

from plm2tree import test_no_equation, loading_data, test_no_equation_feedback
import torch
import pickle
from src.models import EncoderPLM, EncoderPLM_MLM
from src.pre_data import Lang, prepare_test_data
from plm2tree import init_model
from transformers import BertTokenizer, BertModel


USE_CUDA = True

def load_para(lang_path, other_path):
    out_lang = Lang()
    with open(lang_path, 'rb') as file:
        out_lang = pickle.loads(file.read())

    with open(other_path, 'rb') as f:
        generate_nums, copy_nums, hidden_size, embedding_size, dropout = pickle.load(f)

    return out_lang, generate_nums, copy_nums, hidden_size, embedding_size, dropout




def test_api(model_path, lang_path, other_path, pretrain_path, data_path, encoder_type="plm", beam_size=3, test_flag=False):

    print(f"tesing in model {model_path}")
    out_lang, generate_nums, copy_nums, hidden_size, embedding_size, dropout = load_para(lang_path, other_path)
    if encoder_type == "plm":
        encoder = EncoderPLM(hidden_size=hidden_size, auto_transformer=False, pretrain_path=pretrain_path, dropout=dropout)
    elif encoder_type == "plm_mlm":
        encoder = EncoderPLM_MLM(hidden_size=hidden_size, auto_transformer=False, pretrain_path=pretrain_path,
                             dropout=dropout)
    else:
        print(f"encoder type:{encoder_type} not in our lib(use 'plm' or 'plm_mlm')")

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(out_lang.word2index[num])

    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    encoder.bert_model.resize_token_embeddings(len(tokenizer))

    encoder_state_dict = torch.load(os.path.join(f"./{model_path}/best_encoder.ckpt"))
    encoder.load_state_dict(encoder_state_dict)
    predict, generate, merge = init_model(hidden_size, embedding_size,copy_nums, generate_nums, out_lang, dropout)
    generate_sd = torch.load(os.path.join(f"./{model_path}/best_generate.ckpt"))
    merge_sd = torch.load(os.path.join(f"./{model_path}/best_merge.ckpt"))
    predict_sd = torch.load(os.path.join(f"./{model_path}/best_predict.ckpt"))
    generate.load_state_dict(generate_sd)
    merge.load_state_dict(merge_sd)
    predict.load_state_dict(predict_sd)

    val_data, val_generate_nums, val_copy_nums, answer_list = loading_data(data_path, val_data=True, test_flag=test_flag)

    if USE_CUDA:
        encoder.cuda()
        generate.cuda()
        merge.cuda()
        predict.cuda()
    else:
        encoder.cpu()
        generate.cpu()
        merge.cpu()
        predict.cpu()

    val_pair = prepare_test_data(val_data, out_lang, tokenizer, PLM_flag=True, tree=True)

    with torch.no_grad():
        equation_acc, value_acc, temp_list, val_ac_num = test_no_equation_feedback(val_pair, generate_num_ids, encoder, predict, generate,
                                                              merge,
                                                              out_lang,
                                                              beam_size, answer_list)
        # equation_acc, value_acc, temp_list = test_no_equation(val_pair, generate_num_ids, encoder,
        #                                                                            predict, generate,
        #                                                                            merge,
        #                                                                            out_lang,
        #                                                                            beam_size, answer_list)


    print(f"accuracy:{value_acc:.3f}")
    # print(temp_list)
    return temp_list, val_ac_num # 对应 test_no_equation_feedback
    # return temp_list
    # print(temp_list)


if __name__ == "__main__":
    model_path = "./models/mengzi-bert-base_7e-5"
    plm_lang_path = "./para/plm_lang.pkl"
    other_path = "./para/plm_other.pkl"
    pretrain_path = "./PLMs/mengzi-bert-base"
    # data_path = "./data/beamsearch_e3.json"
    data_path = "./data/val1.json"
    _ = test_api(model_path, plm_lang_path, other_path, pretrain_path, data_path, encoder_type="plm", beam_size=3, test_flag=False)


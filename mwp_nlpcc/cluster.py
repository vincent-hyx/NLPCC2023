import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import BertTokenizer

from plm2tree import loading_data
from src.models import EncoderPLM
from src.pre_data import prepare_test_data
from test_API import load_para

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def feature_extractor(data_path, pretrain_path, model_path, lang_path, other_path, test_flag=False):
    encoder = EncoderPLM(hidden_size=768, auto_transformer=False, pretrain_path=pretrain_path, dropout=0.5)

    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    encoder.bert_model.resize_token_embeddings(len(tokenizer))

    encoder_state_dict = torch.load(os.path.join(f"./{model_path}/best_encoder.ckpt"))
    encoder.load_state_dict(encoder_state_dict)

    encoder.eval()
    out_lang, generate_nums, copy_nums, hidden_size, embedding_size, dropout = load_para(lang_path, other_path)
    val_data, val_generate_nums, val_copy_nums, answer_list = loading_data(data_path, val_data=True, test_flag=test_flag)
    val_pair = prepare_test_data(val_data, out_lang, tokenizer, PLM_flag=True, tree=True)
    feature_list = []
    with torch.no_grad():
        for item in val_pair:
            input_batch = item[0]
            input_var = torch.LongTensor(input_batch).unsqueeze(0)
            _, problem_output = encoder(input_var[:, 0, :], input_var[:, 1, :])
            feature_list.append(np.array(problem_output.squeeze(0)))
    return np.array(feature_list)

def s_tne(feature_list1, feature_list2, feature_list3, feature_list4):
    # model = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    model = TSNE(n_components=2, random_state=33)
    X_tsne1 = model.fit_transform(np.concatenate([feature_list3,feature_list4],axis=0))
    # X_pca = PCA(n_components=2).fit_transform(feature_list)
    # X_tsne2 = model.fit_transform(feature_list2)
    # X_tsne3 = model.fit_transform(feature_list3)
    # X_tsne4 = model.fit_transform(feature_list4)
    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], c=[_ for _ in range(99)], label="feature")
    # plt.scatter(X_tsne1[1:, 0], X_tsne1[1:, 1], c='y', label="feature_37")
    # plt.scatter(X_tsne1[2:, 0], X_tsne1[2:, 1], c='k', label="feature_62_DA")
    # plt.scatter(X_tsne1[3:, 0], X_tsne1[3:, 1], c='y', label="feature_37_DA")
    # plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], c='g', label="feature_62")
    # plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c='y', label="feature_37")
    # plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1], c='r', label="feature_62_DA")
    # plt.scatter(X_tsne4[:, 0], X_tsne4[:, 1], c='b', label="feature_37_DA")
    # plt.scatter(
    #     model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    #     s=250, marker='*',
    #     c='red', edgecolors='black',
    #     label='centroids'
    # )
    # plt.subplot(122)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], label="PCA")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig('images/tsne_62_37_DA.png', dpi=120)
    plt.show()

if __name__ == "__main__":
    model_path1 = "./models/mengzi-bert-base_7e-5"
    model_path2 = "./models/mengzi-bert-base_4e-5_DA"
    plm_lang_path = "./parameter/plm_lang.pkl"
    other_path = "./parameter/plm_other.pkl"
    pretrain_path = "./PLMs/mengzi-bert-base"
    data_path = "./results/analyze_test_id_62.json"
    data_path1 = "./results/analyze_test_id_37.json"
    feature_list1 = feature_extractor(data_path, pretrain_path, model_path1, plm_lang_path, other_path)
    feature_list2 = feature_extractor(data_path1, pretrain_path, model_path1, plm_lang_path, other_path)

    feature_list3 = feature_extractor(data_path, pretrain_path, model_path2, plm_lang_path, other_path)
    feature_list4 = feature_extractor(data_path1, pretrain_path, model_path2, plm_lang_path, other_path)

    s_tne(feature_list1, feature_list2, feature_list3, feature_list4)
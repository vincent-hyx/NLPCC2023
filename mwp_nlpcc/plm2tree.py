import argparse
import copy
import logging
import os
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW, AutoTokenizer
from transformers import BertTokenizer

from src.AST_transfer import equivalent_transfer
from src.expressions_transfer import from_infix_to_prefix, out_expression_list, compute_prefix_expression
from src.models import Prediction, GenerateNode, Merge, EncoderPLM, EncoderPLM_MLM
from src.pre_data import load_raw_data, transfer_num, prepare_train_batch, prepare_data_with_PLM, load_data, \
    transfer_num_for_val_data, transfer_num_gts
from src.train_and_evaluate import train_tree, time_since, evaluate_tree, compute_prefix_tree_result, \
    compute_val_result, train_tree_for_PLM
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_arguments(parser: argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--pretrain_path', type=str, default="chinese-bert-wwm-ext")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for PLM last linearLayer")
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--train_data_file', type=str, default="./data/training.json")
    parser.add_argument('--val_data_file', type=str, default="./data/val.json")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--output_dir', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--results_dir', type=str, default="results",
                        help="the name of the models, to save the results of testing data")
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=3500)
    parser.add_argument('--DA', type=bool, default=False)

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(f"{k} = {args.__dict__[k]}")
    return args


def loading_data(data_path, val_data=False, DA=False, test_flag=False):
    data = load_data(data_path)
    if val_data:
        pairs, generate_nums, copy_nums, answer_list = transfer_num_for_val_data(data, test_flag=test_flag)
    else:
        pairs, generate_nums, copy_nums = transfer_num(data)

    temp_pairs = []
    for p in pairs:
        temp_eq = from_infix_to_prefix(p[1])
        temp_pairs.append((p[0], temp_eq, p[2], p[3]))
        # edited by xu: equivalent transformer of eq to augment data
        # id 为 8883 的 数据中 做了改动 负号前添零
        if DA:
            eq = copy.deepcopy(temp_eq)
            if len(eq) >= 3 and val_data is False:
                # print(eq)
                # logger.info(p[0])
                eq_list = equivalent_transfer(eq)
                # print(eq_list)
                if len(eq_list):
                    for _ in eq_list:
                        temp_pairs.append((copy.deepcopy(p[0]), _, copy.deepcopy(p[2]), copy.deepcopy(p[3])))

    pairs = temp_pairs
    if val_data:
        return pairs, generate_nums, copy_nums, answer_list
    else:
        return pairs, generate_nums, copy_nums



def init_model(hidden_size, embedding_size, copy_nums, generate_nums, output_lang, dropout):


    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums), dropout=dropout)
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size,dropout=dropout)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size,dropout=dropout)
    return predict, generate, merge


def train(encoder, predict, generate, merge, train_pairs, val_pair, batch_size, n_epochs, learning_rate, generate_nums,
          output_lang, USE_CUDA, schedule, output_dir, beam_size, warmup_steps, result_dir, model_name,answer_list=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"run on {device}")


    need_optimized_parameters = []
    for module in [encoder, predict, generate, merge]:
        need_optimized_parameters += [p for n, p in module.named_parameters() if p.requires_grad]

    t_total = (len(train_pairs) // batch_size + 1) * n_epochs
    optimizer = AdamW([{'params': need_optimized_parameters, 'weight_decay': 0.0}], lr=learning_rate)

    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        assert schedule in ['linear', 'constant'], "Schedule args %s is not in set." % schedule


    """
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    """

    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    best_equation_acc = 0
    best_value_acc = 0
    best_epoch_equation = -1
    best_epoch_value = -1

    for epoch in range(n_epochs):


        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(
            train_pairs, batch_size)
        logger.info(f"epoch:{epoch + 1}")
        start = time.time()
        for idx in tqdm(range(len(input_lengths)), desc=f"training....."):
            loss = train_tree_for_PLM(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                output_lang, num_pos_batches[idx])
            loss_total += loss

            optimizer.step()
            scheduler.step()

            encoder.zero_grad()
            predict.zero_grad()
            generate.zero_grad()
            merge.zero_grad()


        logger.info(f"loss:{loss_total / len(input_lengths)}")
        logger.info(f"training time{time_since(time.time() - start)}")
        logger.info("--------------------------------")

        temp_list = []
        if answer_list is None:
            equation_acc, value_acc = test(val_pair, generate_num_ids, encoder, predict, generate, merge, output_lang, beam_size)
        else:
            equation_acc, value_acc, temp_list = test_no_equation(val_pair, generate_num_ids, encoder, predict, generate, merge, output_lang,
                                           beam_size, answer_list)

        if best_equation_acc < equation_acc:
            best_equation_acc = equation_acc
            best_epoch_equation = epoch + 1
        if best_value_acc < value_acc:
            best_value_acc = value_acc
            best_epoch_value = epoch + 1
            path = result_dir
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
            with open(os.path.join(f"./{result_dir}/test.txt"), 'w', encoding='utf-8') as f:
                f.writelines(temp_list)

            path = output_dir
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
            torch.save(encoder.state_dict(), os.path.join(f"./{output_dir}/best_encoder.ckpt"))
            torch.save(predict.state_dict(), os.path.join(f"./{output_dir}/best_predict.ckpt"))
            torch.save(generate.state_dict(), os.path.join(f"./{output_dir}/best_generate.ckpt"))
            torch.save(merge.state_dict(), os.path.join(f"./{output_dir}/best_merge.ckpt"))

        logger.info(f"best_equation_acc:{best_equation_acc:.4f} in epoch:{best_epoch_equation}")
        logger.info(f"best_value_acc:{best_value_acc:.4f} in epoch:{best_epoch_value}")

        # 将 train accuracy 保存到 "tensorboard/train" 文件夹
        log_dir = os.path.join(f'tensorboard/{model_name}_{learning_rate}_ori_DA', 'train')
        train_writer = SummaryWriter(log_dir=log_dir)
        # 将 test accuracy 保存到 "tensorboard/test" 文件夹
        log_dir = os.path.join(f'tensorboard/{model_name}_{learning_rate}_ori_DA', 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # 绘制
        train_writer.add_scalar('Loss', loss_total / len(input_lengths), epoch + 1)
        # test_writer.add_scalar('Loss', test_loss, epoch + 1)
        # train_writer.add_scalar('Acc', all_train_acc / total, epoch + 1)
        test_writer.add_scalar('equation_acc', equation_acc, epoch + 1)
        test_writer.add_scalar('value_acc', value_acc, epoch + 1)

        # path = output_dir
        # if os.path.exists(path):
        #     pass
        # else:
        #     os.makedirs(path)
        # torch.save(encoder.state_dict(), os.path.join(f"./{output_dir}/epoch_{epoch + 1}_encoder.ckpt"))
        # torch.save(predict.state_dict(), os.path.join(f"./{output_dir}/epoch_{epoch + 1}_predict.ckpt"))
        # torch.save(generate.state_dict(), os.path.join(f"./{output_dir}/epoch_{epoch + 1}_generate.ckpt"))
        # torch.save(merge.state_dict(), os.path.join(f"./{output_dir}/epoch_{epoch + 1}_merge.ckpt"))




def test(test_pairs, generate_num_ids, encoder, predict, generate, merge, output_lang, beam_size):
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()
    for test_batch in tqdm(test_pairs, desc=f"evaluating....."):
        # print(f"{len(test_batch)}")
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], beam_size=beam_size)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                          test_batch[6])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1
    logger.info(f"equation_ac:{equation_ac} || value_ac:{value_ac} || eval_total:{eval_total}")
    logger.info(f"test_equation_acc:{float(equation_ac) / eval_total} test_val_acc:{float(value_ac) / eval_total}")
    logger.info(f"testing time:{time_since(time.time() - start)}")
    logger.info("------------------------------------------------------")
    return float(equation_ac) / eval_total, float(value_ac) / eval_total

def test_no_equation(test_pairs, generate_num_ids, encoder, predict, generate, merge, output_lang, beam_size,answer_list):
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    temp_list = []
    start = time.time()
    for test_batch, answer in tqdm(zip(test_pairs, answer_list), desc=f"evaluating....."):
        # print(f"{len(test_batch)}")
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], beam_size=beam_size)
        # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                          # test_batch[6])
        val_ac, generate_eq, val_value = compute_val_result(test_res[0].out, answer, output_lang, test_batch[4], test_batch[6])
        temp_list.append(f"{val_ac},{generate_eq},{val_value},{answer}\n")
        if val_ac:
            value_ac += 1
            equation_ac += 1
        eval_total += 1

    logger.info(f"equation_ac:{equation_ac} || value_ac:{value_ac} || eval_total:{eval_total}")
    logger.info(f"test_equation_acc:{float(equation_ac) / eval_total} test_val_acc:{float(value_ac) / eval_total}")
    logger.info(f"testing time:{time_since(time.time() - start)}")
    logger.info("------------------------------------------------------")
    return float(equation_ac) / eval_total, float(value_ac) / eval_total, temp_list

def test_no_equation_feedback(test_pairs, generate_num_ids, encoder, predict, generate, merge, output_lang, beam_size,answer_list):
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    temp_list = []
    start = time.time()
    for test_batch, answer in tqdm(zip(test_pairs, answer_list), desc=f"evaluating....."):
        # print(f"{len(test_batch)}")
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], beam_size=beam_size)
        # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                          # test_batch[6])
        for item in test_res:
            search_result = out_expression_list(item.out, output_lang, test_batch[4])
            final_answer = compute_prefix_expression(search_result)
            print(search_result, np.exp(item.score), final_answer)


        val_ac, generate_eq, val_value = compute_val_result(test_res[0].out, answer, output_lang, test_batch[4], test_batch[6])
        temp_list.append(val_value)
        if val_ac:
            value_ac += 1
            equation_ac += 1
        eval_total += 1

    logger.info(f"equation_ac:{equation_ac} || value_ac:{value_ac} || eval_total:{eval_total}")
    logger.info(f"test_equation_acc:{float(equation_ac) / eval_total} test_val_acc:{float(value_ac) / eval_total}")
    logger.info(f"testing time:{time_since(time.time() - start)}")
    logger.info("------------------------------------------------------")
    return float(equation_ac) / eval_total, float(value_ac) / eval_total, temp_list, value_ac




def train_model():
    parser = argparse.ArgumentParser(description="classificaton")
    print(parser)
    opt = parse_arguments(parser)
    # print(opt.learning_rate, type(opt.learning_rate))
    if torch.cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    logger.info(f"[INFO]------ loading training data....")
    train_data, train_generate_nums, train_copy_nums = loading_data(opt.train_data_file, DA=opt.DA)
    logger.info(f"[INFO]------ loading val data....")
    val_data, val_generate_nums, val_copy_nums, answer_list = loading_data(opt.val_data_file, val_data=True)

    generate_nums = train_generate_nums + val_generate_nums
    copy_nums = max(train_copy_nums, val_copy_nums)
    print(generate_nums)

    logger.info(f"[INFO]------ loading PLM....")
    encoder = EncoderPLM(hidden_size=opt.hidden_size, auto_transformer=False, pretrain_path=opt.pretrain_path, dropout=opt.dropout)

    # torch.save(encoder.state_dict(), os.path.join(f"./{opt.output_dir}/epoch_{opt.n_epochs}_encoder.ckpt"))

    # add new token [NUM] to tokenizer
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrain_path)
    # tokens = ["[NUM]", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14", "3.14"]
    # tokenizer.add_tokens(tokens, special_tokens=True)
    special_tokens_dict = {'additional_special_tokens': ["[NUM]", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14", "3.14"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    encoder.bert_model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(opt.pretrain_path)

    output_lang, train_pairs, val_pairs = prepare_data_with_PLM(train_data, val_data, generate_nums,
                                                                 copy_nums, tokenizer=tokenizer, PLM_flag=True,
                                                                 tree=True)
    # sava lang generate_nums copy_nums hidden_size embedding_size dropout

    output_hal = open("./parameter/plm_lang.pkl", 'wb')
    str = pickle.dumps(output_lang)
    output_hal.write(str)
    output_hal.close()

    with open("./parameter/plm_other.pkl", "wb") as file:
        list1 = [generate_nums, copy_nums, opt.hidden_size, opt.embedding_size, opt.dropout]
        pickle.dump(list1, file, True)

    print(output_lang.index2word)
    # logger.info(f"{train_pairs[0]}")
    # logger.info(f"{train_pairs[1]}")
    # logger.info(f"{train_pairs[23170]}")
    # logger.info(f"{train_data[23170]}")
    # logger.info(f"{train_data[7144]}")
    # logger.info(f"{train_pairs[7144]}")
    # logger.info(f"{val_pairs[0]}")
    predict, generate, merge = init_model(opt.hidden_size, opt.embedding_size,
                                                   copy_nums, generate_nums, output_lang, opt.dropout)

    train(encoder, predict, generate, merge, train_pairs, val_pairs, opt.batch_size, opt.n_epochs, opt.learning_rate,
          generate_nums,
          output_lang, USE_CUDA, opt.schedule, opt.output_dir, opt.beam_size, opt.warmup_steps, opt.results_dir, opt.pretrain_path, answer_list)


# def eval_model(model_path, test_data_path, toeknizer_path):
#     test_data = load_data(test_data_path)
#     tokenizer = AutoTokenizer.from_pretrained(toeknizer_path)



if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())

    train_model()

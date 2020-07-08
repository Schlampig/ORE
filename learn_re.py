import os
import json
import random
import pickle
import argparse
import requests
import numpy as np
from time import time
from tqdm import tqdm
from itertools import permutations
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from bert_codes.pytorch_modeling import BertConfig, BertForQA_CLS
from bert_codes.pytorch_optimization import get_optimization, warmup_linear
import bert_codes.entity_tokenization as tokenization
import bert_codes.utils as utils
import ipdb

DATA_DIR = "pretrain_data"
MODEL_DIR = "pretrain_models"


# Configuration
##########################################################################################
t_config = time()

# set hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--load_train_path', type=str, default="your/path/to/put/train_data.json")  # your path
parser.add_argument('--load_test_path', type=str, default="your/path/to/put/test_data.json")  # your path
parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3')
parser.add_argument('--model_name', type=str, default='bert_chinese')  # used pre-trained language model name
parser.add_argument('--suffix_name', type=str, default='re')  # fine-tuned model suffix name

parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--n_batch', type=int, default=128)
parser.add_argument('--class_num', type=int, default=1)  # does relation exist between current entities? yes or no
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip_norm', type=float, default=1.0)
parser.add_argument('--warmup_rate', type=float, default=0.05)
parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--float16', type=bool, default=False)
parser.add_argument('--eval_steps', type=float, default=0.5)
parser.add_argument('--save_best', type=bool, default=True)
parser.add_argument('--vocab_size', type=int, default=21128)

parser.add_argument('--cls_weight', type=str, default=None)  # [a, b], a for pos, b for neg, None for balanced case
parser.add_argument('--max_seq_length', type=int, default=128)  # maximum sentence length
parser.add_argument('--max_lines', type=str, default=300000)  # number of lines loaded from the raw corpus
parser.add_argument('--train_split', type=str, default=0.6)  # probability to choose the sample for train, else for dev
parser.add_argument('--blank_ratio', type=str, default=0.5)  # probability to mask entity in sentence
parser.add_argument('--num_relation', type=str,
                    default=-1)  # maximum number for each relation type from the triples
parser.add_argument('--repeat_time', type=str, default=[6, 4, 5])  # repeat sampling time for each sentence

parser.add_argument('--train_dir', type=str, default=DATA_DIR)
parser.add_argument('--dev_dir', type=str, default=DATA_DIR)
parser.add_argument('--bert_config_file', type=str, default=MODEL_DIR)
parser.add_argument('--vocab_file', type=str, default=MODEL_DIR)
parser.add_argument('--init_restore_dir', type=str, default=MODEL_DIR)
parser.add_argument('--checkpoint_dir', type=str, default='check_points')
parser.add_argument('--setting_file', type=str, default='setting.txt')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--test_log', type=str, default='test_log')
args = parser.parse_args()

args.train_dir = os.path.join(args.train_dir, args.suffix_name + "_train.pkl")
args.dev_dir = os.path.join(args.dev_dir, args.suffix_name + "_dev.pkl")
args.bert_config_file = os.path.join(args.bert_config_file, args.model_name, 'bert_config.json')
args.vocab_file = os.path.join(args.vocab_file, args.model_name, 'vocab.txt')
args.init_restore_dir = os.path.join(args.init_restore_dir, args.model_name, 'pytorch_model.pth')
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name + "_" + args.suffix_name)
args.test_log += "_" + args.model_name + "_" + args.suffix_name + ".txt"
args = utils.check_args(args)

# bert initialization
bert_config = BertConfig.from_json_file(args.bert_config_file)
tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
model = BertForQA_CLS(config=bert_config, num_labels=args.class_num)

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda")
is_cuda = True
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# initialize model
print('init model...')
utils.torch_show_all_params(model)
utils.torch_init_model(model, args.init_restore_dir)  # load the saved model according to the checkpoint_dir when prediction
if args.float16:
    model.half()
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

print("Initial Configuaration Time: {}".format(time() - t_config))


# Data Preparation (in training step)
##########################################################################################
def kmp_match(s1, s2):
    def gen_next(s):
        k = -1
        n = len(s)
        m = 0
        lst = [0] * n
        lst[0] = -1
        while m < n - 1:
            if k == -1 or s[k] == s[m]:
                k += 1
                m += 1
                lst[m] = k
            else:
                k = lst[k]
        return lst

    next_list = gen_next(s2)
    ans = -1
    i = 0
    j = 0
    while i < len(s1):
        if s1[i] == s2[j] or j == -1:
            i += 1
            j += 1
        else:
            j = next_list[j]
        if j == len(s2):
            ans = i - len(s2)
            break
    return ans


def add_tag(span_ids, span_type, input_ids=None, input_ids_mask=None, blank_ratio=-1):
    # [42] for <entity_head_begin>, [43] for <entity_head_end>,
    # [44] for <entity_tail_begin>, [45] for <entity_tail_end>,
    # [13] for <blank>, [-1] for <mask_span>
    # assert
    if input_ids is None or input_ids_mask is None:
        return None
    # find begin and end indexes
    idx_begin = kmp_match(input_ids_mask, span_ids)
    if idx_begin == -1:
        return None
    idx_end = idx_begin + len(span_ids)
    # define tags
    span_tag = [-1]
    blank_tag = [13]
    blank_mode = False
    if "head" in span_type:
        fore_tag, post_tag = [42], [43]
    if "tail" in span_type:
        fore_tag, post_tag = [44], [45]
    # add tags to entity
    if "head" in span_type or "tail" in span_type:
        input_ids_mask = input_ids[:idx_begin] + span_tag * (len(span_ids) + 2) + input_ids[idx_end:]
        if random.random() > blank_ratio:  # blank_ratio == -1 means tagging without blank
            input_ids = input_ids[:idx_begin] + fore_tag + input_ids[idx_begin:idx_end] + post_tag + input_ids[idx_end:]
            blank_mode = True
        else:
            input_ids = input_ids[:idx_begin] + fore_tag + blank_tag * len(span_ids) + post_tag + input_ids[idx_end:]
    if len(input_ids_mask) != len(input_ids):
        raise ValueError("[ERROR] Lengths of input_ids and input_ids_mask should be equal.")
    return idx_begin, idx_end, input_ids, input_ids_mask, blank_mode


def get_input_ids(tokenizer, text, entity_head, entity_tail, relation=None, max_seq_length=128, blank_ratio=0.5,
                  is_check=None):
    assert isinstance(entity_head, str) and isinstance(entity_head, str)
    assert len(text) > 0 and len(entity_head) > 0 and len(entity_tail) > 0
    # get tokens
    lst_text = tokenizer.tokenize(text)
    lst_entity_head = tokenizer.tokenize(entity_head)
    lst_entity_tail = tokenizer.tokenize(entity_tail)
    # cut over-length tokens
    if len(
            lst_text) > max_seq_length - 6:  # 6指的是[CLS]、[SEP]、<entity_head_begin>、<entity_head_end>、<entity_tail_begin>、<entity_tail_end>
        lst_text = lst_text[:max_seq_length - 6]
    lst_text = ["[CLS]"] + lst_text + ["[SEP]"]
    # token to ids
    input_ids = tokenizer.convert_tokens_to_ids(lst_text)
    input_head_ids = tokenizer.convert_tokens_to_ids(lst_entity_head)
    input_tail_ids = tokenizer.convert_tokens_to_ids(lst_entity_tail)
    # initialize blank_mode
    blank_mode = [0, 0]
    # add tags according to the order
    if len(input_head_ids) >= len(input_tail_ids):
        res = add_tag(input_head_ids, "head", input_ids, input_ids, blank_ratio=blank_ratio)
        if res:
            blank_mode[0] = 1 if res[-1] else 0
            res = add_tag(input_tail_ids, "tail", res[2], res[3], blank_ratio=blank_ratio)
            if res:
                blank_mode[1] = 1 if res[-1] else 0
                input_ids, input_ids_mask = res[2], res[3]
            else:
                return None, -1, -1, blank_mode  # entity_tail not exist
        else:
            return None, -1, -1, blank_mode  # entity_head not exist
    else:
        res = add_tag(input_tail_ids, "tail", input_ids, input_ids, blank_ratio=blank_ratio)
        if res:
            blank_mode[1] = 1 if res[-1] else 0
            res = add_tag(input_head_ids, "head", res[2], res[3], blank_ratio=blank_ratio)
            if res:
                blank_mode[0] = 1 if res[-1] else 0
                input_ids, input_ids_mask = res[2], res[3]
            else:
                return None, -1, -1, blank_mode  # entity_head not exist
        else:
            return None, -1, -1, blank_mode  # entity_tail not exist
    # find relation_span
    idx_relation_begin, idx_relation_end = 0, 0
    if isinstance(relation, str) and len(relation) > 0:
        lst_relation = tokenizer.tokenize(relation)
        input_relation_ids = tokenizer.convert_tokens_to_ids(lst_relation)
        res = add_tag(input_relation_ids, "relation", input_ids, input_ids_mask, blank_ratio=blank_ratio)
        if res:
            idx_relation_begin, idx_relation_end = res[0], res[1]
        else:
            return None, -1, -1, blank_mode  # relation cannot be found
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
    if len(input_ids) > max_seq_length:
        raise ValueError("[ERROR] input_ids should be shorter than max_seq_length.")
    # check
    if is_check:
        print(is_check)
        print("text:", "".join(tokenizer.convert_ids_to_tokens(input_ids[:input_ids.index(102)])))
        print("relation:", "".join(tokenizer.convert_ids_to_tokens(input_ids[idx_relation_begin:idx_relation_end])))
    return input_ids, idx_relation_begin, idx_relation_end, blank_mode


def raw2json(tokenizer, load_path, save_path=None, max_lines=100, max_seq_length=128, train_split=0.95, print_time=100,
             blank_ratio=0.4, num_relation=-1, repeat_time=[1, 1, 1]):
    global DATA_DIR
    features_train = list()
    features_dev = list()
    unique_id = 0  # count samples
    c_pos_tuple = 0  # count positive samples without relation
    c_pos_triple = 0  # count positive samples with relation
    c_neg = 0  # count negative samples
    dict_relation = dict()  # record relation types

    with open(load_path, "r") as f:
        for i_line, line in enumerate(f):
            if i_line > max_lines:  # control the number of operated samples
                break
            line_now = json.loads(line)
            lst_samples = line_now.get("EL_res")
            for sample in lst_samples:
                # choose train or dev by probability
                is_dev = -1 if random.random() > train_split else blank_ratio  # dev samples should not be blanked
                repeat_time_now = [1, 1, 1] if is_dev == -1 else repeat_time  # dev samples could not be repeated

                # print
                if i_line % print_time == 0:
                    print("-" * 50)
                    print("* pos_triple-{} + pos_tuple-{} + neg-{} = {} samples from {} lines.".format(c_pos_triple,
                                                                                                       c_pos_tuple,
                                                                                                       c_neg, unique_id,
                                                                                                       i_line))
                # pre-processing entities
                text = sample.get("text")
                facts = sample.get("triples")
                lst_entities = list(sample.get("entity_idx").keys())  # all entities
                lst_pair = [(h, t) for h, t in permutations(lst_entities, 2) if h != t]
                lst_pos_triple = list()
                lst_pos_tuple = list()
                for fact in facts:
                    if fact[-1]:  # a pos_triple
                        lst_pos_triple.append(fact)
                        lst_pair.remove((fact[0], fact[1]))
                    elif fact[1]:  # a pos_tuple
                        lst_pos_tuple.append(fact)
                        lst_pair.remove((fact[0], fact[1]))
                    else:  # a neg entity
                        continue
                if len(lst_pos_triple) == 0:  # there is no pos_triple in this line
                    continue
                random.shuffle(lst_pair)

                # get positive triples
                n_valid_triple = 0
                for i_fact, fact in enumerate(lst_pos_triple):
                    # balance relation type
                    if num_relation > 0:  # otherwise no constraints
                        if fact[-1] in dict_relation.keys():
                            if dict_relation[fact[-1]] > num_relation:
                                continue
                            else:
                                dict_relation[fact[-1]] = dict_relation[fact[-1]] + 1
                        else:
                            dict_relation[fact[-1]] = 1
                    # calculate triples
                    check_tag = "positive_triple:" if i_fact < 5 else None
                    lst_mode = list()
                    for _ in range(repeat_time_now[0]):
                        input_ids, label_start, label_end, blank_mode = get_input_ids(tokenizer, text, fact[0], fact[1],
                                                                                      relation=fact[2],
                                                                                      max_seq_length=max_seq_length,
                                                                                      blank_ratio=is_dev,
                                                                                      is_check=check_tag)
                        if input_ids and blank_mode not in lst_mode:
                            feature = {
                                'unique_id': unique_id,
                                'input_ids': input_ids,
                                'label_start': label_start,
                                'label_end': label_end,
                                'label_class': 1}
                            if is_dev < 0:
                                features_dev.append(feature)
                            else:
                                features_train.append(feature)
                            unique_id += 1
                            c_pos_triple += 1
                            lst_mode.append(blank_mode)
                            n_valid_triple += 1
                            print("[INSERT] OK.")
                            print()
                        else:
                            print("[INSERT] FAILED.")
                            print()
                    if len(lst_mode) > 0:
                        n_valid_triple = n_valid_triple + 1 - len(lst_mode)  # delete sample triples

                # get positive tuples
                for j_fact, fact in enumerate(lst_pos_tuple):
                    if n_valid_triple == 0:
                        if random.random() > 0.5:
                            break
                    if j_fact == max(int(round(0.5 * n_valid_triple)), 1):  # pos_tuple = pos_triple
                        break
                    # calculate tuples
                    check_tag = "positive_tuple:" if i_fact < 5 else None
                    lst_mode = list()
                    for _ in range(repeat_time_now[1]):
                        input_ids, label_start, label_end, blank_mode = get_input_ids(tokenizer, text, fact[0], fact[1],
                                                                                      max_seq_length=max_seq_length,
                                                                                      blank_ratio=is_dev,
                                                                                      is_check=check_tag)
                        if input_ids and blank_mode not in lst_mode:
                            feature = {
                                'unique_id': unique_id,
                                'input_ids': input_ids,
                                'label_start': label_start,
                                'label_end': label_end,
                                'label_class': 1}
                            if is_dev < 0:
                                features_dev.append(feature)
                            else:
                                features_train.append(feature)
                            unique_id += 1
                            c_pos_tuple += 1
                            lst_mode.append(blank_mode)
                            print("[INSERT] OK.")
                            print()
                        else:
                            print("[INSERT] FAILED.")
                            print()

                # get negative tuples
                for k_fact, fact in enumerate(lst_pair):
                    if k_fact == max(int(round(0.5 * n_valid_triple)), 1):  # neg_tuple = pos_tuple + pos_triple
                        break
                    # calculate negative samples
                    check_tag = "negative_tuple:" if i_fact < 5 else None
                    lst_mode = list()
                    for _ in range(repeat_time_now[2]):
                        input_ids, label_start, label_end, blank_mode = get_input_ids(tokenizer, text, fact[0], fact[1],
                                                                                      max_seq_length=max_seq_length,
                                                                                      blank_ratio=is_dev,
                                                                                      is_check=check_tag)
                        if input_ids and blank_mode not in lst_mode:
                            feature = {
                                'unique_id': unique_id,
                                'input_ids': input_ids,
                                'label_start': label_start,
                                'label_end': label_end,
                                'label_class': 0}
                            if is_dev < 0:
                                features_dev.append(feature)
                            else:
                                features_train.append(feature)
                            unique_id += 1
                            c_neg += 1
                            lst_mode.append(blank_mode)
                            print("[INSERT] OK.")
                            print()
                        else:
                            print("[INSERT] FAILED.")
                            print()
                print("-" * 50)

    # save data
    if save_path:
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(os.path.join(DATA_DIR, save_path + "_train.pkl"), 'wb') as fw:
            pickle.dump(features_train, fw)
        with open(os.path.join(DATA_DIR, save_path + "_dev.pkl"), 'wb') as fw:
            pickle.dump(features_dev, fw)
        print("Train size-{}: Dev size-{}".format(len(features_train), len(features_dev)))
        print("Final pos_triple-{} + pos_tuple-{} + neg-{} = {} samples.".format(c_pos_triple, c_pos_tuple, c_neg,
                                                                                 unique_id))
        if dict_relation:
            print("Number of relation types-{}, and average numbers-{}".format(len(dict_relation), sum(
                [v for _, v in dict_relation.items()]) / float(len(dict_relation))))
        else:
            print("Sizes among relation types are imbalanced.")
        return True
    else:
        return features_train, features_dev


# Data Preparation (in test step)
##########################################################################################
def add_tag_test(span_ids, span_type, input_ids=None, input_ids_mask=None):
    # [42] for <entity_head_begin>, [43] for <entity_head_end>,
    # [44] for <entity_tail_begin>, [45] for <entity_tail_end>,
    # [13] for <blank>, [-1] for <mask_span>
    # assert
    if input_ids is None or input_ids_mask is None:
        return None
    # find begin and end indexes
    idx_begin = kmp_match(input_ids_mask, span_ids)
    if idx_begin == -1:
        return None
    idx_end = idx_begin + len(span_ids)
    # define tags
    span_tag = [-1]
    if "head" in span_type:
        fore_tag, post_tag = [42], [43]
    if "tail" in span_type:
        fore_tag, post_tag = [44], [45]
    # add tags to entity
    if "head" in span_type or "tail" in span_type:
        input_ids_mask = input_ids[:idx_begin] + span_tag * (len(span_ids) + 2) + input_ids[idx_end:]
        input_ids = input_ids[:idx_begin] + fore_tag + input_ids[idx_begin:idx_end] + post_tag + input_ids[idx_end:]
    if len(input_ids_mask) != len(input_ids):
        raise ValueError("[ERROR] Lengths of input_ids and input_ids_mask should be equal.")
    return idx_begin, idx_end, input_ids, input_ids_mask


def get_input_ids_test(tokenizer, text, entity_head, entity_tail, max_seq_length=128):
    assert isinstance(entity_head, str) and isinstance(entity_tail, str)
    assert len(text) > 0 and len(entity_head) > 0 and len(entity_tail) > 0
    # get tokens
    lst_text = tokenizer.tokenize(text)
    lst_entity_head = tokenizer.tokenize(entity_head)
    lst_entity_tail = tokenizer.tokenize(entity_tail)
    # cut over-length tokens
    if len(lst_text) > max_seq_length - 6:
        lst_text = lst_text[:max_seq_length - 6]
    lst_text = ["[CLS]"] + lst_text + ["[SEP]"]
    # token to ids
    input_ids = tokenizer.convert_tokens_to_ids(lst_text)
    input_head_ids = tokenizer.convert_tokens_to_ids(lst_entity_head)
    input_tail_ids = tokenizer.convert_tokens_to_ids(lst_entity_tail)
    # add tags according to the order
    output_ids = None
    if len(input_head_ids) >= len(input_tail_ids):
        res = add_tag_test(input_head_ids, "head", input_ids, input_ids)
        if res:
            res = add_tag_test(input_tail_ids, "tail", res[2], res[3])
            if res:
                output_ids = res[2]
    else:
        res = add_tag_test(input_tail_ids, "tail", input_ids, input_ids)
        if res:
            res = add_tag_test(input_head_ids, "head", res[2], res[3])
            if res:
                output_ids = res[2]
    # padding
    if output_ids:
        while len(output_ids) < max_seq_length:
            output_ids.append(0)
        if len(output_ids) > max_seq_length:
            raise ValueError("[ERROR] input_ids should be shorter than max_seq_length.")
    return output_ids


def string2token(tokenizer, text, lst_entities, max_seq_length=128):
    # text: string, the input sentence
    # lst_entities: list, the list of cadidate entities
    if len(lst_entities) <= 1 or len(text) == 0:
        return None, None
    features = list()
    max_seq_len = 0
    for entity_head, entity_tail in permutations(lst_entities, 2):
        input_ids = get_input_ids_test(tokenizer, text, entity_head, entity_tail, max_seq_length=max_seq_length)
        if input_ids:
            features.append({'input_ids': input_ids})
            if len(input_ids) > max_seq_len:
                max_seq_len = len(input_ids)
    return features, max_seq_len


# Batch Generation
##########################################################################################
class GenData(object):
    def __init__(self, batch_size, is_cuda, data_dir, is_train=True):
        with open(os.path.join(data_dir), "rb") as f:
            self.all_data = pickle.load(f)
        self.batch_size = batch_size
        self.is_train = is_train
        self.cuda = is_cuda
        self.data = GenData.make_baches(self.all_data, self.batch_size, self.is_train)
        self.offset = 0

    @staticmethod
    def make_baches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
            return [data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                          :i + batch_size - len(
                                                                                              data)] for i in
                    range(0, len(data), batch_size)]
        # 确保多gpu推断正常运作
        return [data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(
                                                                                          data)] for i in
                range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            self.data = GenData.make_baches(self.all_data, self.batch_size, self.is_train)
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            bsz = len(batch)
            max_seq_len = max([len(sample['input_ids']) for sample in batch])  # 每个batch中sequence长度对齐
            # passage inputs
            input_ids = torch.LongTensor([sample['input_ids'] for sample in batch])[:, :max_seq_len]
            input_mask = torch.LongTensor([[1] * len(sample['input_ids']) for sample in batch])[:, :max_seq_len]
            input_segments = torch.LongTensor([[0] * len(sample['input_ids']) for sample in batch])[:,
                             :max_seq_len]
            label_start = torch.LongTensor([sample['label_start'] for sample in batch])
            label_end = torch.LongTensor([sample['label_end'] for sample in batch])
            label_class = torch.FloatTensor([sample['label_class'] for sample in batch])
            out_batch = {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "input_segments": input_segments,
                "label_start": label_start,
                "label_end": label_end,
                "label_class": label_class
            }
            if self.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()
            yield out_batch


def gen_batch_test(input_data, max_seq_len, is_cuda):
    input_ids = torch.LongTensor([sample['input_ids'] for sample in input_data])[:, :max_seq_len]
    input_mask = torch.LongTensor([[1] * len(sample['input_ids']) for sample in input_data])[:,
                 :max_seq_len]
    input_segments = torch.LongTensor([[0] * len(sample['input_ids']) for sample in input_data])[:,
                     :max_seq_len]
    out_batch = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "input_segments": input_segments}
    if is_cuda:
        for k in out_batch.keys():
            if isinstance(out_batch[k], torch.Tensor):
                out_batch[k] = out_batch[k].cuda()
    return out_batch


# Training (in training step)
##########################################################################################
def train(args=None, tokenizer=None, model=None, is_cuda=None, n_gpu=None):
    # set data generators for train and dev
    train_data_gen = GenData(args.n_batch, is_cuda, args.train_dir, is_train=True)
    dev_data_gen = GenData(args.n_batch, is_cuda, args.dev_dir, is_train=False)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    # set training steps
    steps_per_epoch = len(train_data_gen)
    args.eval_steps = int(args.eval_steps * steps_per_epoch)
    total_steps = steps_per_epoch * args.train_epochs
    print("steps per epoch: {}; total steps: {}; warmup steps: {}"
          .format(steps_per_epoch, total_steps, int(args.warmup_rate * total_steps)))

    # set optimizer
    optimizer = get_optimization(model=model, float16=args.float16, learning_rate=args.lr, total_steps=total_steps,
                                 schedule=args.schedule, warmup_rate=args.warmup_rate, max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate)

    print('***** Training *****')
    global_steps = 1
    best_f1 = 0
    best_acc = 0
    cls_weight_now = torch.Tensor(args.cls_weight).cuda() if args.cls_weight else None
    for i in range(int(args.train_epochs)):
        print('Starting epoch {}'.format(i + 1))
        model.train()
        train_data_gen.reset()
        total_loss = 0
        iteration = 1
        with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1), ncols=50) as pbar:
            for step, batch in enumerate(train_data_gen):
                loss = model(input_ids=batch['input_ids'],
                             token_type_ids=batch['input_segments'],
                             attention_mask=batch['input_mask'],
                             start_positions=batch['label_start'],
                             end_positions=batch['label_end'],
                             target_labels=batch['label_class'],
                             cls_weight=cls_weight_now)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                total_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                pbar.update(1)

                if args.float16:
                    optimizer.backward(loss)
                    lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    loss.backward()

                optimizer.step()
                model.zero_grad()
                global_steps += 1
                iteration += 1

                if global_steps % args.eval_steps == 0:
                    f1, acc = evaluate(tokenizer, model, dev_data_gen)
                    with open(args.log_file, 'a') as aw:
                        aw.write('global steps:{}, f1:{}, acc:{}'
                                 .format(global_steps, f1, acc) + '\n')
                    print('global steps:{}, f1:{}, acc:{}'.format(global_steps, f1, acc))
                    if f1 > best_f1 or acc > best_acc:
                        if f1 > best_f1:
                            best_f1 = f1
                        if acc > best_acc:
                            best_acc = acc
                        utils.torch_save_model(model, args.checkpoint_dir, {'f1': f1, 'acc': acc}, max_save_num=1)
                    model.train()
    return None


# Evaluation (in training step)
##########################################################################################
def evaluate(tokenizer, model, dev_data_gen):
    print("***** Eval *****")
    model.eval()
    dev_data_gen.reset()
    f1_all = 0.0
    acc_all = 0.0
    with torch.no_grad():
        for i_batch, batch in enumerate(dev_data_gen):
            start_logits, end_logits, target_logits = model(input_ids=batch['input_ids'],
                                                            token_type_ids=batch['input_segments'],
                                                            attention_mask=batch['input_mask'])
            # get predicted labels
            start_logits = start_logits.detach().cpu().numpy()  # [bs, len]
            end_logits = end_logits.detach().cpu().numpy()  # [bs, len]
            start_pre = np.argmax(start_logits, axis=-1)  # [bs,]
            end_pre = np.argmax(end_logits, axis=-1)  # [bs,]
            class_pre = target_logits.detach().cpu().numpy()  # [bs,]
            class_pre[class_pre <= 0.5] = 0
            class_pre[class_pre > 0] = 1
            # get true labels
            start_true = batch['label_start'].detach().cpu().numpy()
            end_true = batch['label_end'].detach().cpu().numpy()
            class_true = batch['label_class'].detach().cpu().numpy()
            # calculate each sample in the batch
            f1 = 0.0
            batch_size = target_logits.shape[0]
            for i in range(batch_size):
                lst_text = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].detach().cpu().tolist())
                # get predicted or true relations
                if start_true[i] == end_true[i]:  # non-span
                    if start_pre[i] == start_true[i] and end_pre[i] == end_true[i]:
                        f1 += 1.0
                else:  # span
                    relation_true = lst_text[start_true[i]:end_true[i]]
                    relation_pre = lst_text[start_pre[i]:end_pre[i]]
                    print("compare(true/pre): {} / {}".format(relation_true, relation_pre))
                    # calculate metrics
                    correct = len(set(relation_true).intersection(set(relation_pre)))
                    precision = correct / (len(relation_pre) + 1e-5)
                    recall = correct / (len(relation_true) + 1e-5)
                    f1 += (2 * precision * recall) / (precision + recall + 1e-5)
            f1_all += f1 / batch_size * 100.0  # f1 and acc of the current batch
            acc_all += f1_score(class_true, class_pre, average="macro") * 100.0
            print("{} batch, f1-{:.4f}, acc-{:.4f}.".format(i_batch, f1 / batch_size,
                                                            f1_score(class_true, class_pre, average="macro")))
        # get all f1 and accuracy
        f1_all = f1_all / (i_batch + 1)
        acc_all = acc_all / (i_batch + 1)
        print("f1_all-{:.4f}, acc_all-{:.4f}".format(f1_all, acc_all))
    return f1_all, acc_all


# Prediction (in test step)
##########################################################################################
def predict_ner(doc):
    headers = {"Content-Type": "application/json"}
    url = "your-ner-model-url"
    text = {"text": doc}
    result = requests.request("POST", url, json=text, headers=headers)
    lst_doc = result.json()
    return lst_doc


def id2token(tokenizer, lst_ids):
    lst_text = tokenizer.convert_ids_to_tokens(lst_ids)
    lst_head = lst_text[lst_text.index("[unused42]"):lst_text.index("[unused43]")]
    lst_tail = lst_text[lst_text.index("[unused44]"):lst_text.index("[unused45]")]
    lst_head, lst_tail = lst_head[1:], lst_tail[1:]
    return lst_text, "".join(lst_head), "".join(lst_tail)


# predict each input sentence
def predict_span(tokenizer, model, input_data_gen):
    lst_pre = list()
    model.eval()
    with torch.no_grad():
        start_logits, end_logits, target_logits = model(input_ids=input_data_gen['input_ids'],
                                                        token_type_ids=input_data_gen['input_segments'],
                                                        attention_mask=input_data_gen['input_mask'])
        # post-processing
        start_logits = start_logits.detach().cpu().numpy()  # [bs, len]
        end_logits = end_logits.detach().cpu().numpy()  # [bs, len]
        start_pre = np.argmax(start_logits, axis=-1)  # [bs,]
        end_pre = np.argmax(end_logits, axis=-1)  # [bs,]
        class_pre = target_logits.detach().cpu().numpy()  # [bs,]

        # calculate each test sample
        for i in range(target_logits.shape[0]):
            lst_ids = input_data_gen["input_ids"][i].detach().cpu().tolist()
            lst_text, head_now, tail_now = id2token(tokenizer, lst_ids)
            class_pre_now = 1 if class_pre[i] > 0.5 else 0
            relation_now = None
            if class_pre_now == 1:  # 两个候选实体有关系，lst_pre只存有关系的三元组/二元组
                if start_pre[i] != end_pre[i]:
                    relation_now = "".join(lst_text[start_pre[i]:end_pre[i]])  # 两个候选实体有关系，且关系有span
                lst_pre.append((head_now, tail_now, relation_now))
        return lst_pre


def predict_now(doc, args=None, tokenizer=None, model=None, is_cuda=True, is_print=False):
    assert args and tokenizer and model
    print("***** Predict *****")
    t_predict = time()
    pre_all = list()
    print("NER model ...")
    lst_doc = predict_ner(doc)
    print("Classify model ...")
    for piece in lst_doc:
        input_data, max_seq_len = string2token(tokenizer, text=piece["text"], lst_entities=piece["entity"],
                                               max_seq_length=args.max_seq_length)
        
        if input_data:
            input_data_gen = gen_batch_test(input_data, max_seq_len, is_cuda)
            pre_piece = predict_span(tokenizer, model, input_data_gen)
            if len(pre_piece) > 0:
                pre_all.extend(pre_piece)
    print("predicting time: {}".format(time() - t_predict))
    if is_print:
        print(doc)
        for fact in pre_all:
            if fact[2]:
                print(fact[0], " ", fact[2], " ", fact[1])
            else:
                print(fact[0], " 和 ", fact[1], " 有关系")
    return pre_all


def predict_one(sentence, args=None, tokenizer=None, model=None, is_cuda=True, is_print=True):
    assert args and tokenizer and model
    assert isinstance(sentence, str)
    label_pre = predict_now(sentence, args, tokenizer, model, is_cuda, is_print=is_print)
    return label_pre


# Measurement (in test step)
##########################################################################################
def get_label(lst):
    lst_span = list()
    lst_classify = list()
    for i in lst:
        lst_classify.append(str(i[0]) + "_" + str(i[1]))
        if len(i) == 3 and i[2]:
            lst_span.append(str(i[0]) + "_" + str(i[1]) + "_" + str(i[2]))
        else:
            lst_span.append(str(i[0]) + "_" + str(i[1]) + "_NONE")
    lst_span = set(lst_span)
    lst_classify = set(lst_classify)
    return lst_span, lst_classify


def get_metric(label_pre, label_true, sample, args):
    span_pre, classify_pre = get_label(label_pre)
    span_true, classify_true = get_label(label_true)
    span_inter = set.intersection(span_pre, span_true)
    classify_inter = set.intersection(classify_pre, classify_true)

    # record
    with open(args.test_log, "a") as f_log:
        if len(span_inter) == 0 or len(classify_inter) == 0:
            f_log.write("【EXTRACTION FAILED】\ns")
        f_log.write("Text: " + sample.get("text", "") + "\n")
        f_log.write("Pre: " + "，".join(list(span_pre)) + "\n")
        f_log.write("True: " + "，".join(list(span_true)) + "\n")
        f_log.write("Intersection Span: " + "，".join(list(span_inter)) + "\n")
        f_log.write("Intersection Classify: " + "，".join(list(classify_inter)) + "\n")
        f_log.write("\n")

    span_recall = len(span_inter) / (len(span_true) + 1e-5)
    span_precision = len(span_inter) / (len(span_pre) + 1e-5)
    span_f1 = (2 * span_precision * span_recall) / (span_precision + span_recall + 1e-5)

    classify_recall = len(classify_inter) / (len(classify_true) + 1e-5)
    classify_precision = len(classify_inter) / (len(classify_pre) + 1e-5)
    classify_f1 = (2 * classify_precision * classify_recall) / (classify_precision + classify_recall + 1e-5)

    res = {"span_precision": span_precision,
           "span_recall": span_recall,
           "span_f1": span_f1,
           "classify_recall": classify_recall,
           "classify_precision": classify_precision,
           "classify_f1": classify_f1}

    return res


def predict_all(load_path, args=None, tokenizer=None, model=None, is_cuda=True):
    assert args and tokenizer and model
    assert load_path.endswith(".json")
    with open(load_path, "rb") as f:
        d = json.load(f)
        span_f1, span_recall, span_precision = 0.0, 0.0, 0.0
        classify_f1, classify_recall, classify_precision = 0.0, 0.0, 0.0
        count_sample = 0
        for sample in d:
            if sample.get("text", None):
                print("The {}th sample ...".format(count_sample))
                label_pre = predict_now(sample.get("text", None), args, tokenizer, model, is_cuda)
                label_true = sample.get("triples", None)
                res_now = get_metric(label_pre, label_true, sample, args)
                span_f1 += res_now["span_f1"]
                span_recall += res_now["span_recall"]
                span_precision += res_now["span_precision"]
                classify_f1 += res_now["classify_f1"]
                classify_recall += res_now["classify_recall"]
                classify_precision += res_now["classify_precision"]
                count_sample += 1
    span_f1 /= float(count_sample)
    span_recall /= float(count_sample)
    span_precision /= float(count_sample)
    classify_f1 /= float(count_sample)
    classify_recall /= float(count_sample)
    classify_precision /= float(count_sample)

    # record
    with open(args.test_log, "a") as f_log:
        f_log.write("Span-F1: %s \n" % str(span_f1))
        f_log.write("Span-Recall:  %s \n" % str(span_recall))
        f_log.write("Span-Precision:  %s \n" % str(span_precision))
        f_log.write("Classify-F1:  %s \n" % str(classify_f1))
        f_log.write("Classify-Recall:  %s \n" % str(classify_recall))
        f_log.write("Classify-Precision:  %s \n" % str(classify_precision))
        f_log.write("\n")

    return span_f1, span_recall, span_precision, classify_f1, classify_recall, classify_precision, count_sample - 1


# Main
##########################################################################################
if __name__ == "__main__":
    s = """赛尔提是本作主角，来自爱尔兰的无头骑士，性别常被认错，但确实为女性。赛尔提本来是抱着头、驾着无头马的妖精。
               赛尔提乘坐的黑摩托车，是一匹马变形而成的。二十多年前，岸谷森严使用妖刀罪歌得到她的头。
               陷于迷茫的她为了找回头于是离开爱尔兰追到了日本池袋。
               赛尔提来到池袋后平时是在作运输、保镖之类的工作，并成为当地有名的都市传说。
               赛尔提在渡船上遇上新罗父子，结识后住进了他们家中，就这样与新罗同居至今。
               赛尔提喜欢新罗，是DOLLARS的一员，少数知道首领身份的人。赛尔提是羽岛幽平和圣边琉璃的粉丝。"""

    # create train & dev samples from raw data
    raw2json(tokenizer,
             load_path=args.load_train_path,
             save_path=args.suffix_name,
             max_lines=args.max_lines,
             max_seq_length=args.max_seq_length,
             train_split=args.train_split,
             blank_ratio=args.blank_ratio,
             num_relation=args.num_relation,
             repeat_time=args.repeat_time)

    # train & evaluate model
    train(args=args, tokenizer=tokenizer, model=model, is_cuda=is_cuda, n_gpu=n_gpu)

    # predict only one sample:
    result = predict_one(s, args=args, tokenizer=tokenizer, model=model, is_cuda=is_cuda)
    for i in result:
        print(i)

    # predict samples:
    results = predict_all(load_path=args.load_test_path,
                          args=args,
                          tokenizer=tokenizer,
                          model=model,
                          is_cuda=is_cuda)
    print("Result of Model {} on {} test samples: Span-F1-{},R-{},P-{} | Classify-F1-{},R-{},P-{}"
          .format(args.model_name + "_" + args.suffix_name,
                  results[6], results[0], results[1], results[2], results[3], results[4], results[5]))

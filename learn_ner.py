import os
import re
import json
import random
import pickle
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import torch
from bert_codes.pytorch_modeling import BertConfig, BertForTokenClassification
from bert_codes.pytorch_optimization import get_optimization, warmup_linear
import bert_codes.entity_tokenization as tokenization
import bert_codes.utils as utils
import ipdb

DATA_DIR = "pretrain_data"
MODEL_DIR = "pretrain_models"

# Server Settings
##########################################################################################
"""You can set NER model as a server (interface) for the subsequent relation-extraction process"""


# Configuration
##########################################################################################
t_config = time()

# set hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--load_train_path', type=str, default="your/path/to/put/train_data.json")  # your path
parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3')
parser.add_argument('--model_name', type=str, default='bert_chinese')  # used pre-trained language model name
parser.add_argument('--suffix_name', type=str, default='ner')  # fine-tuned model suffix name

parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--n_batch', type=int, default=128)
parser.add_argument('--class_num', type=int, default=3)  # B, I, O for NER
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
parser.add_argument('--max_lines', type=str, default=10000)  # number of lines loaded from the raw corpus
parser.add_argument('--train_split', type=str, default=0.6)  # probability to choose the sample for train, else for dev

parser.add_argument('--train_dir', type=str, default=DATA_DIR)
parser.add_argument('--dev_dir', type=str, default=DATA_DIR)
parser.add_argument('--bert_config_file', type=str, default=MODEL_DIR)
parser.add_argument('--vocab_file', type=str, default=MODEL_DIR)
parser.add_argument('--init_restore_dir', type=str, default=MODEL_DIR)
parser.add_argument('--checkpoint_dir', type=str, default='check_points')
parser.add_argument('--setting_file', type=str, default='setting.txt')
parser.add_argument('--log_file', type=str, default='log.txt')
args = parser.parse_args()

args.train_dir = os.path.join(args.train_dir, args.suffix_name + "_train.pkl")
args.dev_dir = os.path.join(args.dev_dir, args.suffix_name + "_dev.pkl")
args.bert_config_file = os.path.join(args.bert_config_file, args.model_name, 'bert_config.json')
args.vocab_file = os.path.join(args.vocab_file, args.model_name, 'vocab.txt')
args.init_restore_dir = os.path.join(args.init_restore_dir, args.model_name, 'pytorch_model.pth')
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name + "_" + args.suffix_name)
args = utils.check_args(args)

# bert initialization
bert_config = BertConfig.from_json_file(args.bert_config_file)
tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
model = BertForTokenClassification(config=bert_config, num_labels=args.class_num)

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
utils.torch_init_model(model, args.init_restore_dir)
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


def get_ner_labels(tokenizer, text, lst_entity, max_seq_length=128):
    assert isinstance(text, str) and isinstance(lst_entity, list)
    assert len(text) > 0 and len(lst_entity) > 0
    tag_neg, tag_b, tag_i = 0, 1, 2
    # pre-process text
    lst_text = tokenizer.tokenize(text)
    if len(lst_text) > max_seq_length - 2:  # 2指的是[CLS]、[SEP]
        lst_text = lst_text[:max_seq_length - 2]
    lst_text = ["[CLS]"] + lst_text + ["[SEP]"]
    lst_text = tokenizer.convert_tokens_to_ids(lst_text)  # token to ids
    # pre-process lst_entity (remove overlapped entity)
    lst_entity.sort(key=lambda x: len(x))  # sort entities according to their lengths
    i_e = 0
    while i_e < len(lst_entity):
        for j_e in lst_entity[i_e + 1:]:
            if lst_entity[i_e] in j_e:
                lst_entity.remove(j_e)
        i_e += 1
    # annotating ner tags
    lst_tag = [tag_neg] * len(lst_text)
    for entity in lst_entity:
        lst_e = tokenizer.tokenize(entity)
        lst_e = tokenizer.convert_tokens_to_ids(lst_e)  # token to ids
        lst_s = lst_text  # lst_s is temporary
        lst_begin = list()
        n_e = len(lst_e)
        while len(lst_s) >= n_e:
            idx_now = kmp_match(lst_s, lst_e)
            if idx_now >= 0:
                lst_begin.append(idx_now)
                lst_s = lst_s[idx_now + n_e:]
            else:
                break
        for idx_begin in lst_begin:
            lst_tag[idx_begin] = tag_b
            lst_tag[idx_begin + 1:idx_begin + n_e] = [tag_i] * (n_e - 1)
    num_entity = sum(1 for i in lst_tag if i == tag_b)
    if num_entity == 0:  # no entity found in the text
        return None, None, None
    else:
        # padding
        while len(lst_text) < max_seq_length:
            lst_text.append(0)
            lst_tag.append(tag_neg)
        if len(lst_text) > max_seq_length:
            raise ValueError("[ERROR] input_ids should be shorter than max_seq_length.")
        return lst_text, lst_tag, num_entity  # input_ids, input_tags, number of tagged entities


def print_ner(tokenizer, input_ids, input_tags, text, lst_entity_true):
    lst_entity = list()
    lst_entity_now = list()
    for i, tag in enumerate(input_tags):
        if tag == 0:
            if len(lst_entity_now) > 0:
                s_now = "".join(tokenizer.convert_ids_to_tokens(lst_entity_now))
                lst_entity.append(s_now)
                lst_entity_now = list()
            else:
                continue
        else:
            lst_entity_now.append(input_ids[i])
    print("original text: {}".format(text))
    print("true entities: {}".format(lst_entity_true))
    print("tagged entities: {}".format(lst_entity))
    print()
    return None


def raw2json(tokenizer, load_path, save_path=None, max_lines=100, train_split=0.95, print_time=100):
    global DATA_DIR
    features_train = list()
    features_dev = list()
    unique_id = 0  # count samples
    c_entity = 0  # count entitiess

    with open(load_path, "r") as f:
        for i_line, line in enumerate(f):
            if i_line > max_lines:  # control the number of operated samples
                break
            if i_line % print_time == 0:  # print
                print("-" * 50)
                print("* {} entities of {} samples from {} lines.".format(c_entity, unique_id, i_line))
            line_now = json.loads(line)
            lst_samples = line_now.get("EL_res")
            for sample in lst_samples:
                # pre-processing entities
                text = sample.get("text")
                lst_entity = list(sample.get("entity_idx").keys())  # all entities
                input_ids, input_tags, num_entity = get_ner_labels(tokenizer, text, lst_entity)
                if input_ids:
                    feature = {
                        'unique_id': unique_id,
                        'input_ids': input_ids,
                        'input_tags': input_tags}
                    if random.random() > train_split:  # choose train or dev by probability
                        features_dev.append(feature)
                    else:
                        features_train.append(feature)
                    unique_id += 1
                    c_entity += num_entity
                    print_ner(tokenizer, input_ids, input_tags, text, lst_entity)
    # save data
    if save_path:
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(os.path.join(DATA_DIR, save_path + "_train.pkl"), 'wb') as fw:
            pickle.dump(features_train, fw)
        with open(os.path.join(DATA_DIR, save_path + "_dev.pkl"), 'wb') as fw:
            pickle.dump(features_dev, fw)
        print("Train size-{}: Dev size-{}".format(len(features_train), len(features_dev)))
        print("Final {} entities/ {} samples.".format(c_entity, unique_id))
        return True
    else:
        return features


# Data Preparation (in test step)
##########################################################################################
def text_split(text):
    # split text into sentences
    lst_text = list()
    s = ""
    for i in text:
        if i in ["。", "？", "！", "……", "…", "；", "?", "!", ";", "|"]:
            if len(s) > 0:
                s += i
                lst_text.append(s)
                s = ""
            else:
                continue
        else:
            s += i
    return lst_text


def get_inputs_ids_ner_test(tokenizer, text, max_seq_length=128):
    if (not isinstance(text, str)) or (len(text) == 0):
        return None
    # pre-process text
    input_ids = tokenizer.tokenize(text)
    if len(input_ids) > max_seq_length - 2:  # 2指的是[CLS]、[SEP]
        input_ids = input_ids[:max_seq_length - 2]
    input_ids = ["[CLS]"] + input_ids + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_ids)  # token to ids
    while len(input_ids) < max_seq_length:  # padding
            input_ids.append(0)
    return input_ids


def string2token(tokenizer, text, max_seq_length=128):
    # text: string, the input sentence
    # lst_entities: list, the list of cadidate entities
    if (not isinstance(text, str)) or (len(text) == 0):
        return None, None
    features = list()
    max_seq_len = 0
    lst_text = text_split(text)  # long string could be split into short sentences
    for text_now in lst_text:
        input_ids = get_inputs_ids_ner_test(tokenizer=tokenizer, text=text_now, max_seq_length=max_seq_length)
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
            return [data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)] for i in range(0, len(data), batch_size)]
        # 确保多gpu推断正常运作
        return [data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)] for i in range(0, len(data), batch_size)]

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
            input_segments = torch.LongTensor([[0] * len(sample['input_ids']) for sample in batch])[:, :max_seq_len]
            input_tags = torch.LongTensor([sample['input_tags'] for sample in batch])[:, :max_seq_len]
            out_batch = {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "input_segments": input_segments,
                "input_tags": input_tags
            }
            if self.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()
            yield out_batch


def gen_batch_test(input_data, max_seq_len, is_cuda):
    input_ids = torch.LongTensor([sample['input_ids'] for sample in input_data])[:, :max_seq_len]
    input_mask = torch.LongTensor([[1] * len(sample['input_ids']) for sample in input_data])[:, :max_seq_len]
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
def train(args=None, model=None, is_cuda=None, n_gpu=None):
    # set data generators for train and dev
    train_data_gen = GenData(args.n_batch, is_cuda, args.train_dir, is_train=True)
    dev_data_gen = GenData(args.n_batch, is_cuda, args.dev_dir, is_train=False)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    steps_per_epoch = len(train_data_gen)
    args.eval_steps = int(args.eval_steps * steps_per_epoch)
    total_steps = steps_per_epoch * args.train_epochs
    print("steps per epoch: {}; total steps: {}; warmup steps: {}"
          .format(steps_per_epoch, total_steps, int(args.warmup_rate * total_steps)))

    # set optimizer
    optimizer = get_optimization(model=model,
                                 float16=args.float16,
                                 learning_rate=args.lr,
                                 total_steps=total_steps,
                                 schedule=args.schedule,
                                 warmup_rate=args.warmup_rate,
                                 max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate)

    print('***** Training *****')
    global_steps = 1
    best_f1 = 0
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
                             labels=batch['input_tags'])
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
                    f1 = evaluate(model, dev_data_gen)
                    with open(args.log_file, 'a') as aw:
                        aw.write('global steps:{}, f1:{}'.format(global_steps, f1) + '\n')
                    print('global steps:{}, f1:{}'.format(global_steps, f1))
                    if f1 > best_f1:
                        best_f1 = f1
                        utils.torch_save_model(model, args.checkpoint_dir, {'f1': f1}, max_save_num=1)
                    model.train()
    return None


# Evaluation (in training step)
##########################################################################################
def get_entity_set(lst, tag_b=1, tag_i=2):
    # lst = [0,0,0,1,2,2,0,0,1,2,0,0] -> lst_new = [[3,4,5], [8,9]] -> set_new = {"3_4_5", "8_9"}
    s_rule = str(tag_b) + str(tag_i) + "*"  # regexp = tag_btag_i*
    res = re.finditer(s_rule, "".join([str(i) for i in lst]))
    res = list(res)
    lst_new = list()
    for i in res:
        lst_new.append("_".join([str(j) for j in list(range(i.start(), i.end()))]))
    set_new = set(lst_new)
    return set_new


def evaluate(model, dev_data_gen):
    print("***** Eval *****")
    model.eval()
    dev_data_gen.reset()
    f1_all = 0.0
    with torch.no_grad():
        for i_batch, batch in enumerate(dev_data_gen):
            pre_batch = model(input_ids=batch['input_ids'],
                              token_type_ids=batch['input_segments'],
                              attention_mask=batch['input_mask'])
            # get predicted labels
            pre_batch = pre_batch.detach().cpu().numpy()  # [bs, len, dim]
            pre_batch = np.argmax(pre_batch, axis=-1)  # [bs,len]

            # get true labels
            true_batch = batch['input_tags'].detach().cpu().numpy()

            # calculate each sample in the batch
            f1 = 0.0
            batch_size = true_batch.shape[0]
            for i in range(batch_size):
                set_true = get_entity_set(true_batch[i])
                set_pre = get_entity_set(pre_batch[i])
                correct = len(set.intersection(set_true, set_pre))
                precision = correct / (len(set_pre) + 1e-5)
                recall = correct / (len(set_true) + 1e-5)
                f1 += (2 * precision * recall) / (precision + recall + 1e-5)
            f1_all += f1 / batch_size * 100.0  # f1 of the current batch
            print("{} batch, f1-{:.4f}.".format(i_batch, f1 / batch_size))
        # get all f1 and accuracy
        f1_all = f1_all / (i_batch + 1)
        print("f1_all-{:.4f}".format(f1_all))
    return f1_all


# Prediction (in test step)
##########################################################################################
def get_entity_tuple(lst, tag_b=1, tag_i=2):
    # lst = [0,0,0,1,2,2,0,0,1,2,0,0] -> lst_new = [(3,5), (8,9)]
    s_rule = str(tag_b) + str(tag_i) + "*"  # regexp = tag_btag_i*
    res = re.finditer(s_rule, "".join([str(i) for i in lst]))
    res = list(res)
    lst_tuple = list()
    for i in res:
        lst_tuple.append((i.start(), i.end()))
    return lst_tuple


def predict_now(doc, args=None, tokenizer=None, model=None, is_cuda=True, is_print=False):
    assert args and tokenizer and model
    print("***** Predict *****")
    # pre-process
    input_data, max_seq_len = string2token(tokenizer, doc, max_seq_length=128)
    if input_data:
        input_data_gen = gen_batch_test(input_data, max_seq_len, is_cuda)
    else:
        return None
    # predict
    pre_all = list()
    model.eval()
    with torch.no_grad():
        pre_batch = model(input_ids=input_data_gen['input_ids'],
                    token_type_ids=input_data_gen['input_segments'],
                    attention_mask=input_data_gen['input_mask'])
        # get predicted labelss
        pre_batch = pre_batch.detach().cpu().numpy()  # [bs, len, dim]
        pre_batch = np.argmax(pre_batch, axis=-1) # [bs,len]
        # calculate each test sample
        for i in range(pre_batch.shape[0]):
            input_ids_now = input_data_gen['input_ids'][i].detach().cpu().tolist()
            pre_tags_now = pre_batch[i]
            lst_entity = list()
            for begin, end in get_entity_tuple(pre_tags_now):
                entity_now = "".join(tokenizer.convert_ids_to_tokens(input_ids_now[begin:end])).replace("[PAD]", "").replace("[SEP]", "").replace("[CLS]", "")
                if len(entity_now) >= 2:
                    lst_entity.append(entity_now)
            lst_entity = list(set(lst_entity))
            text_now = "".join(tokenizer.convert_ids_to_tokens(input_ids_now)).replace("[PAD]", "").replace("[SEP]", "").replace("[CLS]", "")
            pre_all.append({"text": text_now, "entity": lst_entity})
    # print results
    if is_print:
        for pre_now in pre_all:
            print("Text: {}".format(pre_now["text"]))
            print("Entities: {}".format("; ".join([e for e in pre_now["entity"]]).rstrip("; ")))
            print()
    return pre_all


def predict_one(doc, args=None, tokenizer=None, model=None, is_cuda=True, is_print=False):
    assert args and tokenizer and model
    assert isinstance(doc, str)
    label_pre = predict_now(doc, args, tokenizer, model, is_cuda, is_print=is_print)
    return label_pre


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
             train_split=args.train_split)

    # train & evaluate model
    train(args=args, model=model, is_cuda=is_cuda, n_gpu=n_gpu)

    # predict only one sample:
    result = predict_one(s, args=args, tokenizer=tokenizer, model=model, is_cuda=is_cuda)
    for i in result:
        print(i)
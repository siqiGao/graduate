import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def argmax(vec):
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]  # max_score的维度为1
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # 维度为1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 转移矩阵，transitions[i][j]表示从label_j转移到label_i的概率,虽然是随机生成的但是后面会迭代更新
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 从任何标签转移到START_TAG不可能
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 从STOP_TAG转移到任何标签不可能

        self.hidden = self.init_hidden()  # 随机初始化LSTM的输入(h_0, c_0)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        '''
        输入：发射矩阵(emission score)，实际上就是LSTM的输出——sentence的每个word经BiLSTM后，对应于每个label的得分
        输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        '''
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas
        for feat in feats:  # w_i
            alphas_t = []
            for next_tag in range(self.tagset_size):  # tag_j
                # t时刻tag_i emission score（1个）的广播。需要将其与t-1时刻的5个previous_tags转移到该tag_i的transition scors相加
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # 1*5
                # t-1时刻的5个previous_tags到该tag_i的transition scors
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*5

                next_tag_var = forward_var + trans_score + emit_score
                # 求和，实现w_(t-1)到w_t的推进
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)  # 1*5

        # 最后将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        '''
        输入：id化的自然语言序列
        输出：序列中每个字符的Emission Score
        '''
        self.hidden = self.init_hidden()  # (h_0, c_0)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # len(s)*5
        return lstm_feats

    def _score_sentence(self, feats, tags):
        '''
        输入：feats——emission scores；tags——真实序列标注，以此确定转移矩阵中选择哪条路径
        输出：真实路径得分
        '''
        score = torch.zeros(1)
        # 将START_TAG的标签３拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
                best_tag_id = argmax(next_tag_var)  # 返回最大值对应的那个tag
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 无需返回最开始的START位
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):  # 损失函数
        feats = self._get_lstm_features(sentence)  # len(s)*5
        forward_score = self._forward_alg(feats)  # 规范化因子/配分函数
        gold_score = self._score_sentence(feats, tags)  # 正确路径得分
        return forward_score - gold_score  # Loss（已取反）

    def forward(self, sentence):
        '''
        解码过程，维特比解码选择最大概率的标注路径
        '''
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def build_corpus( make_vocab=True, data_dir="./ResumeNER"):
    """读取数据"""


    word_lists = []
    tag_lists = []
    with open('yuliao.txt', 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            print(line)
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

print()
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id
def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists



###################
from sklearn_crfsuite import CRF





def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


##################
import torch
import torch.nn.functional as F

# ******** CRF 工具函数*************


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


# ******** LSTM模型 工具函数*************

def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def cal_loss(logits, targets, tag2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L]
        lengths: [B]
    """
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss

# FOR BiLSTM-CRF


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    # 计算golden_scores方法２：利用pack_padded_sequence函数
    # targets[targets == end_id] = pad_id
    # scores_at_targets = torch.gather(
    #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
    # scores_at_targets, _ = pack_padded_sequence(
    #     scores_at_targets, lengths-1, batch_first=True
    # )
    # golden_scores = scores_at_targets.sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return

import torch
import torch.nn as nn
import torch.optim as optim

train_word_lists, train_tag_lists, word2id, tag2id = build_corpus()
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
print(train_word_lists, train_tag_lists, word2id, tag2id)
train_word_lists_2id = [[word2id[j] for j in i]for i in train_word_lists]
train_tag_lists_2id = [[tag2id[j] for j in i]for i in train_tag_lists]
print(train_word_lists_2id, train_tag_lists_2id)
########################
data_raw = [(train_word_lists_2id[i], train_tag_lists_2id[i]) for i in range(len(train_word_lists_2id))]
print(data_raw)


START_TAG = "<start>"
STOP_TAG = "<end>"
EMBEDDING_DIM = 15  # 由于标签一共有B\I\O\START\STOP ....
HIDDEN_DIM = 4  # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2
###########################


############################
# training_data = [(
#     ['昆', '明', '长', '水', '机', '场', '有', '多', '大', '<end>'],
#     ['B-ENTITY', 'I-ENTITY', 'I-ENTITY', 'I-ENTITY', 'I-ENTITY', 'I-ENTITY', 'B-REL', 'I-REL', 'I-REL', '<end>'])
# ]

#########################

model = BiLSTM_CRF(len(word2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


for epoch in range(300):
    for sentence, tags in data_raw:
        model.zero_grad()

        # 输入
        sentence_in = torch.tensor(sentence, dtype=torch.long)

        targets = torch.tensor(tags, dtype=torch.long)

        # 获取loss
        loss = model.neg_log_likelihood(sentence_in, targets)
        print(loss)
        # 反向传播
        loss.backward()
        optimizer.step()

with torch.no_grad():
    # precheck_sent = prepare_sequence(training_data[0][0], word2id)
    print(model(data_raw[0][0]))
# import re
# import pickle
# import codecs
# import numpy as np

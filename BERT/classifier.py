from torch import nn
from fastNLP.modules import LSTM
import torch

# 定义模型
class BiLSTMMaxPoolCls(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=200, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = embed

        self.lstm = LSTM(self.embed.embedding_dim, hidden_size=hidden_size//2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, chars, seq_len):  # 这里的名称必须和DataSet中相应的field对应，比如之前我们DataSet中有chars，这里就必须为chars
        # chars:[batch_size, max_len]
        # seq_len: [batch_size, ]
        chars = self.embed(chars)
        outputs, _ = self.lstm(chars, seq_len)
        outputs = self.dropout_layer(outputs)
        outputs, _ = torch.max(outputs, dim=1)
        outputs = self.fc(outputs)

        return {'pred':outputs}  # [batch_size,], 返回值必须是dict类型，且预测值的key建议设为pred

import os
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle


def read_file_to_dataset(fp):
    ds = DataSet()
    with open(fp, 'r', encoding='utf-8') as f:
        f.readline()  # 第一行是title名称，忽略掉
        for line in f:
            line = line.strip()
            target, chars = line.split('\t')
            ins = Instance(target=target, raw_chars=chars)
            ds.append(ins)
    return ds
data_dir = 'data_bundle_1'
data_bundle = DataBundle()
for name in ['train.tsv', 'dev.tsv', 'test.tsv']:
    fp = os.path.join(data_dir, name)
    fp = fp.replace('\\', '/')
    ds = read_file_to_dataset(fp)
    data_bundle.set_dataset(name=name.split('.')[0], dataset=ds)

from fastNLP.io import ChnSentiCorpPipe
pipe = ChnSentiCorpPipe()
data_bundle = pipe.process(data_bundle)
char_vocab = data_bundle.get_vocab('chars')
# 初始化模型
from fastNLP.embeddings import BertEmbedding

# 这里为了演示一下效果，所以默认Bert不更新权重
bert_embed = BertEmbedding(char_vocab, model_dir_or_name='cn', auto_truncate=True, requires_grad=False)
model = BiLSTMMaxPoolCls(bert_embed, len(data_bundle.get_vocab('target')))


import torch
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-3)
metric = AccuracyMetric()
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=64, dev_data=data_bundle.get_dataset('test'),
                  metrics=metric, device=device, n_epochs=3)
trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型

# 在测试集上测试一下模型的性能
from fastNLP import Tester
print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
tester.test()
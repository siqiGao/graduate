# import numpy as np
# import pandas as pd
# df = pd.read_csv('processed_quetion.txt', sep="\t")
#
# dd = df[['question', 'type']]
# class_mapping = {label: idx for idx, label in enumerate(np.unique(dd['type']))}
# print(class_mapping, len(class_mapping))
# dd['type'] = dd['type'].map(class_mapping)
# print(dd)
# dd.rename(columns={'question':'raw_chars', 'type':'target'},inplace=True)
# print('修改一个列名\nmethod2_inplace:\n',dd)
# order = ['target', 'raw_chars']
# dd = dd[order]
# print(dd)
# train = dd.sample(frac=0.9)
# val = dd.sample(frac=0.1)
# test = dd.sample(frac=0.1)
#
# train.to_csv('train.tsv', sep="\t", index=False, encoding='utf-8')
# val.to_csv('val.tsv', sep="\t", index=False, encoding='utf-8')
# test.to_csv('test.tsv', sep="\t", index=False, encoding='utf-8')
import fastNLP
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

print(data_bundle)  # 查看以下数据集的情况
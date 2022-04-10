from fastNLP.io import WeiboNERPipe
data_bundle = WeiboNERPipe().process_from_file()
print(data_bundle.get_dataset('train')[:2])
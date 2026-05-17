from dataset import Multi30kDataset

train_ds = Multi30kDataset(split='train')
train_ds.build_vocab()
print(len(train_ds.de_vocab), len(train_ds.en_vocab))
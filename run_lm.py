import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

import numpy as np

import utils
import lm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
	# ======================
	# 超参数
	# ======================
	CELL = "lstm"                   # rnn, gru, lstm
	DATASET = 'movie'
	RATIO = 0.9
	WORD_DROP = 10
	MIN_LEN = 5
	MAX_LEN = 200
	BATCH_SIZE = 32
	SEQUENCE_LEN = 50
	EMBED_SIZE = 128
	HIDDEN_DIM = 256
	NUM_LAYERS = 2
	DROPOUT_RATE = 0.0
	EPOCH = 300
	LEARNING_RATE = 0.01
	MAX_GENERATE_LENGTH = 20
	GENERATE_EVERY = 5
	SEED = 100

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			print("{0:15}   ".format(var), all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	data_path = '../../__data/ROCStories.txt'
	train_path = 'train_roc'
	test_path = 'test_roc'
	vocabulary = utils.Vocabulary(
		data_path,
		max_len=MAX_LEN,
		min_len=MIN_LEN,
		word_drop=WORD_DROP
	)
	utils.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, min_len=MIN_LEN, ratio=RATIO, seed=SEED)
	train = utils.Corpus(train_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
	test = utils.Corpus(test_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
	train_generator = utils.Generator(train.corpus)
	test_generator = utils.Generator(test.corpus)

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = lm.LM(
		cell=CELL,
		vocab_size=vocabulary.vocab_size,
		embed_size=EMBED_SIZE,
		hidden_dim=HIDDEN_DIM,
		num_layers=NUM_LAYERS,
		dropout_rate=DROPOUT_RATE
	)
	model.to(device)
	summary(model, (20,))
	criteration = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	# optimizer = torch.optim.Adam(textRNN.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	print()

	# ======================
	# 训练与测试
	# ======================
	best_loss = 1000000
	for epoch in range(EPOCH):
		train_g = train_generator.build_generator(BATCH_SIZE, SEQUENCE_LEN)
		test_g = test_generator.build_generator(BATCH_SIZE, SEQUENCE_LEN)
		train_loss = []
		while True:
			try:
				text = train_g.__next__()
			except:
				break
			optimizer.zero_grad()
			y = model(torch.from_numpy(text[:, :-1]).long().to(device))
			loss = criteration(y.reshape(-1, vocabulary.vocab_size), torch.from_numpy(text[:, 1:]).reshape(-1).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		test_loss = []
		while True:
			with torch.no_grad():
				try:
					text = test_g.__next__()
				except:
					break
				y = model(torch.from_numpy(text[:, :-1]).long().to(device))
				loss = criteration(y.reshape(-1, vocabulary.vocab_size), torch.from_numpy(text[:, 1:]).reshape(-1).long().to(device))
				test_loss.append(loss.item())

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss)))

		if np.mean(test_loss) < best_loss:
			best_loss = np.mean(test_loss)
			print('-----------------------------------------------------')
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			torch.save(model.state_dict(), 'models/' + DATASET + '-' + str(epoch) + '.pkl')
			print('-----------------------------------------------------')

		if (epoch + 1) % GENERATE_EVERY == 0:
			with torch.no_grad():
				# 生成文本
				x = torch.LongTensor([[vocabulary.w2i['_BOS']]] * 3).to(device)
				for i in range(MAX_GENERATE_LENGTH):
					samp = model.sample(x)
					x = torch.cat([x, samp], dim=1)
				x = x.cpu().numpy()
			print('-----------------------------------------------------')
			for i in range(x.shape[0]):
				print(' '.join([vocabulary.i2w[_] for _ in list(x[i, :]) if _ not in
				                [vocabulary.w2i['_BOS'], vocabulary.w2i['_EOS'], vocabulary.w2i['_PAD']]]))
			print('-----------------------------------------------------')


if __name__ == '__main__':
	main()

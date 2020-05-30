import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

import numpy as np

import data
import lm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
	# ======================
	# 超参数
	# ======================
	CELL = "lstm"                   # rnn, gru, lstm
	BATCH_SIZE = 64
	EMBED_SIZE = 128
	HIDDEN_DIM = 256
	NUM_LAYERS = 1
	DROPOUT_RATE = 0.0
	EPOCH = 200
	LEARNING_RATE = 0.01
	MAX_GENERATE_LENGTH = 20
	SAVE_EVERY = 5

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			print("{0:15}   ".format(var), all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	with open('../_data/ROCStories.txt', 'r', encoding='utf8') as f:
		raw = f.read().split("\n")
	raw = list(filter(lambda x: x not in ['', None], raw))
	data_helper = data.DataHelper([raw], use_length=True)

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = lm.LM(
		cell=CELL,
		vocab_size=data_helper.vocab_size,
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
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)
		generator_test = data_helper.test_generator(BATCH_SIZE)
		train_loss = []
		while True:
			try:
				text, length = generator_train.__next__()
			except:
				break
			optimizer.zero_grad()
			y = model(torch.from_numpy(text[:, :-1]).long().to(device))
			loss = criteration(y.reshape(-1, data_helper.vocab_size), torch.from_numpy(text[:, 1:]).reshape(-1).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		test_loss = []
		while True:
			with torch.no_grad():
				try:
					text, label = generator_test.__next__()
				except:
					break
				y = model(torch.from_numpy(text[:, :-1]).long().to(device))
				loss = criteration(y.reshape(-1, data_helper.vocab_size), torch.from_numpy(text[:, 1:]).reshape(-1).long().to(device))
				test_loss.append(loss.item())

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss)))

		if (epoch + 1) % SAVE_EVERY == 0:
			print('-----------------------------------------------------')
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			model.save(model.state_dict(), 'model/lm-' + str(epoch) + '.pkl')

			with torch.no_grad():
				# 生成文本
				x = torch.LongTensor([[data_helper.w2i['_BOS']]] * 3)
				for i in range(MAX_GENERATE_LENGTH):
					samp = model.sample(x)
					x = torch.cat([x, samp], dim=1)
				x = x.cpu().numpy()
			for i in range(x.shape[0]):
				print(' '.join([data_helper.i2w[_] for _ in list(x[i, :]) if _ not in
				                [data_helper.w2i['_BOS'], data_helper.w2i['_EOS'], data_helper.w2i['_PAD']]]))
			print('-----------------------------------------------------')


if __name__ == '__main__':
	main()

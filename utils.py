import collections
import numpy as np


class Vocabulary(object):
	def __init__(self, data_path, max_len=200, min_len=5, word_drop=5):
		self._data_path = data_path
		self._max_len = max_len
		self._min_len = min_len
		self._word_drop = word_drop
		self.token_num = 0
		self.vocab_size_raw = 0
		self.vocab_size = 0
		self.w2i = {}
		self.i2w = {}
		self._build_vocabulary()

	def _build_vocabulary(self):
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'
		with open(self._data_path, 'r', encoding='utf8') as f:
			sentences = f.readlines()
		words_all = []
		for sentence in sentences:
			# _ = list(filter(lambda x: x not in [None, ''], sentence.split()))
			_ = sentence.split()
			if (len(_) >= self._min_len) and (len(_) <= self._max_len):
				words_all.extend(_)
		self.token_num = len(words_all)
		word_distribution = sorted(collections.Counter(words_all).items(), key=lambda x: x[1], reverse=True)
		self.vocab_size_raw = len(word_distribution)
		for (word, value) in word_distribution:
			if value > self._word_drop:
				self.w2i[word] = len(self.w2i)
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)


class Corpus(object):
	def __init__(self, data_path, vocabulary, max_len=200, min_len=5):
		self._data_path = data_path
		self._vocabulary = vocabulary
		self._max_len = max_len
		self._min_len = min_len
		self.corpus = []
		self.corpus_length = []
		self.sentence_num = 0
		self.max_sentence_length = 0
		self.min_sentence_length = 0
		self._build_corpus()

	def _build_corpus(self):
		def _transfer(word):
			try:
				return self._vocabulary.w2i[word]
			except:
				return self._vocabulary.w2i['_UNK']
		with open(self._data_path, 'r', encoding='utf8') as f:
			sentences = f.readlines()
		# sentences = list(filter(lambda x: x not in [None, ''], sentences))
		for sentence in sentences:
			# sentence = list(filter(lambda x: x not in [None, ''], sentence.split()))
			sentence = sentence.split()
			if (len(sentence) >= self._min_len) and (len(sentence) <= self._max_len):
				sentence = [self._vocabulary.w2i["_BOS"]] + sentence + [self._vocabulary.w2i["_EOS"]]
				self.corpus.append(list(map(_transfer, sentence)))
		self.corpus_length = [len(i) for i in self.corpus]
		self.max_sentence_length = max(self.corpus_length)
		self.min_sentence_length = min(self.corpus_length)
		self.sentence_num = len(self.corpus)


def split_corpus(data_path, train_path, test_path, max_len=200, min_len=5, ratio=0.8, seed=0):
	with open(data_path, 'r', encoding='utf8') as f:
		sentences = f.readlines()
	sentences = [_ for _ in filter(lambda x: x not in [None, ''], sentences)
	             if len(_.split()) <= max_len and len(_.split()) >= min_len]
	np.random.seed(seed)
	np.random.shuffle(sentences)
	train = sentences[:int(len(sentences) * ratio)]
	test = sentences[int(len(sentences) * ratio):]
	with open(train_path, 'w', encoding='utf8') as f:
		for sentence in train:
			f.write(sentence)
	with open(test_path, 'w', encoding='utf8') as f:
		for sentence in test:
			f.write(sentence)


class Generator(object):
	def __init__(self, data):
		self._data = data

	def build_generator(self, batch_size, sequence_len, shuffle=True):
		if shuffle:
			np.random.shuffle(self._data)
		data_ = []
		for _ in self._data:
			data_.extend(_)
		batch_num = len(self._data) // (batch_size * sequence_len)
		data = data_[:batch_size * batch_num * sequence_len]
		data = np.array(data).reshape(batch_num * batch_size, sequence_len)
		while True:
			batch_data = data[0:batch_size]                   # 产生一个batch的index
			data = data[batch_size:]                          # 去掉本次index
			if len(batch_data) == 0:
				return True
			yield batch_data


if __name__ == '__main__':
	vocabulary = Vocabulary('F:/code/python/__data/dataset2020/movie2020.txt', word_drop=10)
	split_corpus('F:/code/python/__data/dataset2020/movie2020.txt', 'train_movie', 'test_movie')
	# corpus = Corpus('F:/code/python/__data/dataset2020/news2020.txt', vocabulary)
	test = Corpus('test_movie', vocabulary)
	test_generator = Generator(test.corpus)
	test_g = test_generator.build_generator(64, 50)
	text = test_g.__next__()
	pass
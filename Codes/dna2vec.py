'''
Implementation of the dna2vec algorithm as it is described in https://arxiv.org/pdf/1701.06279.pdf
'''
import numpy as np
from collections import Counter
import math
import time

class dna2vec(object):

    def __init__(self, data_path):

        with open(data_path, 'r') as file:
            S_list = file.readlines()

        for _ in range(len(S_list)):
            S_list[_] = S_list[_].strip()

        self.S_list = S_list


    def build_vocab(self, k_high, k_low, N, vocab_size):
        '''
        Buils vocabulary from a list of strings
        :param S_list: (list)
        :param k_high: (int)
        :param k_low: (int)
        :param N: (int) Number sequence of k-mers to create
        :param embeding_size: (int)
        :return: (list), (dict), (dict)
        '''
        # Extracting Random k-mers (boostrap)

        self.k_high = k_high
        self.k_low = k_low

        Collection = []
        for _ in range(N):
            ind = np.random.randint(low=0, high=len(self.S_list)-1)
            Collection.append(build_rand_kmers(self.S_list[ind], k_high=k_high, k_low=k_low))

        # Building vocabulary with frequency
        word_freq = {}
        for S in Collection:
            for word in S:
                if word not in word_freq.keys():
                    word_freq[word] = 1

                else:
                    word_freq[word] += 1

        # keeping only the most frequent words
        occurrence = [['ukn', 0]] + Counter(word_freq).most_common(vocab_size-1)

        vocab_size = min(vocab_size, len(occurrence))

        word2ind = {}
        ind2word = {}
        for _ in range(vocab_size):
            word2ind[occurrence[_][0]] = _
            ind2word[_] = occurrence[_][0]

        for S in Collection:
            for _ in range(len(S)):
                if S[_] not in word2ind.keys():
                    S[_] = 'ukn'
                    occurrence[0][1] += 1

        Collection_inds = Collection
        for S in Collection_inds:
            for _ in range(len(S)):
                S[_] = word2ind[S[_]]

        word_freq = {}
        total_size = sum([_[1] for _ in occurrence])
        for _ in occurrence:
            word_freq[_[0]] = _[1] / total_size

        self.keep_word_thresh = np.median(list(word_freq.values()))

        probs = []
        for _ in word_freq.keys():
            if word_freq[_] != 0:
                probs.append((np.sqrt(word_freq[_] / self.keep_word_thresh) + 1) * \
                            self.keep_word_thresh / word_freq[_])

            else:
                probs.append(0)

        self.probs = probs/sum(probs)

        self.vocab_size = vocab_size
        self.word_freq = word_freq
        self.Collection = Collection
        self.Collection_inds = Collection_inds
        self.word2ind = word2ind
        self.ind2word = ind2word


    def fit(self, epochs, window, n_negative, embeding_size, learning_rate):
        '''
        Fits the dna2vec model with respect to the previously build vocabulary
        :param epochs: (int) Number of epochs
        :param window: (int) size of half of the window
        :param n_negative: (int)
        :param embeding_size: (int)
        :param learning_rate: (float)
        :return:
        '''
        self.embeding_size = embeding_size
        self.M_in = np.random.uniform(low=0, high=1, size=(self.vocab_size, self.embeding_size))
        self.M_out = np.random.uniform(low=0, high=1, size=(self.vocab_size, self.embeding_size))

        start_time = time.time()
        for epoch in range(epochs):
            count = 0
            for S in self.Collection_inds:
                if count % 10 == 0:
                    print('iter ', count, '%.2f' % (time.time()-start_time))
                count += 1
                for context_ind in range(len(S)):
                    context_word = S[context_ind]
                    for target_ind in range(max(0, context_ind - window), min(context_ind + window + 1, len(S))):
                        if context_ind != target_ind:
                            target_word = S[target_ind]
                            input_words = self.negative_sampling(target_word, n_negative)
                            self.update_coeffs(context_word, input_words, learning_rate)


    def negative_sampling(self, target_word, n_negative):
        '''
        :param target_word: (int)
        :param n_negative: (int)
        :return: (list)
        '''
        output = [target_word]
        while len(output) < n_negative + 1:
            sampled_word = np.random.choice(list(self.word_freq.keys()), p=self.probs)
            output.append(self.word2ind[sampled_word])

        return output


    def update_coeffs(self, context_word, input_words, learning_rate):

        temp = [0]*self.embeding_size
        K = len(input_words)
        for k in range(K):
            target_word = input_words[k]
            if k == 0:
                label = 1

            else:
                label = 0

            inn = np.dot(self.M_in[context_word,:], self.M_out[target_word, :])
            err = label - sigmoid(inn)
            for j in range(self.embeding_size):
                temp[j] += err * self.M_out[target_word, j]

            for j in range(self.embeding_size):
                self.M_out[target_word, j] += learning_rate * err * self.M_in[context_word, j]

        for j in range(self.embeding_size):
            self.M_in[context_word, j] += learning_rate * temp[j]


    def embed_seq(self, sequence, nb_draws):

        embedding = np.zeros((1, self.embeding_size))
        for _ in range(nb_draws):
            k_mers = build_rand_kmers(sequence, self.k_high, self.k_low)
            for k_mer in k_mers:
                if k_mer in self.word2ind.keys():
                    embedding += self.M_in[self.word2ind[k_mer], :]
                else:
                    embedding += self.M_in[self.word2ind['ukn'], :]

        return embedding / nb_draws


    def embed_data(self, data, nb_draws):

        embeddings = np.zeros((len(data), self.embeding_size))
        for _ in range(len(data)):
            embeddings[_, :] = self.embed_seq(data[_], nb_draws)

        return embeddings



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def find_kmers(S, k):
    '''
    Finds all k-mers
    :param S: (string)
    :param k: (int)
    :return: (list)
    '''
    kmers = []
    n = len(S)
    for i in range(n - k + 1):
        kmers.append(S[i:i + k])

    return kmers


def build_rand_kmers(S, k_high, k_low):
    '''
    Extracts k-mers of a string, with random length between k_high and k_low
    :param S: (string)
    :param k_high: (int)
    :param k_low: (int)
    :return: (list)
    '''
    kmers = []
    n = len(S)
    l=0
    while l < n - k_high:
        k = np.random.randint(low=k_low, high=k_high)
        kmers.append(S[l:l + k])
        l += k

    return kmers

if __name__ == '__main__':
    import timeit

    # D2V = dna2vec('../Data/Xtr0.csv')
    # start_time = time.time()
    # print(start_time)
    # D2V.build_vocab(k_high=8, k_low=3, N=100000, vocab_size=20000)
    # print(time.time()-start_time)
    # D2V.fit(epochs=1, window=5, n_negative=20, embeding_size=300, learning_rate=0.01)
    # print(time.time()-start_time)

    # for _ in [2000, 50000, 100000, 500000, 1000000]:
    #     D2V = dna2vec('../Data/Xtr0.csv')
    #     D2V.build_vocab(8, 3, _, 40000)
    #     print(_, D2V.vocab_size)

    D2V = dna2vec('../Data/Xtr0.csv')
    D2V.build_vocab(k_high=8, k_low=3, N=100000, vocab_size=2000)
    start_time = time.time()
    for _ in range(10000):
        D2V.negative_sampling(target_word=1, context_word=50, n_negative=20, window=5)
    print(time.time() - start_time)

    D2V = dna2vec('../Data/Xtr0.csv')
    D2V.build_vocab(k_high=8, k_low=3, N=1000, vocab_size=2000)
    start_time = time.time()
    for _ in range(10000):
        D2V.negative_sampling(target_word=1, context_word=50, n_negative=20, window=5)
    print(time.time() - start_time)
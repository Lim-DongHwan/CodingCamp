from typing import List, Optional, Union
from collections import defaultdict

class BPETokenizer:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        self.corpus = [] if corpus is None else corpus
        self.dictionary = defaultdict(int)
        self.vocab = {}
    
    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]
        self.corpus.extend(corpus)
    
    def get_dictionary(self):
        """ 데이터를 읽어와 어절 별 character 단위로 구분된 후보 사전 생성 """
        dictionary = defaultdict(int)
        for line in self.corpus:
            tokens = line.strip().split(" ")
            for token in tokens:
                dictionary[" ".join(list(token)) + " </w>"] += 1
        return dict(dictionary)
        self.dictionary = dict(dictionary)
    
    def get_pairs(self):
        """ 딕셔너리를 활용하여 바이그램 페어 생성 """
        pairs = defaultdict(int)
        for word, freq in self.dictionary.items():
            word_lst = word.split()
            for i in range(len(word_lst)-1):
                pairs[(word_lst[i], word_lst[i+1])] += freq
        return dict(pairs)
        self.pairs = dict(pairs)
        
    def merge_dictionary(self, pairs, dictionary):
        """ 가장 자주 등장한 바이그램 페어를 merge하여 딕셔너리 업데이트"""
        
        result = defaultdict(int)
        best_pair = max(pairs, key=pairs.get)
        for word, freq in dictionary.items():
            paired = word.replace(" ".join(best_pair), "".join(best_pair))
            result[paired] = dictionary[word]
        self.dictionary = dict(result)
        self.get_vocab()  # merge 이후 vocabulary 업데이트
    
    def train(self, n_iter: int) -> None:
        """ 주어진 반복 횟수만큼 훈련 진행 """
        dictionary = self.get_dictionary()
        for i in range(n_iter):
            pairs = self.get_pairs()
            self.merge_dictionary(pairs, self.dictionary)
    
    
    def get_vocab(self):
        """ 훈련된 딕셔너리를 기반으로 최종 vocabulary 생성 """
        result = defaultdict(int)
        for word, freq in self.dictionary.items():
            tokens = word.split()
            for token in tokens:
                result[token] += freq
        self.vocab = dict(result)
    

class WordTokenizer:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        self.vocab: List[str] = []
        if corpus:
            self.add_corpus(corpus)

    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]
        for sentence in corpus:
            tokens = sentence.split()
            self.vocab.extend(tokens)

    def train(self, *args, **kwargs) -> None:
        pass

    def tokenize(self, 
                 text: Union[List[str], str], 
                 padding: bool = False,
                 max_length: Optional[int] = None) -> Union[List[List[str]], List[str]]:
        if isinstance(text, str):
            text = [text]

        tokenized_texts = [sentence.split() for sentence in text]

        if padding:
            max_len = max(len(tokens) for tokens in tokenized_texts)
            tokenized_texts = [tokens + ['<pad>'] * (max_len - len(tokens)) for tokens in tokenized_texts]

        if max_length is not None:
            tokenized_texts = [tokens[:max_length] for tokens in tokenized_texts]

        return tokenized_texts

    def __call__(self, 
                 text: Union[List[str], str],
                 padding: bool = False,
                 max_length: Optional[int] = None) -> Union[List[List[str]], List[str]]:
        return self.tokenize(text, padding, max_length)

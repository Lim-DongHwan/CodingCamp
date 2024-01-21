import argparse
from typing import Optional, List
from YBIGTA.tokenizers import BPETokenizer, WordTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--use_bpe", type=bool, default=True, help="True for BPE, False for WordTokenizer")
parser.add_argument("--n_iter", type=int, default=10000)

args = parser.parse_args() 

corpus = [
    '와이빅타 파이팅',
    '와이빅타 코딩캠프 준비하시는 분들이 정말 고생이 많으실 것 같다',
    '열심히 수업 듣고 코딩 전문가가 되고 싶다'
]

if args.use_bpe:
    tokenizer = BPETokenizer(corpus)
else:
    tokenizer = WordTokenizer(corpus)

 
tokenizer.train(n_iter=args.n_iter)

  
tokenized_texts = tokenizer.tokenize()
print(tokenized_texts)

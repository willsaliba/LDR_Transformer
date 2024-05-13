import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from copy import deepcopy

### HYPER PARAMS ###
vocab_size = 3000
tokenizer_save_path = "/Users/willsaliba/Documents/code/uni/advTopics/transformer/trained_tokenizerM2/"
tokenizer_train_path = "/Users/willsaliba/Documents/code/uni/advTopics/data/mini_data/train/"

#removing pre-exisiting files
for file_name in os.listdir(tokenizer_save_path):
    file_path = os.path.join(tokenizer_save_path, file_name)
    os.remove(file_path)

""" INTIALISING & SAVING TOKENIZER """
#creating BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

#adding GPT4 pattern and special tokens
regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
tokenizer.pre_tokenizer = Split(pattern=regex_pattern, behavior='isolated')

#getting tokenizer training data
files = []
for file_name in os.listdir(tokenizer_train_path):
    file_path = os.path.join(tokenizer_train_path, file_name)
    files.append(file_path)

trainer = BpeTrainer(vocab_size = vocab_size, special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)
tokenizer.save(tokenizer_save_path + "tokenizer")
sameNew = Tokenizer.from_file(tokenizer_save_path + "tokenizer")

print("DID THIS")

#converting tokenizer to transformer tokenizer and saving
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=1536)
fast_tokenizer.save_pretrained(tokenizer_save_path)

print("\n\n---UNTRAINED GPT4 BPE TOKENIZER SAVED---\n\n")

import os
from tokenizers import Tokenizer, Regex, pre_tokenizers
from tokenizers.normalizers import NFC
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

### HYPER PARAMS ###
vocab_size = 3000
tokenizer_save_path = "/Users/willsaliba/Documents/code/uni/advTopics/transformer/trained_tokenizerM2/test/"
tokenizer_train_path = "/Users/willsaliba/Documents/code/uni/advTopics/data/mini_data/train/"

#creating BPE tokenizer 
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = NFC()

#adding GPT4 pretokenisation 
re_pattern = r"(?i)[sdmt]|ll|ve|re|[^\r\n\w]?+[a-zA-Z]+|\d{1,3}|\s?[^\s\w]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    Split(pattern=Regex(re_pattern), behavior="isolated"),
    ByteLevel(add_prefix_space=False, use_regex=False),
])

#getting tokenizer training data
files = []
for file_name in os.listdir(tokenizer_train_path):
    file_path = os.path.join(tokenizer_train_path, file_name)
    files.append(file_path)

#training tokenizer
trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)

#removing pre-exisiting files in tokenizer_save_path
for file_name in os.listdir(tokenizer_save_path):
    file_path = os.path.join(tokenizer_save_path, file_name)
    os.remove(file_path)

#converting tokenizer to transformer tokenizer and saving
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.save_pretrained(tokenizer_save_path)

print("\n\n--- GPT4 BPE TOKENIZER SAVED ---\n\n")

test_txt = """0 !LEOCAD MODEL AUTHOR LEGO staff (unknown); .ldr version from Nitrofurano;
0 !LEOCAD MODEL DESCRIPTION 293 - Piano - 1973
1 0 -100 -112 130 1 0 0 0 1 0 0 0 1 3008.DAT
1 0 -100 -112 90 1 0 0 0 1 0 0 0 1 3008.DAT"""

tokenizer2 = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer2.pre_tokenizer = Split(pattern=Regex(re_pattern), behavior="isolated")
trainer2 = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer2.train(files, trainer2)
tokens2 = tokenizer2.encode(test_txt)
tokens2 = tokens2.tokens

tokens1 = tokenizer.encode(test_txt)
tokens1 = tokens1.tokens


print("\n\n\n\n")
print(f"TOKENS1: {tokens1}\n\n")
print(f"TOKENS2: {tokens2}\n\n")
print(f"{len(tokens1)} - {len(tokens2)}")





import os
os.system('clear')

from tokenizers import Tokenizer, Regex, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.decoders import ByteLevel as PostByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

### HYPER PARAMS ### 140 or 1042
vocab_size = 10000
tokenizer_train_path = "/Users/willsaliba/Documents/code/uni/advTopics/data/omr8_clean"
tokenizer_save_path = "/Users/willsaliba/Documents/code/uni/advTopics/tokenizers/omr8_base"

#creating BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

#adding GPT4 pretokenisation + byte conversion for chars (allows this to work)
re_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    Split(pattern=Regex(re_pattern), behavior="isolated"),
    ByteLevel(add_prefix_space=False, use_regex=False)
])
tokenizer.decoder = PostByteLevel()

#getting tokenizer training data and training tokenizer
files = []
for file_name in os.listdir(tokenizer_train_path):
    file_path = os.path.join(tokenizer_train_path, file_name)
    files.append(file_path)
trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)

#removing pre-exisiting files in tokenizer_save_path
for file_name in os.listdir(tokenizer_save_path):
    file_path = os.path.join(tokenizer_save_path, file_name)
    os.remove(file_path)

#converting tokenizer to transformer tokenizer and saving
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.save_pretrained(tokenizer_save_path)

print("\n\n--- M2 TOKENIZER SAVED ---\n\n")
print(f"FINAL SIZE: {len(tokenizer.get_vocab())}")

test_text = """
1 0 0 0 0 0 0 -1 0 1 0 1 0 0 3832.dat 
1 0 0 -24 70 -1 0 0 0 1 0 0 0 -1 3004.dat 
1 0 0 -8 10 -1 0 0 0 1 0 0 0 -1 3710.dat 
1 0 0 -8 -20 -1 0 0 0 1 0 0 0 -1 3020.dat 
1 0 0 -16 -60 -1 0 0 0 1 0 0 0 -1 3788.dat 
1 0 0 -8 -90 -1 0 0 0 1 0 0 0 -1 3710.dat 
1 0 0 8 -60 0 0 -1 0 1 0 1 0 0 3022.dat 
1 0 0 8 80 0 0 -1 0 1 0 1 0 0 3022.dat 
"""

encoding = fast_tokenizer.encode(test_text)
decoding = fast_tokenizer.decode(encoding)

print("Equal: ", test_text == decoding)
print(decoding)

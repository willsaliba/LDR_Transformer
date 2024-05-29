import os
os.system('clear')

from tokenizers import Tokenizer, Regex, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.decoders import ByteLevel as PostByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

### HYPER PARAMS ### 1150 4150
vocab_size = 1150
tokenizer_train_path = "/Users/willsaliba/Documents/code/uni/advTopics/data/rand8/train"
tokenizer_save_path = "/Users/willsaliba/Documents/code/uni/advTopics/tokenizers/m2_tokenizers/rand8"

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

print("\n\n--- GPT4 BPE TOKENIZER SAVED ---\n\n")
print(f"FINAL SIZE: {len(tokenizer.get_vocab())}")

test_text = """
1 0 -50.000000 0.000000 70.000000 0.000000 0.000000 -1.000000 0.000000 1.000000 0.000000 1.000000 0.000000 0.000000 3034.dat
1 7 -50.000000 -8.000000 90.000000 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000 4600.dat
1 0 -50.000000 -8.000000 60.000000 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000 3023.dat
1 0 -60.000000 -8.000000 20.000000 0.000000 0.000000 1.000000 0.000000 1.000000 0.000000 -1.000000 0.000000 0.000000 4081.dat
1 0 -40.000000 -8.000000 20.000000 0.000000 0.000000 -1.000000 0.000000 1.000000 0.000000 1.000000 0.000000 0.000000 4081.dat
1 0 -50.000000 -16.000000 80.000000 -1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 -1.000000 3023.dat
1 0 -80.000000 -6.000000 -6.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 3062.dat
1 0 -80.000000 -6.000000 46.000000 0.000000 0.000000 -1.000000 1.000000 0.000000 0.000000 0.000000 -1.000000 0.000000 3062.dat
"""
encoding = fast_tokenizer.encode(test_text)
decoding = fast_tokenizer.decode(encoding)

print("Equal: ", test_text == decoding)
print(decoding)

"""
MACH1 TOKENIZER
-Initial implementation of custom LDR tokenizer, which uses the GPT4 pre-tokenization pattern

GPT4 pattern -> https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
"""
import os
import regex as regx
import pickle

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.preTrainingSize = -1
        self.postTrainingSize = -1

    """
    Training Functionality 
    """

    #Finding frequency of adjacent pairs within each subUnit
    def get_counts(self, ids):
        counts = {}
        for subUnit in ids:
            for pair in zip(subUnit, subUnit[1:]):     #pythonic way to iterate consec elements
                counts[pair] = counts.get(pair, 0) + 1 #if pair DNE val defaulted to 0 
        return counts

    #replacing all occurances of pair, within ids (token_subUnits), with new token idx
    def merge_tokens(self, ids, pair, idx):
        newIDs = []
        for subUnit in ids:
            newUnit, i = [], 0
            while i < len(subUnit):
                if i < len(subUnit)-1 and subUnit[i] == pair[0] and subUnit[i+1] == pair[1]:
                    newUnit.append(idx)
                    i += 2
                else:
                    newUnit.append(subUnit[i])
                    i += 1
            newIDs.append(newUnit)
        return newIDs

    #completes and updates vocabulary
    def run_training(self, directory, vocab_size):
        #ingest all training data from directory
        rawLDR = ""
        for filename in os.listdir(directory):
            if filename.endswith(".ldr"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    rawLDR += file.read()

        #saving initial file size
        self.preTrainingSize = len(rawLDR)

        #pretokenize raw LDR data it using GPT-4 pattern
        gpt4_pattern = regx.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        raw_subUnits = regx.findall(gpt4_pattern, rawLDR)

        #convert raw LDR sub units to byte integers
        subUnits = []
        for subUnit in raw_subUnits:
            byte_subUnit = subUnit.encode("utf-8")
            int_subUnit = list(map(int, byte_subUnit))
            subUnits.append(int_subUnit)

        #perform merges
        num_merges = vocab_size - 256
        for i in range(num_merges):
            frequencies = self.get_counts(subUnits)
            topPair = max(frequencies, key=frequencies.get) #comparares on vals rather then keys
            newID = 256 + i
            subUnits = self.merge_tokens(subUnits, topPair, newID)
            self.merges[topPair] = newID
            #print statement to show progress
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} merges")

        #saving final file size
        self.postTrainingSize = sum(len(subUnit) for subUnit in subUnits)

        #finalise vocabulary
        for idx in range(256): 
            self.vocab[idx] = bytes([idx]) #bytes needs list arg
        for (t0, t1), idx in self.merges.items(): #items() makes map traversable & order inserted
            self.vocab[idx] = self.vocab[t0] + self.vocab[t1] #concatenating 'byte objects'

    #saves tokenizer merges and vocab for inference
    def save_tokenizer(self, filename):
        data = {
            'merges': self.merges,
            'vocab': self.vocab,
            'preTrainingSize': self.preTrainingSize,
            'postTrainingSize': self.postTrainingSize
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    """
    Inference Functionality 
    """
    #loading vairables from saved trained tokenizer
    def load_tokenizer(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.merges = data['merges']
        self.vocab = data['vocab']
        self.preTrainingSize = data['preTrainingSize']
        self.postTrainingSize = data['postTrainingSize']

    #Finding frequency of adjacent pairs within encoding tokenised input
    def get_counts_inf(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1 
        return counts

    #Merging tokens within tokenized input
    def merge_tokens_inf(self, ids, pair, idx):
        newIDs, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newIDs.append(idx)
                i += 2
            else:
                newIDs.append(ids[i])
                i += 1
        return newIDs

    #ENCODE: LDR -> Tokens
    def encode(self, text):
        tokens = list(text.encode("utf-8")) #raw bytes
        #loop till no more merges possible
        while len(tokens) >= 2:
            frequencies = self.get_counts_inf(tokens)
            #want byte pair (key) inside stats that has lowest indx in merges dict (lowest merged first)
            pair = min(frequencies, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            tokens = self.merge_tokens_inf(tokens, pair, idx)
        return tokens

    #DECODE: Tokens -> LDR
    def decode(self, tokens):
        #processing each token by decoding it with
        byteTokens = b"".join(self.vocab[token] for token in tokens)
        text = byteTokens.decode("utf-8", errors="replace")
        return text


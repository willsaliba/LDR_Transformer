import os
from M1_Tokenizer import Tokenizer

#variable reloads trained tokenizer or retrains it
train = False

#intialising tokenizer and either loading or training it
Mach1 = Tokenizer()
if train:
    Mach1.run_training("mini_LDR_data", 500)
    Mach1.save_tokenizer('trained_tok.pickle')
else:
    Mach1.load_tokenizer('trained_tok.pickle')

#testing encode and decode functionality
text = '''
0 !LEOCAD MODEL AUTHOR LEGO staff (unknown); .ldr from Nitrofurano;
0 !LEOCAD MODEL DESCRIPTION 107 - Canada Post Truck - 1985
0 !LEOCAD GROUP BEGIN truck1
1 0 -50 -64 130 1 0 0 0 1 0 0 0 1 3957.DAT
1 0 -90 -40 -30 1 0 0 0 1 0 0 0 1 4070.DAT
1 4 -80 -80 110 1 0 0 0 1 0 0 0 1 3010.DAT
1 4 -80 -80 -10 0 0 1 0 1 0 -1 0 0 3002.DAT
'''

theTokens = Mach1.encode(text)
result = Mach1.decode(theTokens)
print(f"\nEncode & Decode Success: {result == text} \n")

print(f"Dataset size pre-training: {Mach1.preTrainingSize}")
print(f"Dataset size post-training: {Mach1.postTrainingSize}\n")
import os
from M1_Tokenizer import Tokenizer

#hyper params controlling training / loading
vocab_size = 30000
data_dir = '/Users/willsaliba/Documents/1code/uni/advTopics/data/omr/sorted/2bricks'
trained_tokenizer = 'test2.pickle'

#intialising tokenizer and either loading or training it
Mach1 = Tokenizer()
Mach1.run_training(data_dir, vocab_size)
print(f"Tokenizer successfully trained")
Mach1.save_tokenizer(trained_tokenizer)
print(f"Tokenizer successfully saved as: {trained_tokenizer}\n\n")

#quick test to ensure tokenizer loaded properly
text = '''
0 !LEOCAD MODEL AUTHOR LEGO staff (unknown); .ldr from Nitrofurano;
0 !LEOCAD MODEL DESCRIPTION 107 - Canada Post Truck - 1985
0 !LEOCAD GROUP BEGIN truck1
1 0 -50 -64 130 1 0 0 0 1 0 0 0 1 3957.DAT
1 0 -90 -40 -30 1 0 0 0 1 0 0 0 1 4070.DAT
1 4 -80 -80 110 1 0 0 0 1 0 0 0 1 3010.DAT
1 4 -80 -80 -10 0 0 1 0 1 0 -1 0 0 3002.DAT
'''
result = Mach1.decode(Mach1.encode(text))
print(f"\nTokenizer Working Properly: {result == text} \n")

#inspecting compression
print(f"Dataset size pre-training: {Mach1.preTrainingSize}")
print(f"Dataset size post-training: {Mach1.postTrainingSize}\n")

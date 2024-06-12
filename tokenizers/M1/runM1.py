from M1_Tokenizer import Tokenizer

#pick the tokenizer
trained_tokenizer = 'mark1/m1_turbo/omr8_turbo.pickle'

#loading trained tokenizer values
Mark1 = Tokenizer()
Mark1.load_tokenizer(trained_tokenizer)
print(f"Tokenizer loaded from: {trained_tokenizer}")
print(f"Num merges: {len(Mark1.merges)}")
print(f"Final Vocab: {len(Mark1.vocab)}")

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
print(f"\nTokenizer Working Properly: {Mark1.decode(Mark1.encode(text)) == text} \n")

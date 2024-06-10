**ML LEGO COMPOSITIONAL UNDERSTANDING**  
This repository contains the code to create a hugging face model which can be trained on LEGO ldr files, 
enabling the model to generate new assemblies and complete LEGO assemblies given intial bricks.  

The sucess of these models can be inspected in this report:  

<u>Overview</u>  
There are multiple LDR datasets available for download here:  
There are multiple trained models available for download here:  

*Using this repository you can:*
- Apply various preprocessing steps to LDR datasets
- Train your own custom tokenizer
- Train your own custom transformer (with custom tokenizer or pretrained one)
- Run inference on a model
- Evaluate a model using our LEGO assembly specific metrics.

<u>Data Preprocessing</u>  
The preprocess.py file contains various functions for preproccessing the data. An explanation of these preprocessing steps and their
effect on our models is available in the research paper linked above.   

<u>Training Tokenizer</u>  
In the tokenizers folder, there's a file train_M2.py, which enables you to select the dataset for training the tokenizer and adjust its  vocabulary size.  

M2 is a hugging face tokenizer implemented with BPE algorithm and GPT4 pretokenization pattern. The folders stage1-3 in the tokenizers folder contain the M2 tokenizers trained and tested in our research effort.  

<u>Training Model</u>  
The train_model.py file trains a Hugging Face Transformer. This file enables you to select your training dataset and tokenizer, as well as adjust any training or model paramters.  

The link above contains the various models trained and tested in our research effort.  

<u>Inference / Evaluation</u>  
Within the generation folder there is a generate.py file which allows you to run inference using a prompt. You can pass the prompt "1" to generate a totally new LEGO assembly, or provide the first few bricks from an assembly and the model will add additional bricks.  

Within the generation folder there are two evaluate files which test a models performance in predicting the remainder of lego assemblies
using the initial 80% of blocks. To understand these metrics view the research report above. For models without quaternion conversion preprocessing step, use the evaluate.py file, and conversely for models with the conversion, use evaluate_q.py.  






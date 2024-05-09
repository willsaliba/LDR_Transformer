Hey will

# Files
Datsets: contains the different datasets we are using

gpt_x_brick_logs, logs, new_logs: are all trained transformer results so dont change

tokenize: This is the file for the GPT tokeniser we used for the transformer. This will be the format we need the tokenizer in

all the train.py files are different transformers being trained (mainly look at train.py)

evaluate.py: my class i was making to run the etsts. might not need due to the email we received from break and make.
# Transformer

To train the transformer look at train.py as an example (Also i made you a copy of this just called trainWill.py to mess around with and try get working)

# Integrating new tokenizer
1. We will need to import it

2. We will need to change this line in the main function to lead it
    ```py
    tokenizer = load_tokenizer(
        corpus=train_lines + eval_lines,
        vocab_size=vocab_size,
        model_max_length=n_positions,
    )
    ```
3. This may mean we need to chnage the load_tokenizer function 

4. After we have it loaded like that it should be able to integerate into the transforemr. The transforer is initlised in this lines

```py
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
    )

    if checkpoint_dir and checkpoint_dir.exists():
        model = AutoModelForCausalLM.load_pretrained(checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_config(config)
```

Other stuff:
    generate.py calls a trained model to generate. But we still need to upload ours to hugging face to get it to run. output is gen.ldr. 
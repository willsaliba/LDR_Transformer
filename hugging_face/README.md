This code demonstrates the usage of GPT-2 on LDR-files. The code does not make any special assumptions about the nature of LDR-files apart from the function that parses LDR lines into lines of text, where it only considers lines associated with sub-blocks (i.e. starting with `1`).

For demonstration purposes, I also provide the pre-trained model (`checkpoint/`) and tokenizer (`tokenizer/`).

# Setup
The usage of Docker is highly recommended.
To build a docker container, execute:
```bash
docker build --tag <tag-for-your-container> docker/
```

To start and attach your container for the first time, execute:
```bash
docker run -v ./:/home/mambauser/lego/ --name=<name-for-your-container> -it --gpus all <tag-of-your-container-from-the-first-step> bash
```

To exit your container without stopping it, use `Ctrl+P` followed by `Ctrl+Q`.
If you exit your container via the `exit` command, the container will be stopped (meaning all of its processes will be killed). Your data will still be available. To start and attach again, execute:

```bash
docker start <name-of-your-container> && docker attach <name-of-your-container>
```
The above are minimal example commands that should be enough to get you started. To learn more about Docker, please consult with the Docker docs.

# Data
The training and test data are coming from [LTRON's random stack generation](https://github.com/aaronwalsman/ltron/blob/v1.0.0/ltron/dataset/random_stack.py)

# Training
To train the model with the default parameters, run:
```bash
python train.py <path-to-data-dir>
```
Execute `python train.py --help` to see what other options are readily available via CLI.

# Generation
Once you have a model checkpoint, you can generate a new LDR file by running:
```bash
python generate.py <path-to-checkpoint-dir>
```
Execute `python generate.py --help` to see what other options are readily available via CLI.



#### ZACH README ####

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
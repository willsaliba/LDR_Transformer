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
#standard imports
from pathlib import Path
import re
import typing as T
#installed modules
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import typer

#Updated to use 'raw' string for the regex pattern, to remove error message
FN_P = r"([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(
    rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)"
)

#loading ldr files
def load_all_ldrs(root_dir: Path, decimals: int = 2, ):
    """
    This reads all LDR files from the specified directory
    and rounds up all numeric entries to the specified number of decimals;
    the rounding part works well for synthetic data, use with care on
    real models.
    """

    src_files = sorted(list(root_dir.glob("*.ldr")) + list(root_dir.glob("*.mpd")))

    all_lines = []
    for src_file in src_files:
        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1:
                continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry):
                    processed.append(str(int(float(numeric_entry))))
                else:
                    processed.append(
                        str(np.round(float(numeric_entry), decimals=decimals))
                    )
            processed.append(m[0][-1])  # part ID
            file_lines.append(" ".join(processed))
        all_lines.append("\n".join(file_lines))
    return all_lines

#creates the LDR text data set
class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer,):
        self.examples = tokenizer.batch_encode_plus(lines).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

#loads tokenizer if given one, otherwise trains a new one
def load_tokenizer(
    tokenizer_params_path: Path = Path("./tokenizer/"),
    default_tokenizer: str = "gpt2",
    vocab_size: int = 52000,
    corpus=None,
    model_max_length: int = 1024,
):
    #if pretrained tokenizer exists, load it
    if tokenizer_params_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
        logger.info(f"Loaded tokenizer from {tokenizer_params_path}")
        return tokenizer
    #else train new tokenizer    
    logger.info(f"Could not load tokenizer from {tokenizer_params_path}. Training a new one")
    old_tokenizer = AutoTokenizer.from_pretrained(default_tokenizer, model_max_length=model_max_length)
    tokenizer = old_tokenizer.train_new_from_iterator(
        corpus,
        vocab_size,
        new_special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer.save_pretrained(tokenizer_params_path)
    return tokenizer


def main(
    ldr_root_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/rand/rand4"),
    output_dir: Path = Path("./new_logs/"),
    checkpoint_dir: T.Optional[Path] = None,
    n_positions: int = 1536,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 1000,
    save_steps: int = 10000,
    eval_steps: int = 10000,
    fp16: bool = True,
    save_total_limit: int = 5,
    learning_rate: float = 5e-4,
    vocab_size: int = 5000,
):
    train_lines = load_all_ldrs(ldr_root_dir / "train")
    eval_lines = load_all_ldrs(ldr_root_dir / "test")

    tokenizer = load_tokenizer(
        corpus=train_lines + eval_lines,
        vocab_size=vocab_size,
        model_max_length=n_positions,
    )

    # See: https://github.com/huggingface/transformers/issues/4122#issuecomment-713433149
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = LDRTextDataset(train_lines, tokenizer)
    eval_dataset = LDRTextDataset(eval_lines, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)

    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=n_positions,)
    
    #loading pretrained model if it exisits, otherwise training new one
    if checkpoint_dir and checkpoint_dir.exists():
        model = AutoModelForCausalLM.load_pretrained(checkpoint_dir)
        logger.info(f"model loaded from {checkpoint_dir}")
    else:
        model = AutoModelForCausalLM.from_config(config)
        logger.info("training model from scratch")
    logger.info(f"# trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    #setting training arguments (most are args for main func)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        fp16=False, #setting to false bc training on Mac
        save_total_limit=save_total_limit,
        push_to_hub=False,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
    )

    #intialising trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    #training the model
    trainer.train()


if __name__ == "__main__":
    typer.run(main)

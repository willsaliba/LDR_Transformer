#standard imports
import os
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

from tokenizers import Tokenizer, trainers
from tokenizers.trainers import BpeTrainer

#NEED TO UPDATE TO TOKENIZER PATTERN (GPT-4)
FN_P = r"([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(
    rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)"
)

#loading ldr files
def load_all_ldrs(root_dir: Path, decimals: int = 2):
    """This reads all LDR files from the specified directory and rounds all floats to specified number 
    of decimals, this works well for synthetic data, be careful for real models real models."""

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

#main function
def main(
    output_dir: Path = Path("./new_logs/"),
    checkpoint_dir: T.Optional[Path] = None,
    n_positions: int = 1536,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 1000,
    save_steps: int = 10000,
    eval_steps: int = 10000,
    fp16: bool = False, #changed to false bc training on Mac
    save_total_limit: int = 5,
    learning_rate: float = 5e-4,
):
    
    #Getting the training and evaluation data
    ldr_root_dir = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/mini_data")
    train_lines = load_all_ldrs(ldr_root_dir / "train")
    eval_lines = load_all_ldrs(ldr_root_dir / "test")
    
    #Loading Tokenizer
    tokenizer_path = Path('/Users/willsaliba/Documents/code/uni/advTopics/transformer/trained_tokenizerM2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=n_positions)
    print("-- Tokenizer Loaded --")
    
    # tokenizer.train_from_iterator(files, trainer=trainer)

    # #creating training dataset
    # train_dataset = LDRTextDataset(train_lines, tokenizer)
    # eval_dataset = LDRTextDataset(eval_lines, tokenizer)

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # #will need to change
    # config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=n_positions,)
    
    # #loading pretrained model if it exisits, otherwise training new one
    # if checkpoint_dir and checkpoint_dir.exists():
    #     model = AutoModelForCausalLM.load_pretrained(checkpoint_dir)
    #     logger.info(f"model loaded from {checkpoint_dir}")
    # else:
    #     model = AutoModelForCausalLM.from_config(config)
    #     logger.info("training model from scratch")
    # logger.info(f"# trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # #setting training arguments (most are args for main func)
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
    #     num_train_epochs=num_train_epochs,
    #     per_device_train_batch_size=per_device_train_batch_size,
    #     logging_steps=logging_steps,
    #     save_steps=save_steps,
    #     eval_steps=eval_steps,
    #     fp16=fp16,
    #     save_total_limit=save_total_limit,
    #     push_to_hub=False,
    #     learning_rate=learning_rate,
    #     evaluation_strategy="steps",
    # )

    # #intialising trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    
    # #training the model
    # trainer.train()


if __name__ == "__main__":
    typer.run(main)

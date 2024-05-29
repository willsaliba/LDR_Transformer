import os
from pathlib import Path
import re
import typing as T
from concurrent.futures import ProcessPoolExecutor
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
    get_scheduler
)
import typer

#regex pattern to extract numeric values from ldr files for rounding large decimal places
FN_P = r"([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

#loads ldr and does some pre-processing
def load_all_ldrs(root_dir: Path, decimals: int = 2):
    """
    This reads all LDR files from the specified directory
    and rounds up all numeric entries to the specified number of decimals;
    the rounding part works well for synthetic data, use with care on
    real models.
    """
    
    src_files = sorted(root_dir.glob("*.mpd"))
    all_lines = []
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'):
            print(f"Skipping macOS metadata file: {src_file.name}")
            continue

        print(f"processing {src_file.name}")
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

#class to store ldr data in tensor format for quick training
class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.examples = tokenizer.batch_encode_plus(lines).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def main(
    #data, tokenizer and model directories
    model_name: str = "Omr8_m2t",
    ldr_train_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/8"),
    output_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/models/m2/m2t_omr8"),
    tokenizer_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/tokenizers/m2_tokenizers/omr8_turbo"),
    #universal model/training parameters
    checkpoint_dir: T.Optional[Path] = None,
    n_positions: int = 1536,
    save_total_limit: int = 3,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 1000,
    #true hyper params
    vlads_machine: bool = True,
    custom_tokenizer: bool = True,
    slide_50: bool = False,
):
    #setting training params based on large or small dataset
    if slide_50:
        num_train_epochs = int(10)
        save_steps, eval_steps = int(1000), int(1000)
        learning_rate = float(5e-5)
        gradient_accumulation_steps = int(4)
    else:
        num_train_epochs = int(10)
        save_steps, eval_steps = int(10000), int(10000)
        learning_rate = float(1e-5)
        gradient_accumulation_steps = int(1)

    #setting up to train GPU on either vlad machine or our apple machine
    if vlads_machine:
        fp16 = True
        logger.info(f"Training on Vlads Machine")
    else:
        fp16 = False
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        logger.info(f"Training on Mac")
    
    #loading correct tokenizer (custom vs english)
    if custom_tokenizer: tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.model_max_length = n_positions

    #loading all ldr files from the train and test directories
    train_lines = load_all_ldrs(ldr_train_dir / "train")
    eval_lines = load_all_ldrs(ldr_train_dir / "test")

    #putting ldr data into format that can train model & intiailising data collator to handle data formating during training
    train_dataset = LDRTextDataset(train_lines, tokenizer)
    eval_dataset = LDRTextDataset(eval_lines, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #using the GPT2 transformer configuriton
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=n_positions)
    model = AutoModelForCausalLM.from_config(config).to(device)
    logger.info(f"# trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    #setting training args (most are main argument)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        fp16=fp16,
        save_total_limit=save_total_limit,
        push_to_hub=False,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    #saving the trained model
    model.save_pretrained(Path(output_dir, model_name))

if __name__ == "__main__":
    typer.run(main)

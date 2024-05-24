import os
import re
import typing as T
from pathlib import Path
import numpy as np
import torch
from loguru import logger
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

#regex pattern to extract numeric values from ldr files for rounding large decimal places
FN_P = r"([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

#loads ldr and does some pre-processing
def load_all_ldrs(root_dir: Path, decimals: int = 3):
    """
    This reads all LDR files from the specified directory and rounds up numeric entries to specified
    number of decimals. Works well for synthetic data, use with care on real models.
    """
    
    src_files = list(root_dir.glob("*.ldr")) + list(root_dir.glob("*.mpd"))
    all_lines = []
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'):
            print(f"Skipping macOS metadata file: {src_file.name}")
            continue
        # print(f"processing {src_file.name}")
        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1: continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry):
                    processed.append(str(int(float(numeric_entry))))
                else:
                    processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
            processed.append(m[0][-1])  # part ID
            file_lines.append(" ".join(processed))
        all_lines.append("\n".join(file_lines))
    return all_lines

#class to store ldr data in tensor format for quick training
class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer, max_length: int):
        # Ensure that all examples are truncated to the max_length
        self.examples = tokenizer.batch_encode_plus(
            lines,
            max_length=max_length, 
            truncation=True,
            padding='max_length', 
        ).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def main(
    #data, tokenizer and model directories
    ldr_train_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/rand/rand2"),
    output_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/hugging_face/trained_model"),
    tokenizer_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/hugging_face/tokenizer_M2/trained_M2/base/rand/rand2"),
    model_name: str = "Rand2",
    #model parameters
    checkpoint_dir: T.Optional[Path] = None,
    n_positions: int = 1536,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 1000,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    fp16: bool = False,
    save_total_limit: int = 3,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    #training device
    vlads_machine: bool = False,
):
    #setting up to train GPU on either vlad machine or our apple machine
    if vlads_machine:
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        logger.info(f"Using device: {device}")

    #loading custom built and trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #model won't train without this

    #loading all ldr files from the train and test directories
    train_lines = load_all_ldrs(ldr_train_dir / "train")
    eval_lines = load_all_ldrs(ldr_train_dir / "test")

    #putting ldr data into format that can train model & intiailising data collator to handle data formating during training
    train_dataset = LDRTextDataset(train_lines, tokenizer, max_length=n_positions)
    eval_dataset = LDRTextDataset(eval_lines, tokenizer, max_length=n_positions)
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
        greater_is_better=False,
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

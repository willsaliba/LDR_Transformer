from pathlib import Path
import re
import typing as T
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_scheduler
)
import typer

"""
"""

quartonian = False
if quartonian:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")
else:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def load_all_ldrs(root_dir: Path, decimals: int = 2):
    """
    This reads all LDR files from the specified directory and rounds up all numeric entries to the 
    specified number of decimals; rounding works well for synthetic data, use with care on real models.
    """
    print("Beginning Processing all Lines")    
    src_files = sorted(root_dir.glob("*.mpd"))
    all_lines = []
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'):
            print(f"Skipping macOS metadata file: {src_file.name}")
            continue

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
    print("Completed Processing all Lines")
    return all_lines

class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.examples = tokenizer.batch_encode_plus(lines).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def main(
    #important params
    ldr_root_dir: Path = Path("data/OMR_Sorted"),
    tokenizer_dir: Path = Path("tokenizers/rand8"),
    output_dir: Path = Path("models/secnd_models"),
    model_name: str = "Omr_M2_Sorted",
    custom_tokenizer: bool = True,
    #other params
    checkpoint_dir: T.Optional[Path] = None,
    n_positions: int = 1536,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 10000,
    save_steps: int = 10000,
    eval_steps: int = 10000,
    fp16: bool = False,
    save_total_limit: int = 3,
    learning_rate: float = 1e-5
):
    train_lines = load_all_ldrs(ldr_root_dir / "train")
    eval_lines = load_all_ldrs(ldr_root_dir / "evaluation")
    if not train_lines or not eval_lines: logger.error("Training or evaluation dataset is empty. Check your data source.")

    if custom_tokenizer == True:
        print("Loading M2") 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else: 
        print("Loading GPT2") 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = n_positions

    train_dataset = LDRTextDataset(train_lines, tokenizer)
    eval_dataset = LDRTextDataset(eval_lines, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=n_positions,)
    
    if checkpoint_dir and checkpoint_dir.exists(): model = AutoModelForCausalLM.load_pretrained(checkpoint_dir)
    else: model = AutoModelForCausalLM.from_config(config)
    logger.info(f"# trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
        load_best_model_at_end=True,  
        metric_for_best_model="loss",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(Path(output_dir, model_name))

if __name__ == "__main__":
    typer.run(main)

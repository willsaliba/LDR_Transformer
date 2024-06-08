from pathlib import Path

import os
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import typer
from concurrent.futures import ProcessPoolExecutor
import re

quartonian = False

if quartonian:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")
else:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def load_80_percent_prompt(root_dir: Path, decimals: int = 2):
    """
    This reads all LDR files from the specified directory and rounds up all numeric entries to the 
    specified number of decimals; rounding works well for synthetic data, use with care on real models.
    """
    src_files = sorted(root_dir.glob("*.mpd")) + sorted(root_dir.glob("*.ldr"))
    all_lines = []
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'): continue

        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1: continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry): processed.append(str(int(float(numeric_entry))))
                else: processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
            processed.append(m[0][-1])  # part ID
            file_lines.append(" ".join(processed))
            if len(file_lines) == 4: break
        all_lines.append("\n".join(file_lines))

    return all_lines[0]

def main(
    ldr_root_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/mini/"),
    checkpoint_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/models/OMR8_M2T"),
    tokenizer_params_path: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/tokenizers/omr8_turbo"),
    max_new_tokens: int = 500,
    top_k: int = 51,
    top_p: float = 0.85,
    do_sample: bool = True,
    temp_file_path: Path = Path("outputs/a_out.ldr"),
    output_file_path: Path = Path("outputs/a_gen.ldr"),
    n_positions: int = 1536,
):
    #loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.model_max_length = n_positions
    print("Tokenizer Loaded")

    #loading model
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).eval() #.to(device)
    generation_config = GenerationConfig(
        max_length=model.config.n_positions,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Model Loaded")

    #setting prompt
    text_prompt = load_80_percent_prompt(ldr_root_dir)
    text_prompt += "\n"

    #comverting prompt to tensor and generating output
    prompt = torch.as_tensor([tokenizer.encode(text_prompt)]) #.to(device)
    out = model.generate(prompt, generation_config=generation_config)
    print("Output Generated")
    
    #decoding output and saving file to temp
    decoded = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    temp_file_path.write_text(decoded)
    with open(temp_file_path, 'r') as infile: lines = infile.readlines()

    #processing file so that it doesnt contain broken code
    filtered_lines = ["0 FILE LDR_TRANSFORMER_OUTPUT.ldr\n", "0 Main\n", "0 Name: TEST_OUTPUT_ASSEMBLY.ldr\n", "0 Author: WillZach_MODEL\n"]
    for line in lines:
        entries = line.split()
        #processing last line which for some reason trails on
        if len(filtered_lines) == 11 and entries[0] == '1' and entries[14].endswith('.dat'):
            final_line = " ".join(entries[:15]) + "\n"
            filtered_lines.append(final_line)
            break
        elif (len(entries) == 15 and entries[0] == '1' and entries[-1].endswith('.dat')):
            filtered_lines.append(line)

    #saving final file
    with open(output_file_path, 'w') as outfile: outfile.writelines(filtered_lines)

if __name__ == "__main__":
    typer.run(main)
 
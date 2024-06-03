import os
import re
import typer
from pathlib import Path
import numpy as np
import shutil
import random
from pathlib import Path

quartonian = False
if quartonian:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")
else:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def load_all_ldrs(root_dir: Path, save_dir: Path, decimals: int = 2):
    print("Beginning Processing all Lines")    
    
    src_files = sorted(root_dir.glob("*.mpd"))
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'): continue

        #processing individual file lines
        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1: continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry):processed.append(str(int(float(numeric_entry))))
                else: processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
            processed.append(m[0][-1])  # part ID
            processed.append("\n")
            file_lines.append(" ".join(processed))
            
        #save new file
        output_file_path = Path(save_dir, src_file.name)
        with open(output_file_path, 'w') as outfile: outfile.writelines(file_lines)
        print("saved file: ", src_file.name)

    print("Completed Processing all Lines")

def rand_1000_files(root_dir: Path, save_dir: Path):
    # Get all .mpd files in the root directory
    all_mpd_files = list(root_dir.glob('*.mpd'))
    
    # Ensure there are more than 1000 files
    if len(all_mpd_files) < 1100:
        raise ValueError("There are less than 1000 .mpd files in the root directory")
    
    # Randomly select 1000 files
    selected_files = random.sample(all_mpd_files, 1100)
    
    # Copy the selected files to the save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    for file in selected_files:
        shutil.copy(file, save_dir)

def main():
    root_dir = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/omr8_clean/train")
    save_dir = Path("/Users/willsaliba/Documents/code/uni/advTopics/data/omr8_clean/eval_set")
    # load_all_ldrs(root_dir, save_dir)
    rand_1000_files(root_dir, save_dir)

if __name__ == "__main__":
    typer.run(main)
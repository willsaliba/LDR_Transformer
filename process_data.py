import os
import re
import typer
from pathlib import Path
import numpy as np
import shutil
import random
from pathlib import Path
from sklearn.decomposition import PCA
import math

"""
This file contains a range of functions used at various points in the research process to preprocess
and manipulate the datasets as well as perform analysis on the datasets. 

These include:
- initial_preprocessing: removes all metadata from files and rounds floating point entries to 2 decimal places
- create_test_dataset: randomly selects 1100 files to create the test dataset
- count_line_types: counts the frequency of each line type in a dataset
- create_sorted_dataset: creates copy of dataset where individual brick lines sorted based on position
- create_quaternion_dataset: creates a dataset where rotation matrices are converted to quaternions
"""

FN_P = r"([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def initial_preprocessing(root_dir: Path, save_dir: Path, decimals: int = 2):
    #This function removes all metadata from files and round floating point entries to 2 decimal places
    src_files = list(root_dir.glob("*.ldr")) + list(root_dir.glob("*.mpd")) 
    for src_file in src_files:
        if src_file.name.startswith('._'): continue
        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1: continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry):processed.append(str(int(float(numeric_entry))))
                else: processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
            processed.append(m[0][-1])
            processed.append("\n")
            file_lines.append(" ".join(processed))
            
        #save new file
        output_file_path = Path(save_dir, src_file.name)
        with open(output_file_path, 'w') as outfile: outfile.writelines(file_lines)
        print("saved file: ", src_file.name)

def create_test_dataset(root_dir: Path, save_dir: Path):
    #this function randomly selects 1100 files to create the test dataset
    all_files = list(root_dir.glob('*.ldr')) + list(root_dir.glob('*.mpd'))
    if len(all_files) < 1100: raise ValueError("There are less than 1000 .mpd files in the root directory")
    selected_files = random.sample(all_files, 1100)
    save_dir.mkdir(parents=True, exist_ok=True)
    for file in selected_files:
        shutil.copy(file, save_dir)

def count_line_types(root_dir: Path):
    #this function counts the frequency of each line type in a dataset
    all_files = list(root_dir.glob('*.ldr')) + list(root_dir.glob('*.mpd'))
    counts = [0] * 6
    for file in all_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = int(line[0])
                counts[value] += 1
    print(counts)

def find_distance(point1, point2):
    x1, y1, z1 = map(float, point1)
    x2, y2, z2 = map(float, point2)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def create_sorted_dataset(root_dir: Path, save_dir: Path):
    all_files = list(root_dir.glob("*.ldr")) + list(root_dir.glob("*.mpd")) 

    for file in all_files:
        with open(file, 'r') as f:
            lines = f.readlines()

            #determine central axis and create split lines for sorting
            split_lines = []
            x_sum, z_sum = 0, 0
            for line in lines:
                entries = line.split()
                x_sum += float(entries[2])
                z_sum += float(entries[4])
                split_lines.append(entries)
            x_mean, z_mean = x_sum / len(lines), z_sum / len(lines)

            #sort lines, first by y value, then by distance from central axis
            sorted_lines = sorted(split_lines, key=lambda x: (
                -float(x[3]), 
                find_distance((x_mean, x[3], z_mean), (x[2], x[3], x[4]))
            ))
            final_lines = [" ".join(line) + '\n' for line in sorted_lines]
            #save new file
            save_path = Path(save_dir, file.name)
            with open(save_path, 'w') as outfile: outfile.writelines(final_lines)
            print("saved file: ", file.name)


def create_quaternion_dataset(root_dir: Path, save_dir: Path):
    pass


def main():
    root_dir = Path("data/rand8_clean/train")
    save_dir = Path("data/RAND_Sorted/train")

    create_sorted_dataset(root_dir, save_dir)

if __name__ == "__main__":
    typer.run(main)

"""
omr8
pre meta wipe: counts = [112356, 223384, 0, 0, 0, 0]
post meta wipe: counts = [0, 223384, 0, 0, 0, 0]

rand8
pre meta wipe: counts = [200000, 400000, 0, 0, 0, 0]
post meta wipe: counts = [0, 400000, 0, 0, 0, 0]
"""
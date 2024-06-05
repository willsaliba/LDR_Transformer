from pathlib import Path
import typer
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R
import re
import itertools
from multiset import Multiset
import pandas as pd

quartonian = False
if quartonian:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")
else:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def parse_ldr_lines(lines, decimals=2):
    assembly = {
        'shape': [],
        'color': [],
        'position': [],
        'orientation': [],
        'pose': [],
        'edges': ([], [])
    }

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15 or parts[0] != '1':
            continue

        color = int(parts[1])
        shape_file = parts[-1]
        position = np.array(list(map(float, parts[2:5])))
        orientation_matrix = np.array(list(map(float, parts[5:14]))).reshape((3, 3))

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = orientation_matrix
        pose_matrix[:3, 3] = position

        assembly['color'].append(color)
        assembly['shape'].append(shape_file)
        assembly['position'].append(position)
        assembly['orientation'].append(orientation_matrix)
        assembly['pose'].append(pose_matrix)

    assembly['pose'] = np.array(assembly['pose'])
    return assembly

def round_line(line, decimals=2):
    m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
    if len(m) != 1: return line
    processed = []
    for numeric_entry in m[0][:-1]:
        if int(float(numeric_entry)) == float(numeric_entry): processed.append(str(int(float(numeric_entry))))
        else: processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
    processed.append(m[0][-1])
    return " ".join(processed)

def position_accuracy(predicted, ground_truth):
    try:
        if len(predicted['position']) == 0 or len(ground_truth['position']) == 0: return None
        predicted_positions = np.array(predicted['position'])
        ground_truth_positions = np.array(ground_truth['position'])
        mse = np.mean((predicted_positions - ground_truth_positions) ** 2)
        return mse
    except:
        print("Error calculating position accuracy")
        return None

def geodesic_distance(R1, R2):
    return np.linalg.norm(logm(np.dot(R1.T, R2)), 'fro')

def rotation_matrix_to_quaternion(matrix):
    rotation = R.from_matrix(matrix)
    return rotation.as_quat()

def quaternion_distance(Q1, Q2):
    dot_product = np.dot(Q1, Q2)
    return 1 - np.abs(dot_product)

def orientation_accuracy(predicted, ground_truth):
    try:
        if len(predicted['orientation']) == 0 or len(ground_truth['orientation']) == 0:
            return None
        predicted_orientations = np.array(predicted['orientation'])
        ground_truth_orientations = np.array(ground_truth['orientation'])
        distances = [
            geodesic_distance(pred, gt)
            for pred, gt in zip(predicted_orientations, ground_truth_orientations)
        ]
        avg_distance = np.mean(distances)
        return avg_distance
    except:
        return None

def quaternion_orientation_accuracy(predicted, ground_truth):
    try:
        if len(predicted['orientation']) == 0 or len(ground_truth['orientation']) == 0:
            return None
        predicted_orientations = np.array(predicted['orientation'])
        ground_truth_orientations = np.array(ground_truth['orientation'])
        distances = [
            quaternion_distance(
                rotation_matrix_to_quaternion(pred),
                rotation_matrix_to_quaternion(gt)
            )
            for pred, gt in zip(predicted_orientations, ground_truth_orientations)
        ]
        avg_distance = np.mean(distances)
        return avg_distance
    except:
        return None

def color_accuracy(predicted, ground_truth):
    try:
        if len(predicted['color']) == 0 or len(ground_truth['color']) == 0:
            return None
        predicted_colors = np.array(predicted['color'])
        ground_truth_colors = np.array(ground_truth['color'])
        correct_colors = np.sum(predicted_colors == ground_truth_colors)
        total_colors = len(ground_truth['color'])
        accuracy = correct_colors / total_colors
        return accuracy
    except:
        return None

def shape_accuracy(predicted, ground_truth):
    try:
        if len(predicted['shape']) == 0 or len(ground_truth['shape']) == 0:
            return None
        predicted_shapes = np.array(predicted['shape'])
        ground_truth_shapes = np.array(ground_truth['shape'])
        correct_shapes = np.sum(predicted_shapes == ground_truth_shapes)
        total_shapes = len(ground_truth['shape'])
        accuracy = correct_shapes / total_shapes
        return accuracy
    except:
        return None

def color_set_accuracy(predicted, ground_truth):
    predicted_colors = Counter(predicted['color'])
    target_colors = Counter(ground_truth['color'])
    correct = 0
    for color in predicted_colors:
        if color in target_colors: correct += min(predicted_colors[color], target_colors[color])
    return correct / len(ground_truth['color'])

def shape_set_accuracy(predicted, ground_truth):
    predicted_shapes = Counter(predicted['shape'])
    target_shapes = Counter(ground_truth['shape'])
    correct = 0
    for color in predicted_shapes:
        if color in target_shapes: correct += min(predicted_shapes[color], target_shapes[color])
    return correct / len(ground_truth['shape'])

def f1b(predicted, ground_truth):
    try:
        predicted_bricks = Multiset(zip(predicted['shape'], predicted['color']))
        if (0, 0) in predicted_bricks: predicted_bricks.remove((0, 0))
        ground_truth_bricks = Multiset(zip(ground_truth['shape'], ground_truth['color']))
        if (0, 0) in ground_truth_bricks: ground_truth_bricks.remove((0, 0))
        tp = len(predicted_bricks & ground_truth_bricks)
        fp = len(predicted_bricks - ground_truth_bricks)
        fn = len(ground_truth_bricks - predicted_bricks)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1_score
    except:
        return None

def try_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check_line(entries):
    if len(entries) < 15: return False
    if entries[0] != '1': return False
    if not entries[1].isdigit(): return False
    for i in range(2, 13):
        if not try_convert_to_float(entries[i]): return False
    if not entries[14].endswith('.dat'): return False
    return True

def evaluate_file(file_path, tokenizer, model, device):
    print(f"Processing file: {file_path}")
    try:
        #reading in lines and ensuring there is the correct number of lines
        with open(file_path, 'r', encoding='utf-8') as file: all_lines = [round_line(line) for line in file.readlines()]
        if len(all_lines) != 8: 
            print(f"Invalid number of lines in file {file_path}: {len(all_lines)}\n")
            return None

        # Get prompt and evaluation bricks, ensuring we don't go out of bounds
        prompt_bricks, eval_bricks = all_lines[:6], all_lines[6:8]
        prompt_text = "\n".join(prompt_bricks) + "\n"

        #converting prompt to tensor and generating output
        prompt = tokenizer(prompt_text, return_tensors='pt')
        outputs = model.generate(
            prompt.input_ids.to(device),
            attention_mask=prompt.attention_mask.to(device),
            max_length=1516,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_lines = [round_line(line.strip()) for line in decoded_output.split("\n")]

        #filtering out lines that are not bricks
        filtered_lines = []
        for line in generated_lines:
            entries = line.split()
            if check_line(entries) == False: return None
            #processing last line which for some reason trails on
            if len(filtered_lines) == 7 and entries[0] == '1' and entries[14].endswith('.dat'):
                filtered_lines.append(" ".join(entries[:15]) + "\n")
                break
            elif (len(entries) == 15 and entries[0] == '1' and entries[-1].endswith('.dat')):
                filtered_lines.append(line)
        if len(filtered_lines) != 8: return None
        generated_bricks = filtered_lines[6:8]
        
        #inspecting results and converting to parasble format
        print(f"target: {eval_bricks}")
        print(f"result: {generated_bricks}")
        generated_assembly = parse_ldr_lines(generated_bricks)
        target_assembly = parse_ldr_lines(eval_bricks)

        #calculating results for current file
        position_acc = position_accuracy(generated_assembly, target_assembly)
        orientation_acc, quaternion_orientation_acc = orientation_accuracy(generated_assembly, target_assembly), quaternion_orientation_accuracy(generated_assembly, target_assembly)
        color_acc, shape_acc = color_accuracy(generated_assembly, target_assembly), shape_accuracy(generated_assembly, target_assembly)
        color_set_acc, shape_set_acc = color_set_accuracy(generated_assembly, target_assembly), shape_set_accuracy(generated_assembly, target_assembly)
        f1b_score = f1b(generated_assembly, target_assembly)
        print(f"Pos: {position_acc:.3f} Orien: {quaternion_orientation_acc:.3f} Color: {color_acc:.3f} Shape: {shape_acc:.3f} ColorSet: {color_set_acc:.3f} ShapeSet: {shape_set_acc:.3f} F1B: {f1b_score:.3f}\n")

        return {
            'position_acc': position_acc,
            'orientation_acc': orientation_acc,
            'quaternion_orientation_acc': quaternion_orientation_acc,
            'color_acc': color_acc,
            'shape_acc': shape_acc,
            'color_set_acc': color_set_acc,
            'shape_set_acc': shape_set_acc,
            'f1b_score': f1b_score
        }

    except:
        print(f"Failed to evaluate file {file_path.name}\n")
        return None

def main(
    model_dir: Path = Path("models/first_models/OMR8_ENG"), 
    tokenizer_path: Path = Path(""), 
    test_files_path: Path = Path("data/mini_clean"), 
    csv_filename: str = "generation/results/TEST.csv",
    custom_tokenizer: bool = False,
    n_positions: int = 1536,
    num_test_files: int = 1000,
):
    #intialising model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    print(f"---Model Loaded: {model_dir.name}---")

    #loading tokenizer
    if custom_tokenizer == True:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"---Tokenizer Loaded: {tokenizer_path.name}---")
    else: 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("---Loaded GPT2---") 
    tokenizer.model_max_length = n_positions
    
    #processing all files to get all results
    print(f"---Evaluating files in: {test_files_path.name}---\n")
    all_results = []
    failed = 0
    for file_path in itertools.islice(test_files_path.glob("*.mpd"), num_test_files+100):
        if len(all_results) == num_test_files: break
        if len(all_results) % 20 == 0 and len(all_results) != 0: print(f"\n---Evaluated: {len(all_results)} ---\n")
        result = evaluate_file(file_path, tokenizer, model, device)
        if result != None: all_results.append(result)
        else: failed += 1
    print(f"---Evaluated {len(all_results)} files---")
    if len(all_results) == 0: return

    #getting mean of results
    avg_position_acc = np.mean([r['position_acc'] for r in all_results if r['position_acc'] is not None])
    avg_orientation_acc = np.mean([r['orientation_acc'] for r in all_results if r['orientation_acc'] is not None])
    avg_quaternion_orientation_acc = np.mean([r['quaternion_orientation_acc'] for r in all_results if r['quaternion_orientation_acc'] is not None])
    avg_color_acc = np.mean([r['color_acc'] for r in all_results if r['color_acc'] is not None])
    avg_shape_acc = np.mean([r['shape_acc'] for r in all_results if r['shape_acc'] is not None])
    avg_color_set_acc = np.mean([r['color_set_acc'] for r in all_results if r['color_set_acc'] is not None])
    avg_shape_set_acc = np.mean([r['shape_set_acc'] for r in all_results if r['shape_set_acc'] is not None])
    avg_f1b_score = np.mean([r['f1b_score'] for r in all_results if r['f1b_score'] is not None])

    #printing results
    print("\n---RESULTS:---\n")
    print("Custom Metrics:")
    print(f"Ave Position Accuracy (MSE): {avg_position_acc:.4f}")
    print(f"Ave Orientation Accuracy (Q): {avg_quaternion_orientation_acc:.4f}")
    # print(f"Ave Orientation Accuracy (GD): {avg_orientation_acc:.4f}")
    print(f"Ave Color Accuracy: {avg_color_acc:.2f}")
    print(f"Ave Shape Accuracy: {avg_shape_acc:.2f}\n")

    print("Set Metrics:")
    print(f"Ave Color Set Accuracy: {avg_color_set_acc:.2f}")
    print(f"Ave Shape Set Accuracy: {avg_shape_set_acc:.2f}\n")

    print("B&M Metrics:")
    print(f"Ave F1B Score: {avg_f1b_score:.4f}")
    print(f"Failed Files: {failed}\n")

    # Save average results to a CSV file
    averages = {
        'Metric': [
            'Position Accuracy (MSE)', 
            'Orientation Accuracy (Geodesic Distance)', 
            'Quaternion Orientation Accuracy', 
            'Color Accuracy', 
            'Shape Accuracy', 
            'Color Set Accuracy', 
            'Shape Set Accuracy', 
            'F1B Score',
            'Failed Genrations'
        ],
        'Average': [
            avg_position_acc, 
            avg_orientation_acc, 
            avg_quaternion_orientation_acc, 
            avg_color_acc, 
            avg_shape_acc, 
            avg_color_set_acc, 
            avg_shape_set_acc, 
            avg_f1b_score,
            failed
        ]
    }
    averages_df = pd.DataFrame(averages)
    averages_df.to_csv(csv_filename, index=False)
    print(f"Average results saved to {csv_filename}\n")

if __name__ == "__main__":
    typer.run(main)



    # model_dir: Path = Path("OMR8_ENG/omr8_eng"), 
    # tokenizer_path: Path = Path("advTopics/tokenizers/"), 
    # test_files_path: Path = Path("data/omr8/eval_set"), 
    # csv_filename: str = "o8_eng_res.csv",

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import typer

def main(
    checkpoint_dir: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/models/OMR8_M2T"),
    tokenizer_params_path: Path = Path("/Users/willsaliba/Documents/code/uni/advTopics/tokenizers/m2_tokenizers/omr8_turbo"),
    max_new_tokens: int = 500,
    top_k: int = 51,
    top_p: float = 0.85,
    do_sample: bool = True,
    output_file_path: Path = Path("gen.ldr"),
    n_positions: int = 1536,
):
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.model_max_length = n_positions

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).eval().to(device)
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
    prompt = torch.as_tensor([tokenizer.encode("1")]).to(device)
    out = model.generate(prompt, generation_config=generation_config)
    decoded = tokenizer.decode(
        out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    output_file_path.write_text(decoded)


if __name__ == "__main__":
    typer.run(main)
 
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import typer


def main(
    checkpoint_dir: Path,
    tokenizer_params_path: Path = Path("./tokenizer/"),
    max_new_tokens: int = 256,
    top_k: int = 50,
    top_p: float = 0.85,
    do_sample: bool = True,
    output_file_path: Path = Path("gen.ldr"),
):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
    # See: https://github.com/huggingface/transformers/issues/4122#issuecomment-713433149
    tokenizer.pad_token = tokenizer.eos_token
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

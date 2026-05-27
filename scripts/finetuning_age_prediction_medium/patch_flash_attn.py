"""
Patch scGPT FlashTransformerEncoderLayer to work with flash_attn 2.x:
  1. Replace attention_dropout= with dropout=
  2. Remove batch_first= (not supported in MHA 2.x)
  3. Add use_flash_attn=True (disabled by default in 2.x!)
Run once on the cluster before submitting run_medium_49k.sbatch.
"""

from pathlib import Path

MODEL_PATH = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/external/MethylGPT/methylgpt/modules/scGPT/scgpt/model/model.py")

OLD = """        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            **factory_kwargs,
        )"""

NEW = """        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=True,
            **factory_kwargs,
        )"""

code = MODEL_PATH.read_text(encoding="utf-8", errors="replace")

if NEW in code:
    print("Already patched — nothing to do.")
elif OLD in code:
    MODEL_PATH.write_text(code.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched OK: {MODEL_PATH}")
else:
    print("ERROR: Pattern not found. Show me lines 638-650:")
    lines = code.splitlines()
    for i, line in enumerate(lines[635:655], start=636):
        print(f"{i}: {line}")

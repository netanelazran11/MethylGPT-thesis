"""
Patch scGPT model.py to support flash_attn 2.x (FlashMHA moved to flash_attn.modules.mha).
Run once on the cluster before submitting run_medium_49k.sbatch.
"""

from pathlib import Path

MODEL_PATH = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/external/MethylGPT/methylgpt/modules/scGPT/scgpt/model/model.py")

OLD = """try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False"""

NEW = """try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    try:
        from flash_attn.modules.mha import FlashMHA  # flash_attn 2.x
        flash_attn_available = True
    except ImportError:
        import warnings
        warnings.warn("flash_attn is not installed")
        flash_attn_available = False"""

code = MODEL_PATH.read_text(encoding="utf-8", errors="replace")

if NEW in code:
    print("Already patched — nothing to do.")
elif OLD in code:
    MODEL_PATH.write_text(code.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched OK: {MODEL_PATH}")
else:
    print("ERROR: Pattern not found — file may have changed. Check manually.")

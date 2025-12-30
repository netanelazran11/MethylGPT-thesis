# NotImplemented - The code will be updated soon.
# MethylGPT CpG Site Imputation

MethylGPT enables imputation of CpG site methylation values. Below are detailed instructions for inferring imputed CpG sites using MethylGPT.

---

## 📌 Background

MethylGPT is pre-trained on a defined list of CpG sites available here:
- [Pre-trained CpG list](https://github.com/albert-ying/MethylGPT/blob/main/tutorials/pretraining/probe_ids_type3.csv)

Depending on the presence of your target CpG site in this list, follow the respective scenarios below:

---

## 🚩 Scenario 1: Target CpG is in Training List

If your target CpG site exists in the pre-trained list, follow these steps:

1. **Locate your CpG site** in your dataset.
2. **Modify data preprocessing:**
   - Refer to line 283 in [pretraining.py](https://github.com/albert-ying/MethylGPT/blob/main/methylgpt/pretraining.py)
   - Add target CpG positions to the `masked_positions` (**not** `imputed_positions`).
3. **Replace original CpG values** with the designated `mask_value`.
4. **Run inference:** Execute MethylGPT to infer imputed values, which will be output directly.

---

## 🚧 Scenario 2: Target CpG is NOT in Training List

If the target CpG site was not included during initial training, additional fine-tuning is required:

1. **Add the new CpG site** to the existing model vocabulary.
2. **Fine-tune MethylGPT** with new data containing this CpG site to enable representation learning.
3. **Post fine-tuning:**
   - Follow the same instructions as **Scenario 1**.
   - Add target CpG positions to `masked_positions`.
   - Replace values with `mask_value`.
   - Execute inference to get imputed results.

---

Following these steps, you can effectively utilize MethylGPT for accurate CpG site methylation imputation, irrespective of its initial presence in the training dataset.

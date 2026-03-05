## Failure-Aware SAE for VLMs

This project build a Failure-aware SAE for VLMs, which can improve the robustness of VLM.



### 1. Prepare Datasets (ImageNet-1K / CIFAR-10 / CIFAR-100)

First, download and organize the datasets locally:

- ImageNet-1K
- CIFAR-10
- CIFAR-100

Make sure dataset paths in scripts are correctly configured before running the next steps.

---

### 2. Generate Adversarial Samples

Use the black-box patch attack script to generate attacked images:

```bash
python /data/attack.py
```

This step produces adversarial examples that will later be evaluated by VLM classifiers.

And all of the OOD datasets, you can download from this link https://zenodo.org/records/2535967 .

---

### 3. Split Samples into VLM success / fail

Run VLM classifiers to separate images based on recognition outcomes:

```bash
python /data/vlm_adv_id_classifier.py
python /data/vlm_ood_classifier.py
```

Outputs are grouped into two categories:

- success: VLM predicts correctly
- fail: VLM predicts incorrectly

These labels are used as supervision for downstream failure-aware learning.

---

### 4. Train the Failure-Aware Prediction Head

Train the prediction head with online Top-K SAE features:

```bash
nohup python /home/lzx/patchsae/train_error_mlp_onlinetopksae_training.py \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 8 \
  --topk 300 \
  --lr 3e-4 \
  --sae-lr 1e-4 \
  --lambda 0.5 \
  --seed 0 \
  --device cuda:0 
```

---

### 5. Summary
In short, our pipeline is:

1. Dataset preparation
2. Adversarial data generation
3. VLM fail/success split
4. Failure-aware head training

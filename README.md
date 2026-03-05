## End-to-End Pipeline: Failure-Aware Prediction for VLMs

This project follows a three-stage pipeline to build a **failure-aware prediction head** on top of CLIP-based features.



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

This step produces adversarial examples that will later be evaluated by CLIP-based classifiers.

---

### 3. Split Samples into CLIP success / fail

Run CLIP classifiers to separate images based on recognition outcomes:

```bash
python /data/clip_adv_id_classifier.py
python /data/clip_ood_classifier.py
```

Outputs are grouped into two categories:

.success: CLIP predicts correctly
.fail: CLIP predicts incorrectly

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
3. CLIP-based fail/success split
4. Failure-aware head training

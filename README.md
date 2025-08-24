
# 🧠 Egocentric Video Understanding using OpenCLIP on Infant-Homeview Dataset

This project is part of an applied research initiative at **Indiana University Luddy School of Informatics**. It explores contrastive image-language modeling over **100,000+ hours of egocentric baby videos** captured in real-world home environments.

---

## 📦 Dataset Overview

> **Note**: This is *not the standard SAYCam dataset*. The dataset was provided as part of an internal IU research project.

- **Source**: Raw egocentric videos (infant headcam)
- **Duration**: ~100k+ hours of footage
- **Processing Pipeline**:
  - 🎞️ Frame extraction using `ffmpeg` (1 FPS to 5 FPS)
  - 🔊 Audio transcription using `Whisper AI` (OpenAI) to generate weak supervision labels
  - 🧊 Tar-based storage using `WebDataset` format (sharded to ~4,000 `.tar` files)
  - 🤖 Sample format: `{image, text}` pairs per frame

---

## ⚙️ Technical Infrastructure

- **Compute**: [Big Red 200 Supercomputer](https://kb.iu.edu/d/avjq) (IU HPC cluster)
- **GPU**: NVIDIA A100 (×4) across multiple SLURM jobs
- **Storage**: Lustre parallel filesystem (15+ TB raw data)
- **Job Management**: SLURM (`sbatch`, `srun`, `torchrun`)

---

## 🛠️ Pretraining Pipeline

- ✅ Custom data loaders for `WebDataset` format  
- ✅ Multi-GPU distributed training via `torchrun`  
- ✅ Mixed precision training with AMP (`--precision amp`)  
- ✅ Model logging + monitoring via `Weights & Biases`  
- ✅ Validation loss graph and Clip Loss tracking implemented

---

## 🔍 Models Trained

We trained **multiple OpenCLIP backbones** end-to-end on the Infant-Homeview dataset:

| Model Name     | Command Flag         | Parameters    |
|----------------|----------------------|---------------|
| ResNet-50      | `--model RN50`       | 102M          |
| ResNet-101     | `--model RN101`      | 130M          |
| ViT-B/32       | `--model ViT-B-32`   | 151M          |
| ViT-B/16       | `--model ViT-B-16`   | 151M          |
| ViT-L/14       | `--model ViT-L-14`   | 428M          |

⏱️ We used **gradual scaling** based on GPU availability and convergence trends. Results improved with larger backbones on the domain-specific egocentric dataset.

---

## 📈 Key Achievements

- 🔄 Complete training-validation split over 4M image-text pairs
- 🔥 Achieved stable Clip Loss reductions with `ViT-B-32` and `ViT-L-14`
- 🎯 Validated on unseen real-world baby-view frames (custom benchmark)
- 📉 Loss convergence observed at scale (see W&B graphs)
- 🧪 Verified image-text matching via inference on unseen samples
- 🧠 Grounded text captions from Whisper-based pseudo-labels

---

## 🧾 Citation / Acknowledgements

This project builds upon:

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [WebDataset](https://github.com/webdataset/webdataset)
- Indiana University HPC Resources

> All data and experiments are conducted under ethical research guidelines and internal research agreements.

---

## 📁 Repository Structure

```bash
├── open_clip_train/       # Custom scripts with val loss, batch logs etc.
├── data_pipeline/         # FFmpeg + Whisper preprocessing scripts
├── job_scripts/           # SLURM sbatch scripts (bash)
├── logs/                  # W&B or local logs
├── README.md              # This file
└── clipgpu_test1.py       # CLIP inference sanity check
```

---

## 🧩 Repository Strategy

Since this work builds on `open_clip`, but your fork was from `openai/CLIP`, we recommend:

> ✅ **Create a new GitHub repo** (e.g., `openclip-homeview-egocentric`)  
> ✅ Push your full working codebase there  
> ✅ Include logs, visuals, and checkpoints selectively

This will clarify your contribution and distinguish your repo from the older OpenAI CLIP release.

---

## 🚀 Future Steps

- Add `inference.py` to evaluate your trained checkpoints
- Push sample `.tar` files (5–10) to test reproducibility
- Optional: Export graphs or logs from W&B for results

---

### Author: Yeshwanth Satheesh 

# 🧠 Egocentric Infant Vision Understanding via OpenCLIP  
*Fine-Tuning Vision Transformers on Homeview Baby Dataset*

[![OpenCLIP Fork](https://img.shields.io/badge/Forked%20from-openai%2FCLIP-blue)](https://github.com/openai/CLIP)  
[![Big Red GPU](https://img.shields.io/badge/Compute-BigRed%20Supercomputer-red)](https://kb.iu.edu/d/bfqu)

---

## 📍 Overview

This repository extends the OpenCLIP training framework to process and fine-tune Vision Transformers (ViT, ResNet backbones, etc.) on a large-scale, **egocentric video dataset of infant environments** captured in natural home settings. The dataset is a derivative of **100K+ hours** of raw video footage, curated, cleaned, and converted into a WebDataset tar format to support scalable training.

> 🧪 This project is part of a research initiative at **Luddy School of Informatics, Indiana University (IU)** and operates on non-public datasets from the Infant Homeview study, distinct from public datasets like SAYCam.

---

## 🧪 Research Pipeline

### 🔁 Egocentric Preprocessing + Vision-Language Training

```mermaid
flowchart TD
  A[Raw Egocentric Videos (100K+ hrs)] --> B[Frame Extraction with FFmpeg (1 frame/sec)]
  B --> C[Audio Transcription using Whisper AI]
  C --> D[Data Cleaning, Filtering, Labeling]
  D --> E[WebDataset Sharding (.tar format)]
  E --> F[OpenCLIP Training]
  F --> G[Model Checkpoints + Evaluation]
  G --> H[wandb Logging + GPU-based Inference]

  🧠 Models Used
	•	✅ ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
	•	✅ Self-supervised DINO Pretraining used in alternate runs
	•	✅ Fully trained using torchrun + SLURM on IU Big Red (NVIDIA A100 GPUs)


📊 Results
	•	Achieved strong label alignment between images and baby-centric context (e.g., toys, animals, rooms).
	•	🧠 Validation accuracy and loss showed stable convergence.
	•	All runs tracked with Weights & Biases (wandb) including both train/ and val/ loss.
	•	Explored multiple splits: 80-20, 75-25 for train-val.

🗂️ Sample Project Repository Structure

open_clip/                     # Forked + Customized Training Framework
├── src/
│   ├── open_clip_train/       # Training CLI wrapper and utils
│   └── open_clip/             # Model definition, tokenizers, pretrained loaders

/data/
├── cleaned_file_frame_dataset_tar/   # Sharded WebDataset tar files (train + val)

logs/
├── clip_homeviewrun*          # Multiple model logs (per run/model)
└── wandb/                     # Optional local wandb run logs

scripts/
├── train_slurm_clip.sh         # Slurm training script
├── infer_single_image.py       # Image+text inference demo
├── whisper_transcribe.py       # Whisper transcription script
└── extract_frames_ffmpeg.sh    # Frame extraction via FFmpeg


---

## ⚙️ Scripts & Examples

For full reproducibility and implementation reference, see the scripts below:

| Script Name                          | Purpose                                      |
|-------------------------------------|----------------------------------------------|
| `scripts/train_slurm_clip.sh`       | SLURM job script to train OpenCLIP models    |
| `scripts/infer_single_image.py`     | Inference on a single image with custom texts|
| `scripts/whisper_transcribe.py`     | Transcribe egocentric audio using Whisper AI |
| `scripts/extract_frames_ffmpeg.sh`  | Extract video frames using FFmpeg (1 fps)    |

📁 See [**scripts/**](./scripts/) folder for complete code examples.
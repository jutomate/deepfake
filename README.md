# Deepfake Technology Proof of Concept (POC)
## Overview
This repository contains an end-to-end Proof of Concept (POC) evaluating open-source deepfake synthesis and detection pipelines executed locally on consumer hardware without dedicated NVIDIA CUDA accelerators.
## Hardware and Runtime Profile
- Platform: Windows 11 x64
- Compute Architecture: AMD Radeon Integrated Graphics (CPU execution / DirectML)
- Python Runtime: Python 3.10
## Architecture and Components
### Category 1: Visual Manipulation (FaceFusion)
- Tool: FaceFusion 3.9.1
- Model: hyperswap_1a_256
- Pipeline: Extracts facial landmarks, swaps source identity onto the target subject, and outputs the visual stream.
### Category 2: Voice Synthesis (Coqui XTTSv2)
- Tool: Coqui XTTSv2
- Script: 02_audio_xtts/clone.py
- Architecture: Zero-shot voice cloning using a short reference audio sample.
- Resolution / Workaround: Employs a custom SoundFile loader to circumvent Windows torchcodec FFmpeg dynamic library conflicts on CPU.
### Category 3: Visual Forgery Detection (Deepfake ViT)
- Architecture: Vision Transformer (ViT)
- Model: dima806/deepfake_vs_real_image_detection
- Script: 03_detection/detect.py
- Pipeline: Extracts sampled frames (1 FPS) from both authentic and manipulated footage, assesses pixel boundary artifacts, and calculates aggregate forgery probability scores.
- Benchmark Results:
  - Original Video Forgery Score: 8.21% (PASS / Authentic)
  - Doctored Video Forgery Score: 0.09% (EVASION / False Negative)
  - Forensic Takeaway: Full-frame spatial classification fails against blended face swaps; facial cropping and temporal analysis are required.
## Artifact Merging
The final output (doctored_final.mp4) multiplexes the face-swapped video stream with the cloned vocal audio track using FFmpeg stream mapping without re-encoding quality degradation.
## Directory Layout
- 01_facefusion/: Configuration and setup for visual swapping.
- 02_audio_xtts/: Audio synthesis scripts and voice sample references.
- 03_detection/: Frame extraction pipeline and comparative detection benchmark.
- .gitignore: Excludes virtual environments, model weights, and media artifacts. 

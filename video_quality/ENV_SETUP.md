# WorldArena Video Quality – Environment Setup

This guide documents how to recreate the working environment for the `video_quality` pipeline, including split outputs for standard metrics and `action_following`.

## 1) Prerequisites
- OS: Linux with CUDA 12.8 runtime (matches torch 2.10.0 build in current env).
- Python: 3.10.x (conda recommended).
- GPU with sufficient VRAM for SAM/RAFT/CLIP models.

## 2) Create conda env
```bash
conda create -y -p /path/to/envs/WorldArena python=3.10
conda activate /path/to/envs/WorldArena
```

## 3) Install Python deps
Use the curated requirements aligned to the current working env:
```bash
pip install -r video_quality/requirements.txt
```
If you prefer full reproducibility with optional tooling (Jupyter, etc.), you can `pip install jupyter notebook jupyterlab` additionally.

## 4) External assets / checkpoints
Update paths in `video_quality/config/config.yaml` to point to your local checkpoints:
- `ckpt.action_following` (CLIP ViT-B-32)
- `ckpt.background_consistency.raft`, `ckpt.dynamic_degree.raft`, `ckpt.flow_score.raft`
- `ckpt.photometric_smoothness.cfg` and `ckpt.photometric_smoothness.model`
- `ckpt.motion_smoothness.model`
- `ckpt.image_quality.musiq`
- `ckpt.aesthetic_quality.clip` / `aesthetic_head`
- `ckpt.subject_consistency.*`
- `ckpt.sam3_model_ckpt`, `ckpt.vlm_model`, etc.

## 5) Data layout
- Standard metrics use `data.val_base` and `data.gt_path`.
- `action_following` uses `data_action_following.val_base` and `data_action_following.gt_path` (must exist; no fallback).
  - Ensure you run `preprocess_datasets_diversity.py --summary_json ... --gen_video_dir ... --output_base <parent-of-generated_dataset_action_following>` so that `generated_dataset_action_following/` and `gt_dataset_action_following/` match the config paths.

## 6) Running evaluations
- Standard metrics (example):
```bash
bash video_quality/run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> "image_quality,flow_score,trajectory_accuracy"
```
- Only `action_following` (uses `data_action_following` paths):
```bash
bash video_quality/run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON>
```
Both scripts call `evaluate.py --overwrite` to regenerate outputs in `save_path` / `save_path_action_following`.

## 7) VLM Evaluation (Interaction Quality / Perspectivity / Instruction Following)
- Environment: clone from your absolute path or install fresh; a conda env name like `WorldArena_VLM` is recommended.
  - Clone example:
    ```bash
    conda create -y -p /path/to/envs/WorldArena_VLM --clone your absolute path
    conda activate /path/to/envs/WorldArena_VLM
    ```
  - If not cloning, install minimal deps:
    ```bash
    conda create -y -p /path/to/envs/WorldArena_VLM python=3.10
    conda activate /path/to/envs/WorldArena_VLM
    pip install -r video_quality/requirements_phygenbench.txt
    ```
- Run:
  ```bash
  bash video_quality/run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> [metrics] [config]
  ```
  Ensure VLM checkpoints in the config (such as `vlm_model`) point to your absolute path, and that `VIDEO_DIR` / `SUMMARY_JSON` match the evaluation inputs.

## 8) JEPA Similarity
- 环境：参考 `requirements_jedi.txt` 创建名为 `WorldArena_JEPA` 的环境（或按官方 JEDi 依赖）。示例：
  ```bash
  conda create -y -p /path/to/envs/WorldArena_JEPA python=3.10
  conda activate /path/to/envs/WorldArena_JEPA
  pip install -r requirements_jedi.txt
  ```
- Weights: see [video_quality/JEPA_readme.txt](JEPA_readme.txt) to download `vith16.pth.tar` and `ssv2-probe.pth.tar` into `video_quality/JEDi/pretrained_models/`.
- 运行：
  ```bash
  bash video_quality/run_evaluation_JEPA.sh
  ```

## 9) Notes
- If you change CUDA/toolkit versions, reinstall torch/torchvision/torchaudio/xformers with matching wheels.
- Keep `requirements.txt` in sync with the conda env; current versions reflect the live environment captured on 2026-02-14.

## WorldArena Video Quality Evaluation Environment and Usage Guide

Scope: standard metrics, VLM interaction quality/perspectivity/instruction following, JEPA similarity.
Environment split: `WorldArena` (base/action following), `WorldArena_VLM` (VLM evaluation), `WorldArena_JEPA` (JEPA similarity).

### 1. Prerequisites
- OS: Linux, CUDA 12.8 (aligned with torch cu128).
- Python: 3.10.
- GPU: sufficient for SAM/RAFT/CLIP/VLM inference.

### 2. Base Environment `WorldArena`
Used for standard metrics and action following.
```bash
cd WorldArena
conda create -y -n WorldArena python=3.10
conda activate WorldArena
pip install -r video_quality/requirements.txt
# Optional: pip install jupyter notebook jupyterlab
```

### 3. VLM Environment `WorldArena_VLM`
Used for Interaction Quality / Perspectivity / Instruction Following.
```bash
cd WorldArena
conda create -y -n WorldArena_VLM python=3.10
conda activate WorldArena_VLM
# Core GPU stack (cu128 with torch 2.9.1)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
# VLM dependencies
pip install -r video_quality/requirements_worldarena_vlm.txt
# Required libs not listed in the slim deps
pip install PyYAML==6.0.3 Pillow==10.0.0
# If you need local wheels (flash-attn etc.), uncomment and set paths in requirements_worldarena_vlm.txt
```

### 4. JEPA Environment `WorldArena_JEPA`
Used for JEPA similarity.
```bash
cd WorldArena
conda create -y -n WorldArena_JEPA python=3.10
conda activate WorldArena_JEPA
pip install -r requirements_jedi.txt
# Download weights to video_quality/JEDi/pretrained_models/
mkdir -p video_quality/JEDi/pretrained_models
cd video_quality/JEDi/pretrained_models
wget -O vith16.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -O ssv2-probe.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
```

### 5. Data and Naming Rules
- Prepare `summary.json` (example):
```json
[
  {
    "gt_path": "/path/to/gt_video/episode40.mp4",
    "image": "/path/to/gt_frames/episode40.png",
    "prompt": [
      "In a fixed robotic workspace, generate a rigid, physically consistent..."
    ]
  },
    ...
]
```
- Generated video directory must be named `modelname_sort`; put only videos inside, named `{taskname}_episode_{xx}.mp4`; no subfolders allowed.

### 6. External Weights / Paths
Configure local weights and I/O paths in video_quality/config/config.yaml (do not change model_name: test):
- `action_following`: ["your absolute path"/models/ViT-B-32.pt](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
- `semantic_alignment`:

    - `caption`: ["your absolute path"/models/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
    - `CLIP`: ["your absolute path"/models/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
- `depth_accuracy`: ["your absolute path"/depth-anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- `aesthetic_quality`:
    - `clip`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/clip_model/ViT-L-14.pt](https://huggingface.co/jinaai/clip-models/blob/main/ViT-L-14.pt)
    - `aesthetic_head`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth]("https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true")
- `background_consistency`:
    - `clip`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/clip_model/ViT-B-32.pt](https://huggingface.co/jinaai/clip-models/blob/main/ViT-B-32.pt)
    - `raft`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/raft_model/models/raft-things.pth](https://huggingface.co/RaphaelLiu/EvalCrafter-Models/tree/main/RAFT/models)
- `dynamic_degree`:
    - `raft`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/raft_model/models/raft-things.pth](https://huggingface.co/RaphaelLiu/EvalCrafter-Models/tree/main/RAFT/models)
- `flow_score`:
    - `raft`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/raft_model/models/raft-things.pth](https://huggingface.co/RaphaelLiu/EvalCrafter-Models/tree/main/RAFT/models)
- `photometric_smoothness`:
    - `cfg`: "your absolute path"/WorldArena/video_quality/WorldArena/third_party/SEA-RAFT/config/eval/spring-M.json (already included locally)
    - `model`: ["your absolute path"/WorldArena/video_quality/WorldArena/third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/blob/main/model.safetensors)
- `motion_smoothness`:
    - `model`: ["your absolute path"/WorldArena/video_quality/WorldArena/third_party/checkpoints/VFIMamba.pkl](https://huggingface.co/MCG-NJU/VFIMamba/blob/main/model.pkl)
- `image_quality`:
    - `musiq`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/blob/main/musiq_spaq_ckpt-358bb6af.pth)
- `subject_consistency`:
    - `repo`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/dino_model/facebookresearch_dino_main](https://github.com/facebookresearch/dino)
    - `weight`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/dino_model/dino_vitbase16_pretrain.pth](https://huggingface.co/Xiaomabufei/lumos/blob/main/dino_vitbase16_pretrain.pth)
    - `model`: dino_vitb16
    - `raft`: ["your absolute path"/WorldArena/video_quality/WorldArena/vbench_cache/raft_model/models/raft-things.pth](https://huggingface.co/RaphaelLiu/EvalCrafter-Models/tree/main/RAFT/models)
- `sam3_model_ckpt`: ["your absolute path"/models/sam](https://huggingface.co/facebook/sam3/tree/main) (Note: download [bpe_simple_vocab_16e6.txt.gz](https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf/blob/main/bpe_simple_vocab_16e6.txt.gz) into this sam folder.)
- `vlm_model`: ["your absolute path"/models/qwenvl3](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

### 7. Run Evaluation

For the first two evaluations, directly use the generated video directory and summary_json; JEPA requires a GT video directory (only .mp4 files following naming rules, no nesting).
- VLM metrics (interaction quality, perspectivity, instruction following) (requires `WorldArena_VLM` env):
```bash
bash video_quality/run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> 
```
- JEPA similarity (requires `WorldArena_JEPA` env):
```bash
bash video_quality/run_evaluation_JEPA.sh
```
The following metrics first run a format preprocessing step (the bash already includes it), producing the structure under data_action_following configured in config:
```
data_action_following
  ├── gt_dataset/
  │      ├── {task_name}/
  │      │   ├── episode_{x}/
  │      │   │   ├── prompt/
  │      │   │   │   ├── init_frame.png
  │      │   │   │   └── prompt.txt
  │      │   │   └── video/
  │      │   │       ├── frame_00000.jpg
  │      │   │       ├── ...
  │      │   │       └── frame_0000n.jpg
  │      │   ├── episode_{x+1}/
  │      │   └── ...
  │      ├── {task_name1}/
  │      └── ...
  ├── generated_dataset/
  │      ├── {task_name}/
  │      │   ├── episode_{x}/
  │      │   │   ├── 1/
  │      │   │   │   └── video/
  │      │   │   │       ├── frame_00000.jpg
  │      │   │   │       ├── ...
  │      │   │   │       └── frame_0000n.jpg
  │      │   │   ├── 2/
  │      │   │   └── 3/
  │      │   ├── episode_{x+1}/
  │      │   └── ...
  │      ├── {task_name1}/
  │      └── ...
  │

```
For action following, episode_{x} must contain subfolders 1, 2, 3; other metrics only need subfolder 1.

Videos in subfolders 2 and 3 can be created by modifying the original prompt to guide two different actions; you can call an LLM or write manually. Sample LLM prompt:
```txt
"""You are a robotics data augmentation expert. Your goal is to rewrite a robot instruction into two diverse variations to test policy generalization. 

STRICT RULES for Variations:
1. **Consistency Constraint (DO NOT CHANGE)**: 
   - You MUST keep the **Target Object** from the original instruction (e.g., if it's "red block", keep it "red block") to match the provided initial image.
   - You MUST keep the **Robot Part** (e.g., "arm"), but you can switch between "left" and "right". Do NOT change "arm" to "hand" or other parts.

2. **Variation 1: Spatial & Entity Mirroring (Maximum Logical Difference)**:
   - **Switch the Arm**: If the original used the "left arm", you MUST use the "right arm" (and vice versa).
   - **Reverse the Direction**: If the original moved "left", move "right". If "forward", move "backward".
   - **Flip the Destination**: Move the object to the opposite side of the workspace compared to the original goal.

3. **Variation 2: Goal Redefinition & Trajectory Re-planning**:
   - **Change the Final Goal**: Instead of the original destination, invent a completely new one (e.g., "drop it into the container", "stack it on another object", "hide it behind the bin").
   - **Exaggerate Path & Amplitude**: Use high-contrast movement styles (e.g., "via an exaggeratedly high overhead arc", "dragging it slowly across the surface", "approaching from the extreme far side").

4. **Diversity**: The resulting actions should require completely different joint configurations (Action sequences) from the original.

Example Input: "Use the left arm to pick up the banana and put it on the green plate."
Example Output: [
    "Use the right arm to grasp the banana and move it to the far right edge of the table, away from the plates.", 
    "Use the left arm to lift the banana in a wide circular sweep and carefully tuck it inside the brown box instead of the plate."
]

Output ONLY a raw JSON list containing exactly 2 strings."""
```
Use these two instructions to generate two new action videos. If the action-guided video lacks a modifiable prompt, consider using other actions from the same task to achieve different actions.

Finally, place the two new-action videos into corresponding directories (all must be MP4, no nested files, names match generated video names). Name the three directories `modelname_sort` `modelname_1_sort` `modelname_2_sort`:

- action following (requires `WorldArena` env):
```bash
bash video_quality/run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON>
# If using a specific split, run preprocess_datasets_diversity.py first, or ensure config data_action_following path exists
```
- Other metrics (requires `WorldArena` env):
```bash
bash video_quality/run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> 
```

- Metric aggregation (requires `WorldArena` env):
```bash
python video_quality/csv_results/aggregate_results.py --model_name <MODEL_NAME> --base_dir . --csv_name aggregated_results.csv
```
You can view all metric results under the csv_results directory.

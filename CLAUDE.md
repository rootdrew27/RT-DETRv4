# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RT-DETRv4 is a real-time object detection framework that uses knowledge distillation from Vision Foundation Models (DINOv3) to train lightweight detectors. It builds on RT-DETR, D-FINE, and DEIM architectures.

## Common Commands

```bash
# Training (distributed, 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0

# Single GPU training
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0

# Testing/Evaluation
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --test-only -r model.pth

# Fine-tuning from checkpoint
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0 -t model.pth

# Resume training
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp -r checkpoint.pth

# Export to ONNX
python tools/deployment/export_onnx.py --check -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth

# Inference
python tools/inference/torch_inf.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth --input image.jpg --device cuda:0

# Model info (FLOPs, params)
python tools/benchmark/get_info.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml
```

## Architecture Overview

### Model Pipeline
```
Input Image -> Backbone (HGNetv2) -> HybridEncoder -> DFINETransformer Decoder -> Detections
                                          |
                                   Distillation Branch <-- DINOv3 Teacher
```

### Engine Module Structure (`engine/`)
- **backbone/**: Backbone networks (HGNetv2, ResNet variants, CSP variants)
- **core/**: Configuration system (`YAMLConfig`) and workspace management (component registry)
- **data/**: DataLoaders, COCO dataset, augmentation transforms (Mosaic, mixup, photometric)
- **misc/**: Utilities (distributed training, logging, box operations, visualization)
- **optim/**: Optimizers, EMA, learning rate schedulers (FlatCosine), warmup
- **rtv4/**: Core RT-DETRv4 components:
  - `rtv4.py`: Main model combining backbone, encoder, decoder
  - `hybrid_encoder.py`: Feature pyramid with distillation branch
  - `dfine_decoder.py`: Deformable attention transformer decoder
  - `rtv4_criterion.py`: Multi-task loss (classification, bbox, GIoU, distillation)
  - `matcher.py`: Hungarian bipartite matching
  - `dinov3_teacher.py`: Teacher model wrapper for distillation
- **solver/**: Training loop (`DetSolver`), evaluation engine, checkpoint management

### Configuration System

Configs use YAML with inheritance via `__include__`:
```yaml
__include__: [
  '../dfine/dfine_hgnetv2_s_coco.yml',
  '../base/rtv4.yml'
]
```

Config hierarchy: `configs/rtv4/*.yml` -> `configs/dfine/*.yml` -> `configs/base/*.yml` -> `configs/dataset/*.yml`

Component instantiation uses registry pattern with `@register()` decorator. Reference components in YAML by `type` field.

### Training Pipeline Stages

1. **Heavy augmentation** (epochs 4-64): Mosaic, mixup, RandomIoUCrop
2. **Flat LR phase** (epochs 4-64): Constant learning rate
3. **Refinement** (epochs 64-120): Progressive augmentation removal
4. **Final phase** (last 12 epochs): No augmentation, cosine LR decay

### Key Configuration Points

- Dataset paths: `configs/dataset/coco_detection.yml`
- Batch size/augmentation: `configs/base/dataloader.yml`
- Model architecture: `configs/base/dfine_hgnetv2.yml`
- Distillation settings: `configs/base/rtv4.yml`
- Teacher model paths: Set `dinov3_repo_path` and `dinov3_weights_path` in model config

### Custom Dataset Training

1. Set `remap_mscoco_category: False` in dataset config
2. Update `num_classes` to match your dataset
3. Organize data in COCO format (images/, annotations/)
4. Modify paths in `configs/dataset/custom_detection.yml`

## Tools Directory

- **tools/deployment/**: ONNX and TensorRT export
- **tools/inference/**: PyTorch, ONNX, TensorRT inference scripts
- **tools/benchmark/**: Model profiling and latency measurement
- **tools/visualization/**: Fiftyone visualization
- **tools/dataset/**: Dataset preparation utilities

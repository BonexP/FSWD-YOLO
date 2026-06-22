# Saved local changes

Branch: `saved/local-changes` (commit `60302c34`)

**Files saved:**
- `scripts/train.py` — defaults: img-size=800, mixup=0.3, scale=0.6, plots=False
- `val_with_wandb.py` — standalone validation with W&B logging
- `val_continued.py` — resume validation after killed training
- `draw_gt_yolo.py` — draw YOLO ground truth boxes on images
- `fix_yolo_class_ids.py` — normalize class IDs (0.0→0, 1.0→1)
- `tile_dataset_800.py` — tile dataset into 800×800 patches
- `tile_dataset_800_debug.py` — debug dataset structure before tiling

**Recover:**
```bash
git diff next..saved/local-changes -- scripts/train.py
git cherry-pick saved/local-changes
```

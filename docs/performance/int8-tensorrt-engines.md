# INT8 TensorRT engines — export, benchmarks, and accuracy (issue #7)

> **Verified in the field:** 2026-09-01, Jetson Orin Nano 8GB Super (P3767-0005),
> JetPack 6.2.2 / L4T R36.5.2, TensorRT 10.3, ultralytics 8.3.225
> (`ultralytics/ultralytics:latest-jetson-jetpack6`), clocks locked per
> [the clocks/fan runbook](../troubleshooting/lock-clocks-fan-maxn-super.md) (MAXN_SUPER).

## TL;DR

| Model | INT8 status | Verdict |
|---|---|---|
| `yolo11n` (detect) | ✅ built, calibrated on COCO val | **Switched**: `src/detection_server.py` (with runtime FP16 fallback), `src/video_detector.py` default, and Jetson-side `detect.py`/`detectClean.py` now load `/ssd/yolo11n-int8.engine` |
| `yolo11n-seg` | ❌ build blocked | Stays FP16 — TensorRT 10.3 tactic gap (see below) |
| `racetrack_model` (seg, imgsz 416) | ❌ build blocked | Stays FP16 — same gap; **train-split** calibration cache banked for a post-upgrade retry |

INT8 on the Orin Nano (Ampere iGPU, no DLA) delivered **+27.6 % throughput /
−21.6 % mean latency** for the detection model at a **−1.5 pt mAP50-95** cost —
squarely inside the 20–30 % end-to-end gain the issue predicted.

## Numbers (batch=1, 640×640 unless noted, locked clocks)

### Engine-level — `trtexec --useSpinWait --warmUp=2000 --iterations=300`

| Engine | Precision | Mean GPU compute | Throughput | Size |
|---|---|---|---|---|
| `yolo11n` | FP16 | 4.271 ms | 233.7 qps | 8.61 MB |
| `yolo11n` | **INT8** | **3.346 ms** | **298.4 qps** | **5.59 MB (−35 %)** |
| `yolo11n-seg` | FP16 | 5.072 ms | 196.8 qps | 9.41 MB |
| `racetrack_model` (416×416) | FP16 | 2.974 ms | 335.8 qps | 8.98 MB |

### Accuracy — `yolo val` (COCO val2017 5000 imgs; racetrack valid 894 imgs, `task=segment imgsz=416`)

| Model / engine | P | mAP50 | mAP50-95 | Δ mAP50-95 |
|---|---|---|---|---|
| yolo11n FP16 | 0.653 | 0.547 | 0.391 | — |
| yolo11n INT8 | 0.645 | 0.533 | **0.376** | **−1.5 pt** |
| yolo11n-seg FP16 (box / mask) | 0.643 | 0.539 / 0.512 | 0.387 / 0.321 | baseline only |
| racetrack FP16 (box / mask) | 0.599 | 0.660 / 0.602 | 0.474 / 0.366 | baseline only |

Visual spot-check (20 COCO val images, conf > 0.35): **14/20 images produce
identical class sets**; on shared detections the mean confidence delta is
**0.086**. The 6 differing images disagree only on borderline-confidence
detections near the 0.35 cut (2 FP16-only vs 6 INT8-only class hits total) —
consistent with the −1.5 pt mAP and imperceptible in the webcam demo stream.

## The seg-model INT8 blocker (TensorRT 10.3)

Both segmentation models fail engine building with:

```
Error Code 10: Internal Error (Could not find any implementation for node
/model.23/proto/cv3/conv/Conv + PWN(PWN(/model.23/proto/cv3/act/Sigmoid), PWN(/model.23/proto/cv3/act/Mul)).)
```

What was tried (all failed with the same node):

1. Plain `int8=True` — fails on `proto/cv2/act/Mul` (never sets FP16 fallback).
2. `int8=True half=True` — same failure: ultralytics 8.3.225 `utils/export/engine.py`
   uses `elif half:` after `if int8:`, so **kFP16 is never set together with kINT8**
   (upstream bug; also bites anyone wanting mixed-precision INT8).
3. Bind-mounting a one-word patch (`elif` → `if`) over the container's `engine.py`
   — full 5000-image calibration re-used from cache, build still fails on the same
   fused Conv+PWN node → not a precision-fallback problem.
4. Lean standalone builder (imports only `tensorrt`, ~200 MB RSS) serving the
   MinMax cache verbatim, after dropping page cache (free RAM 2.0 → 6.5 GiB) —
   same node. Rules out memory pressure as the cause.
5. `trtexec --int8 --fp16 --calib=<cache>` — different failure: trtexec's
   calibrator is hardwired to ENTROPY2, rejects the MINMAX cache, then dies on
   `images bound to nullptr`.

Conclusion: a genuine TRT 10.3 tactic gap on Orin (SM8.7) for the Segment head's
proto-branch fused `Conv + PWN(SiLU)` in INT8. Matches unresolved upstream
reports (ultralytics #19974, #16415, discussion #8545 — same error on other
GPUs/TRT versions). **Retry after the next JetPack/TensorRT upgrade**; the
calibration caches are committed under `models/` (`*.cache`) and the export
scripts live on the Jetson at `/ssd/int8/` so the retry is a 2-command job.

## Gotchas found on the way (worth remembering)

- **`yolo export` overwrites the sibling `.engine`** — output mirrors the input
  stem. Always export from a precision-suffixed copy of the `.pt`
  (`/ssd/int8/yolo11n-int8.pt` → `yolo11n-int8.engine`). The first run here
  clobbered `/ssd/yolo11n.engine` (rebuilt from the same `.pt`, no data lost).
- **Ultralytics engines carry a metadata header** (4-byte LE length + JSON +
  serialized engine). `trtexec --loadEngine` aborts (`LLVM ERROR: out of
  memory`) on such files — strip the header first (see `/ssd/int8` bench logs).
- **Old engines have no metadata** → `yolo val` crashes
  (`model.metadata.get`) and task guessing fails. Engine task is guessed from
  the *filename* first: name a racetrack engine without `-seg` and you must
  pass `task=segment` yourself (as `src/formula_1_segmentation.py` already does
  for the `.pt`).
- **racetrack_model was trained at imgsz 416** — val/inference needs
  `imgsz=416`, not the 640 default.
- **INT8 calibration uses the `val` split** of `data=` (ultralytics
  `get_int8_calibration_dataloader`), batch 1, MinMax algorithm; the cache is
  `<stem>.cache` next to the ONNX and is re-used verbatim on rebuilds. The
  racetrack cache was produced against the **train split** (3320 images) per
  issue #7, via `Racetrack.v1i.yolov11/data-calib-train.yaml` (`val:` pointed
  at `train/images` — ultralytics has no `split=` export arg).
- **Old `models/*.engine` copies in this repo were TRT-8-era** (pre-JetPack
  6.2.2) and could not deserialize on TensorRT 10.3 at all — they are not a
  runnable FP16 baseline, which is why this change also refreshes them with
  current builds (which additionally carry ultralytics metadata).
- COCO auto-download pulls **train2017+test2017 too (~20 GB)** even though only
  val2017 (778 MB) is needed for calibration. Kill it after `val2017.txt`
  appears, or pre-seed `/ssd/datasets/coco/`.
- On 8 GB unified memory, TRT tactic timing sees **free** (not available) RAM —
  drop page caches before big builds (`docker run --rm --privileged … echo 3 >
  /proc/sys/vm/drop_caches`) or tactics get skipped as "insufficient memory".

## Where things live

| Artifact | Jetson | Repo |
|---|---|---|
| INT8 det engine (production) | `/ssd/yolo11n-int8.engine` | `models/yolo11n-int8.engine` |
| FP16 engines (production) | `/ssd/{yolo11n,yolo11n-seg,racetrack_model}.engine` | `models/*.engine` (refreshed, TRT 10.3, with metadata) |
| Seg INT8 calibration caches (banked for retry) | `/ssd/int8/*.cache` | `models/*-int8.cache` |
| Export/bench/val scripts + all logs | `/ssd/int8/` | — |
| COCO val (5000) for future calibration/val | `/ssd/datasets/coco/` | — |

The seg INT8 retry after a TRT upgrade:

```bash
# with the elif→if patch mounted (or a fixed ultralytics), calibration is cached:
docker run --rm --runtime nvidia -v /ssd:/ssd -v /ssd/datasets:/datasets \
  -v /ssd/int8/engine_patched.py:/ultralytics/ultralytics/utils/export/engine.py:ro \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  yolo export model=/ssd/int8/yolo11n-seg-int8.pt format=engine int8=True half=True \
  data=coco.yaml device=0

# racetrack (train-split calibration cache already on disk):
docker run --rm --runtime nvidia -v /ssd:/ssd -v /ssd/datasets:/datasets \
  -v /ssd/int8/engine_patched.py:/ultralytics/ultralytics/utils/export/engine.py:ro \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  yolo export model=/ssd/int8/racetrack_model-int8.pt format=engine int8=True half=True \
  data=/ssd/Racetrack.v1i.yolov11/data-calib-train.yaml device=0
```

## Reproducing the benchmarks

On the Jetson (`/ssd/int8/`): `bench_all.sh` (trtexec), `val_all.sh` (mAP),
`summarize.sh` (tables). Raw logs in `/ssd/int8/logs/`
(`bench-*.log`, `val-*.log`, `progress*.log`).

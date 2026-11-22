"""Run RTMW-m on one image and save its 133 COCO-WholeBody keypoints."""
import argparse
import json
from pathlib import Path

from python_paths import configure_python_paths

configure_python_paths()

import numpy as np
from PIL import Image
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules
except ModuleNotFoundError as exc:
    if exc.name and exc.name.split(".")[0] == "mmpose":
        import sys
        try:
            import torch
            torch_info = f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
        except ImportError:
            torch_info = "PyTorch is not installed"
        raise SystemExit(
            f"MMPose не установлен в интерпретаторе {sys.executable} ({torch_info}). "
        ) from exc
    raise
from image_utils import make_cv_readable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"venv", "scripts"} else SCRIPT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "models/openmmlab/rtmw/rtmw-m_8xb1024-270e_cocktail14-256x192.py")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/openmmlab/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth")
    p.add_argument("--output", type=Path, default=Path("results/rtmw_keypoints.json"))
    p.add_argument("--device", default=None, help="cuda:0 or cpu; auto-select if omitted")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Optional person box in image pixels; default uses the full image")
    args = p.parse_args()

    original_image = args.image.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not original_image.is_file() or not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Image, RTMW config, or RTMW checkpoint does not exist")
    image = make_cv_readable(original_image, args.output.resolve().parent / "normalized")
    width, height = Image.open(image).size
    if args.bbox:
        bbox = np.asarray([args.bbox], dtype=np.float32)
    else:
        bbox = np.asarray([[0, 0, width - 1, height - 1]], dtype=np.float32)

    register_all_modules()
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    # PyTorch 2.6+ defaults to weights_only=True. The official RTMW
    # checkpoint stores NumPy dtype metadata; allow only NumPy reconstruction
    # and dtype classes while keeping the restricted checkpoint loader on.
    import torch
    safe_numpy_globals = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
    ]
    safe_numpy_globals.extend(
        value for value in vars(np.dtypes).values() if isinstance(value, type))
    torch.serialization.add_safe_globals(safe_numpy_globals)

    model = init_model(str(config), str(checkpoint), device=device)
    samples = inference_topdown(model, str(image), bboxes=bbox, bbox_format="xyxy")
    if not samples:
        raise RuntimeError("RTMW returned no pose instances")
    pred = samples[0].pred_instances
    points = np.asarray(pred.keypoints)[0]
    scores = np.asarray(pred.keypoint_scores)[0]
    if points.shape != (133, 2) or scores.shape != (133,):
        raise RuntimeError(f"Expected 133 COCO-WholeBody points, got {points.shape} / {scores.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image": str(image), "original_image": str(original_image),
        "image_width": width, "image_height": height,
        "keypoint_format": "coco_wholebody_133",
        "keypoints": [[float(x), float(y), float(s)]
                      for (x, y), s in zip(points, scores)],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(points)} RTMW points: {args.output.resolve()}")


if __name__ == "__main__":
    main()

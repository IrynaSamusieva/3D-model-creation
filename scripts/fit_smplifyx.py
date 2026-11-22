"""Prepare one-image SMPLify-X input and launch the official fitter."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from python_paths import configure_python_paths

configure_python_paths()

from image_utils import make_cv_readable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"venv", "scripts"} else SCRIPT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=ROOT / "image.avif")
    p.add_argument("--keypoints", required=True, type=Path, help="OpenPose JSON from rtmw_to_openpose.py")
    p.add_argument("--repo", required=True, type=Path, help="Official SMPLify-X repository root")
    p.add_argument("--model-folder", type=Path, default=ROOT / "models_smplx_v1_1" / "models", help="SMPL-X model folder")
    p.add_argument("--vposer", type=Path, default=ROOT / "vposer_v1_0" / "vposer_v1_0", help="VPoser checkpoint folder")
    p.add_argument("--output", type=Path, default=Path("results/smplifyx"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    keys, repo = args.keypoints.resolve(), args.repo.resolve()
    output = args.output.resolve()
    image = make_cv_readable(args.image, output / "normalized")
    model_folder, vposer = args.model_folder.resolve(), args.vposer.resolve()
    main_py = repo / "smplifyx" / "main.py"
    config = repo / "cfg_files" / "fit_smplx.yaml"
    for path in (image, keys, main_py, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not model_folder.is_dir() or not vposer.is_dir():
        raise NotADirectoryError("SMPL-X model folder and VPoser folder must both exist")

    input_root = output / "input"
    (input_root / "images").mkdir(parents=True, exist_ok=True)
    (input_root / "keypoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, input_root / "images" / image.name)
    # SMPLify-X's OpenPose parser appends ``_keypoints.json`` to each image stem.
    shutil.copy2(keys, input_root / "keypoints" / f"{image.stem}_keypoints.json")
    fit_dir = output / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.python, str(main_py), "--config", str(config),
           "--data_folder", str(input_root), "--output_folder", str(fit_dir),
           "--model_folder", str(model_folder), "--vposer_ckpt", str(vposer),
           "--model_type", "smplx", "--use_vposer", "True",
           "--use_hands", "True", "--use_face", "True",
           "--use_joints_conf", "True", "--use_cuda", "True" if args.device == "cuda" else "False",
           # The optional torch-mesh-isect package is not installed; the
           # collision penalty is not required to produce the SMPL-X fit.
           "--interpenetration", "False",
           "--interactive", "False",
           "--visualize", "True" if args.visualize else "False",
           "--save_meshes", "True", "--max_persons", "1"]
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    print("Launching SMPLify-X:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo, check=True)
    print(f"SMPL-X mesh and fit parameters are in {fit_dir}")


if __name__ == "__main__":
    main()

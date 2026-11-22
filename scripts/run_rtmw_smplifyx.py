"""Run RTMW -> SMPLify-X and export an SMPL-X mesh."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name.lower() in {"scripts", "venv"} else APP_DIR
SCRIPT_DIR = APP_DIR
sys.path.insert(0, str(SCRIPT_DIR))
from python_paths import child_environment


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True,
                   env=child_environment())


def check_environment(label, interpreter, packages, require_cuda):
    package_list = ",".join(packages)
    code = (
        "import importlib.util,sys; "
        f"names={package_list!r}.split(','); "
        "missing=[name for name in names if importlib.util.find_spec(name) is None]; "
        "print('Python:',sys.executable); "
        "print('Missing packages:',', '.join(missing) if missing else 'none'); "
        "import torch; "
        "print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available(),'built with:',torch.version.cuda); "
        + ("sys.exit(3) if missing or not torch.cuda.is_available() else None" if require_cuda
           else "sys.exit(3) if missing else None")
    )
    result = subprocess.run([str(interpreter), "-c", code], cwd=ROOT,
                            text=True, capture_output=True, env=child_environment())
    print(f"\n[{label} environment: {interpreter}]")
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode:
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(
            f"{label} environment is missing required packages or CUDA PyTorch. "
            "Install them into the project-local environment described in README.md, "
            "or pass the correct Python executable with the matching --*-python option."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=ROOT / "image.avif")
    p.add_argument("--smplifyx-repo", required=True, type=Path)
    p.add_argument("--smplx-models", type=Path, default=ROOT / "models_smplx_v1_1" / "models")
    p.add_argument("--vposer", type=Path, default=ROOT / "vposer_v1_0" / "vposer_v1_0")
    p.add_argument("--output", type=Path, default=Path("results/pipeline"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--mmpose-python", type=Path,
                   default=Path(r"D:\Python314\python.exe"),
                   help="Python executable on D: with MMPose and CUDA PyTorch")
    p.add_argument("--smplifyx-python", type=Path,
                   default=Path(r"D:\Python314\python.exe"),
                   help="Python executable on D: with SMPLify-X and CUDA PyTorch")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Person box in pixel coordinates; defaults to full photo")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()
    args.mmpose_python = args.mmpose_python.resolve()
    args.smplifyx_python = args.smplifyx_python.resolve()
    for label, interpreter in (("MMPose", args.mmpose_python), ("SMPLify-X", args.smplifyx_python)):
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"{label} Python environment was not found: {interpreter}\n"
                "Check the relocated Python path on D: (see README.md), or pass "
                f"--{'mmpose' if label == 'MMPose' else 'smplifyx'}-python with its python.exe path."
            )
    check_environment("MMPose", args.mmpose_python,
                      ("torch", "mmcv", "mmengine", "mmpose"), args.device == "cuda")
    check_environment("SMPLify-X", args.smplifyx_python,
                      ("torch", "smplx", "human_body_prior"), args.device == "cuda")
    image = args.image.resolve()
    out = args.output.resolve()
    raw = out / "rtmw" / f"{image.stem}_rtmw.json"
    openpose = out / "smplifyx_input" / "keypoints" / f"{image.stem}.json"
    fit_dir = out / "smplifyx" / "fit"
    avatar_dir = out / "avatar"

    infer = [args.mmpose_python, SCRIPT_DIR / "rtmw_infer.py", "--image", image,
             "--output", raw, "--device", args.device]
    if args.bbox:
        infer += ["--bbox", *args.bbox]
    run(infer)

    # RTMW converts AVIF to PNG and records the readable image path in its JSON.
    with raw.open("r", encoding="utf-8") as f:
        inference_result = json.load(f)
    fitting_image = Path(inference_result["image"])

    run([args.mmpose_python, SCRIPT_DIR / "rtmw_to_openpose.py", "--input", raw, "--output", openpose])
    run([args.smplifyx_python, SCRIPT_DIR / "fit_smplifyx.py", "--image", fitting_image,
         "--keypoints", openpose, "--repo", args.smplifyx_repo.resolve(),
         "--model-folder", args.smplx_models.resolve(), "--vposer", args.vposer.resolve(),
         "--output", out / "smplifyx", "--python", args.smplifyx_python,
         "--device", args.device, *( ["--visualize"] if args.visualize else [] )])
    run([args.smplifyx_python, SCRIPT_DIR / "export_smplx.py", "--fit-dir", fit_dir,
         "--image-name", image.name, "--output", avatar_dir])
    print(f"\nPipeline complete. Final SMPL-X avatar: {avatar_dir}")


if __name__ == "__main__":
    main()

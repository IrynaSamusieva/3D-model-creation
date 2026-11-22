"""Collect the SMPL-X mesh and fitted parameter file into a delivery folder."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-dir", required=True, type=Path, help="SMPLify-X output folder (the fit/ folder)")
    p.add_argument("--image-name", required=True, help="Input image filename including extension")
    p.add_argument("--output", type=Path, default=Path("results/avatar"))
    args = p.parse_args()
    stem = Path(args.image_name).stem
    fit = args.fit_dir.resolve()
    mesh = fit / "meshes" / stem / "000.obj"
    params = fit / "results" / stem / "000.pkl"
    if not mesh.is_file() or not params.is_file():
        raise FileNotFoundError(f"Expected SMPLify-X outputs: {mesh} and {params}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out_mesh = out / f"{stem}_smplx.obj"
    out_params = out / f"{stem}_smplx_parameters.pkl"
    shutil.copy2(mesh, out_mesh)
    shutil.copy2(params, out_params)
    manifest = {
        "model": "SMPL-X",
        "mesh_obj": out_mesh.name,
        "fitted_parameters_pickle": out_params.name,
        "source": "Official SMPLify-X fitting output",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Avatar mesh: {out_mesh}\nFit parameters: {out_params}")


if __name__ == "__main__":
    main()

"""Small image helpers shared by the inference and fitting scripts."""
from pathlib import Path


def make_cv_readable(image_path: Path, output_dir: Path) -> Path:
    """Convert AVIF to PNG; return other supported image files unchanged."""
    image_path = image_path.expanduser().resolve()
    if image_path.suffix.lower() not in {".avif", ".avifs"}:
        return image_path
    try:
        import pillow_avif  # noqa: F401  registers the AVIF decoder in Pillow
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "AVIF support is missing. Install pillow-avif-plugin in this Python environment: "
            "python -m pip install pillow-avif-plugin"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{image_path.stem}.png"
    with Image.open(image_path) as image:
        image.convert("RGB").save(png_path, format="PNG")
    return png_path.resolve()

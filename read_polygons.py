import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from reading_order import GraphBasedOrdering


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract text line polygons from htr3.polygons.json file and save each as an image."
    )
    parser.add_argument(
        "--polygons_file",
        type=str,
        required=True,
        help="Path to the htr3.polygons.json file.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the source image file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./line_crops",
        help="Directory where cropped line images will be saved.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional base directory to prepend to relative image paths.",
    )
    parser.add_argument(
        "--polygon_buffer",
        type=float,
        default=0.0,
        help="Buffer (in pixels) applied to each polygon before cropping.",
    )
    return parser.parse_args()


def polygon_to_tuples(polygon: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(point[0]), int(point[1])) for point in polygon]


def _buffer_polygon(
    polygon: Sequence[Sequence[int]],
    buffer_amount: float,
) -> List[Tuple[int, int]]:
    if buffer_amount == 0:
        return polygon_to_tuples(polygon)

    poly = Polygon(polygon)
    if not poly.is_valid:
        poly = poly.buffer(0)

    buffered = poly.buffer(buffer_amount, join_style=2)
    if buffered.is_empty:
        return polygon_to_tuples(polygon)

    if buffered.geom_type == "MultiPolygon":
        buffered = max(buffered.geoms, key=lambda g: g.area)

    return [(int(round(x)), int(round(y))) for x, y in buffered.exterior.coords[:-1]]


def crop_line(
    image: Image.Image,
    polygon: Sequence[Sequence[int]],
    buffer_amount: float = 0,
) -> Optional[Image.Image]:
    polygon_points = _buffer_polygon(polygon, buffer_amount)
    if len(polygon_points) < 3:
        return None

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon_points, fill=255)
    bbox = mask.getbbox()
    if not bbox:
        return None

    rgba = image.convert("RGBA")
    masked = Image.new("RGBA", image.size)
    masked.paste(rgba, mask=mask)
    return masked.crop(bbox)


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_htr3_polygons(path: Path):
    """Load polygons from htr3.polygons.json format."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("line_polygons", [])


def _compute_bbox_from_polygon(polygon: Sequence[Sequence[int]]) -> List[float]:
    """Compute bounding box from polygon coordinates."""
    if not polygon:
        return [0, 0, 0, 0]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _order_polygons(polygons: List[Sequence[Sequence[int]]]) -> List[Sequence[Sequence[int]]]:
    """Order polygons using reading order based on computed bounding boxes."""
    if not polygons:
        return polygons

    boxes = []
    for polygon in polygons:
        bbox = _compute_bbox_from_polygon(polygon)
        boxes.append(bbox)

    orderer = GraphBasedOrdering()
    indices = orderer.order(boxes)
    if not indices:
        return polygons
    return [polygons[i] for i in indices]


def _resolve_image_path(image_path: str, image_root: Optional[Path]) -> Path:
    path = Path(image_path).expanduser()
    if path.is_absolute() or not image_root:
        return path
    return (image_root / path).resolve()


def process_polygons(
    polygons: List[Sequence[Sequence[int]]],
    image_path: Path,
    output_root: Path,
    polygon_buffer: float,
) -> int:
    """Process polygons and save cropped line images."""
    if not polygons:
        return 0

    ordered_polygons = _order_polygons(polygons)
    print("ordered_polygons")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return 0

    
    image_name = Path(image_path).stem
    target_dir = output_root
    #ensure_directory(target_dir)

    saved = 0
    filenames = []
    for idx, polygon in enumerate(ordered_polygons):
        if not polygon:
            continue
        cropped = crop_line(image, polygon, polygon_buffer)
        if cropped is None:
            continue
        filename = target_dir / f"{image_name}_line_{idx:03d}.png"
        cropped.save(filename)
        saved += 1
        filenames.append(filename)
    return filenames


def main():
    args = parse_args()
    polygons_path = Path(args.polygons_file)
    image_path = _resolve_image_path(args.image_path, Path(args.image_root).resolve() if args.image_root else None)
    output_root = Path(args.output_dir)

    ensure_directory(output_root)

    polygons = load_htr3_polygons(polygons_path)

    saved = process_polygons(
        polygons,
        image_path,
        output_root,
        args.polygon_buffer,
    )

    image_name = Path(image_path).name
    print(f"{image_name}: saved {saved} line images")
    print(f"Finished. Total line images saved: {saved}")


if __name__ == "__main__":
    main()


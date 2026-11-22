"""Convert RTMW COCO-WholeBody-133 JSON to OpenPose JSON for SMPLify-X."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="JSON from rtmw_infer.py")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    src = json.loads(args.input.read_text(encoding="utf-8"))
    kp = src["keypoints"]
    if src.get("keypoint_format") != "coco_wholebody_133" or len(kp) != 133:
        raise ValueError("Input must contain 133 COCO-WholeBody keypoints")

    zero = [0.0, 0.0, 0.0]
    def point(index):
        x, y, score = kp[index]
        return [float(x), float(y), float(score)]
    def midpoint(a, b):
        pa, pb = point(a), point(b)
        if pa[2] <= 0 or pb[2] <= 0:
            return zero.copy()
        return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, min(pa[2], pb[2])]

    # COCO WholeBody indices: body 0..16, feet 17..22, face 23..90,
    # left hand 91..111, right hand 112..132. OpenPose BODY_25 order.
    body25 = [
        point(0), midpoint(5, 6), point(6), point(8), point(10),
        point(5), point(7), point(9), midpoint(11, 12),
        point(12), point(14), point(16), point(11), point(13), point(15),
        point(2), point(1), point(4), point(3),
        point(17), point(18), point(19), point(20), point(21), point(22),
    ]
    # RTMW gives 68 face landmarks; OpenPose JSON allocates 70 slots.
    face = [point(i) for i in range(23, 91)] + [zero.copy(), zero.copy()]
    left_hand = [point(i) for i in range(91, 112)]
    right_hand = [point(i) for i in range(112, 133)]
    flatten = lambda points: [v for pnt in points for v in pnt]
    out = {
        "version": 1.3,
        "image_width": int(src["image_width"]),
        "image_height": int(src["image_height"]),
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": flatten(body25),
            "face_keypoints_2d": flatten(face),
            "hand_left_keypoints_2d": flatten(left_hand),
            "hand_right_keypoints_2d": flatten(right_hand),
            "pose_keypoints_3d": [], "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": [],
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote OpenPose-compatible JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()

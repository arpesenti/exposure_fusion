import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exposure fusion of multiple images",
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE",
                        help="Input image files (at least 2)")
    parser.add_argument("-o", "--output", default="fusion.jpg",
                        help="Output image path (default: fusion.jpg)")
    parser.add_argument("-d", "--depth", type=int, default=3,
                        help="Pyramid depth (default: 3)")
    parser.add_argument("--time-decay", type=float, default=None,
                        help="Time decay factor for sequential images")
    parser.add_argument("--align", action="store_true",
                        help="Align images before fusion")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress messages")

    args = parser.parse_args()

    if len(args.images) < 2:
        parser.error("At least two input images are required")

    if args.depth < 1:
        parser.error("Depth must be at least 1")

    try:
        from PIL import Image
    except ImportError:
        print("error: Pillow is required. Install with: pip install exposure_fusion",
              file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading {len(args.images)} images...", file=sys.stderr)

    images = []
    for path in args.images:
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[:, :, ::-1]
        images.append(img)

    if args.align:
        if args.verbose:
            print("Aligning images...", file=sys.stderr)
        from exposure_fusion import align_images
        images = align_images(images)

    if args.verbose:
        print("Fusing images...", file=sys.stderr)

    from exposure_fusion import exposure_fusion
    result = exposure_fusion(images, depth=args.depth, time_decay=args.time_decay)

    if args.verbose:
        print(f"Writing {args.output}...", file=sys.stderr)
    out = result
    if out.ndim == 3:
        out = out[:, :, ::-1]
    Image.fromarray(out).save(args.output)

    print(args.output)

import argparse
import sys

from exposure_fusion import load_image, save_image, align_images, exposure_fusion


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

    if args.verbose:
        print(f"Reading {len(args.images)} images...", file=sys.stderr)

    images = [load_image(p) for p in args.images]

    if args.align:
        if args.verbose:
            print("Aligning images...", file=sys.stderr)
        images = align_images(images)

    if args.verbose:
        print("Fusing images...", file=sys.stderr)

    result = exposure_fusion(images, depth=args.depth, time_decay=args.time_decay)

    if args.verbose:
        print(f"Writing {args.output}...", file=sys.stderr)

    save_image(args.output, result)

    print(args.output)

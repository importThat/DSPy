import dsproc
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input picture file")
    parser.add_argument("-o", "--output", type=str, required=True, help ="Name of the output file")
    parser.add_argument("-fs", "--fs", type=int, required=True, help="Sampling frequency")
    parser.add_argument("-ld", "--line_duration", type=float, default=0.008,
                        help="How long each row of pixels is xmitted for. If the image looks weird "
                             "in the spectrum try changing this")
    return parser.parse_args()


def main():
    args = parse_args()
    s = dsproc.SpectrumPainter(args.fs, args.line_duration)

    samps = s.convert_image(filename=args.input)
    samps = samps.astype(np.complex64)

    print(f"Wave created and saved to {args.output}_{args.fs}")
    samps.tofile(args.output)
    print("Thankyou for choosing us!")


if __name__ == "__main__":
    main()






# +---------------------------------------------+
# - Project:    CPSC 483 Final Exam             -
# - File:       cifar.py                        -
# - Author:     Zachary Worcester               -
# - Email:      zworcester0@csu.fullerton.edu   -
# +---------------------------------------------+
# |             Project Description             |
# | An implementation of a CNN that classifies  |
# | images based on the CIFAR-10 Dataset.       |
# |||||||||||||||||||||||||||||||||||||||||||||||

"""A program that can train and predict a CIFAR-10 image classifier."""

import argparse
from final.cifarcnn import CIFARCNN

def main():
    """defines a main entry point"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e',
        '--epochs',
        default=20,
        type=int,
        help='if training, number of epochs to train for'
        )
    parser.add_argument(
        '-m',
        '--model',
        default='training/cp.ckpt',
        help='model to use for generation'
        )
    parser.add_argument('-i',
                        '--image',
                        default=None,
                        help='image to predict'
                        )

    args = parser.parse_args()

    model = CIFARCNN(args.epochs, args.model)

    if args.image is not None:
        print(f'The image {args.image} is a : {model.make_prediction(args.image)}')

if __name__ == "__main__":
    main()

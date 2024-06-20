from pretrain import pretrain
from pretest import pretest
from posttrain import posttrain
from posttest import posttest
import argparse

parser = argparse.ArgumentParser(description='Denoise simplified')
parser.add_argument('--repetition', '-r', type=int)
args = parser.parse_args()

for ref_angle in [35, 55, 125, 145]:
        for sf in [40]:
            for target_sep in [1]:
                    # 5min
                    pretrain(ref_angle, sf, target_sep, learning_rate=1e-5, epochs=5000, repetition=args.repetition)
                    # 50min
                    posttrain(ref_angle, sf, target_sep, learning_rate=1e-5, epochs=500, repetition=args.repetition)
                    # 45min
                    pretest(ref_angle, sf, target_sep, runs=1000, repetition=args.repetition)
                    # 45min
                    posttest(ref_angle, sf, target_sep, runs=1000, repetition=args.repetition, prefix="post")
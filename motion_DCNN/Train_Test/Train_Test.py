# from pretest import pretest
# from posttrain import posttrain
# from posttest import posttest
from pretrain import pretrain
import os

for repetition in range(10):

    for ref_direction in [45, 135, -45, -135]:

        for target_sep in [4]:

            recorded_epoch_pre = [0, 9, 19, 49, 99, 199, 499, 999]
            pretrain(ref_direction=ref_direction, target_sep=target_sep,
                     lr=1e-7, epochs=1000, repetition=repetition, recorded_epoch=recorded_epoch_pre)

            for nepoch in [None]:
                pretest(ref_direction=ref_direction, target_sep=target_sep,
                        runs=1000, nepoch=nepoch, repetition=repetition,
                        coherence_levels=[0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1])

            recorded_epoch_post = [0, 9, 19, 49, 99, 199, 499, 999, 1999]
            posttrain(ref_direction=ref_direction, target_sep=target_sep,
                      lr=1e-7, epochs=2000, repetition=repetition, recorded_epoch=recorded_epoch_post)

            for nepoch in [None]:
                posttest(ref_direction=ref_direction, target_sep=target_sep,
                         runs=1000, nepoch=nepoch, repetition=repetition,
                         coherence_levels=[0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1])

            for nepoch in [1, 10, 20, 50, 100, 200, 500, 1000, 2000]:
                posttest(ref_direction=ref_direction, target_sep=target_sep,
                         runs=1000, nepoch=nepoch, repetition=repetition,
                         coherence_levels=[0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1])

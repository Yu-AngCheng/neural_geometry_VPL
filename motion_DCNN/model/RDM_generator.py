import numpy as np
from copy import deepcopy
import cv2 as cv
import torch


def RDM_generator(direction, coherence, groups=1, fieldsize=(112, 112),
                  nDot=125, dotSize=3, speed=7.5, frames=16):

    height, width = fieldsize
    assert height == width
    assert isinstance(groups, int)
    assert isinstance(height, int)
    assert isinstance(width, int)
    assert isinstance(nDot, int)
    assert isinstance(dotSize, int)
    assert isinstance(frames, int)
    assert height % 2 == 0
    assert width % 2 == 0

    circle_mask = np.zeros(fieldsize, dtype=np.uint8)
    cv.circle(circle_mask, (height // 2, width // 2),
              height // 2, (255, 255, 255), thickness=-1)
    # cv.imshow('temp.png', circle_mask)
    # cv.waitKey()
    # cv.destroyAllWindows()

    new_fieldsize = (height + 2, width + 2)
    if frames % groups != 0:
        new_frames = groups - frames % groups + frames
    else:
        new_frames = frames
    assert new_frames % groups == 0

    N_coh_dot = np.floor(coherence * nDot).astype(int).item()
    N_random_dot = nDot - N_coh_dot
    raw_pic = np.zeros(new_fieldsize, dtype=np.uint8)
    pic_list = [list() for _ in range(groups)]

    for i_group in range(groups):
        dot_position = np.random.rand(nDot, 2) * np.array(new_fieldsize)
        for i_frame in range(new_frames // groups):

            temp_index = np.random.permutation(nDot)
            coh_dot_index = temp_index[:N_coh_dot]

            dot_position[coh_dot_index, 0] += speed * np.cos(np.deg2rad(direction + 90))
            dot_position[coh_dot_index, 1] += speed * np.sin(np.deg2rad(direction + 90))

            if N_random_dot > 0:
                random_dot_index = temp_index[-N_random_dot:]
                dot_position[random_dot_index, :] = np.random.rand(N_random_dot, 2) * np.array(new_fieldsize)

            dot_position[:, 0] %= new_fieldsize[0]
            dot_position[:, 1] %= new_fieldsize[1]

            dot_position_index = np.floor(deepcopy(dot_position)).astype(int)
            pic_temp = deepcopy(raw_pic)
            pic_temp[tuple(dot_position_index.T)] = 255
            se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dotSize, dotSize))
            pic_temp = cv.dilate(pic_temp, se)

            pic_temp = pic_temp[1:-1, 1:-1]
            pic_temp = cv.bitwise_and(pic_temp, pic_temp, mask=circle_mask)
            pic_temp[pic_temp == 0] = 0

            # cv.imshow('temp.png',pic_temp)
            # cv2.waitKey()
            # cv.destroyAllWindows()

            pic_list[i_group].append(deepcopy(pic_temp))

    RDM = [val for tup in zip(*pic_list) for val in tup]
    start = np.random.randint(0, new_frames - frames + 1)
    RDM = np.array(RDM[start:start + frames], dtype=np.float32)
    assert RDM.shape[0] == frames
    RDM = np.tile(RDM, (3, 1, 1, 1))

    # for t in range(frames):
    #     cv.imwrite("frame_"+str(t)+"_.jpg", RDM.transpose((2, 3, 0, 1))[:, :, :, t].astype(np.uint8))

    # out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (width, height), True)
    # for t in range(frames):
    #     out.write(RDM.transpose((2, 3, 0, 1))[:, :, :, t].astype(np.uint8))
    # out.release()

    # se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dotSize, dotSize))
    # mean_analytic = np.average([255, 0],
    #                            weights=[nDot * se.sum() * np.pi / 4, height * width - nDot * se.sum() * np.pi / 4])
    # RDM = RDM - mean_analytic

    RDM[0] = RDM[0] - 104
    RDM[1] = RDM[1] - 117
    RDM[2] = RDM[2] - 128
    return RDM


def Sample_generator(ref_direction, sep, coherence, **kwargs):
    X1 = RDM_generator(ref_direction + sep, coherence, **kwargs)
    X2 = RDM_generator(ref_direction - sep, coherence, **kwargs)
    X_All = np.concatenate((X1[None, ...], X2[None, ...]), axis=0)
    X_All = torch.from_numpy(X_All)
    Y1 = RDM_generator(ref_direction, coherence, **kwargs)
    Y2 = RDM_generator(ref_direction, coherence, **kwargs)
    Y_All = np.concatenate((Y1[None, ...], Y2[None, ...]), axis=0)
    Y_All = torch.from_numpy(Y_All)
    answer = torch.from_numpy(np.array([0, 1], dtype=np.int64))
    return {"target": X_All, "reference": Y_All, "label": answer}


if __name__ == "__main__":
    X = Sample_generator(30, 10, 0.5, groups=1)
    Y = Sample_generator(-30, 10, 0.5, groups=1)

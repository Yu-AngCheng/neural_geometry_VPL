import numpy as np
import cv2
import torch

def gabor_patch(size, theta, sf, phase, contrast, sigma):
    """

    Create a Gabor Patch
    size : int
        Image size n x n (pixel)
    sf : double
        Spatial Wavelength (pixel per cycle)
    theta : double
        Grating orientation starting from horizon (deg)
    sigma : double
        Standard deviation of gaussian envelope (pixel)
    phase : double
        0 to 2 pi
    """

    assert isinstance(size, int)
    assert sf > 0
    assert 0 <= contrast <= 1
    assert sigma > 0

    # Make 2D map
    tmp = np.linspace(1, size, size)
    Xm, Ym = np.meshgrid(tmp, tmp)
    X0, Y0 = size / 2, size / 2

    # 2D Gaussian distribution
    gauss = np.exp(-((Xm - X0) ** 2 + (Ym - Y0) ** 2) / (2 * sigma ** 2))

    # Change orientation by adding Xm and Ym together in different proportions
    thetaRad = (theta / 360) * 2 * np.pi
    Xt = Xm * np.sin(thetaRad)
    Yt = Ym * np.cos(thetaRad)

    # Combine all the elements
    gabor = (np.sin(((Xt + Yt) * 2 * np.pi / sf) + phase) * contrast * gauss + 1) / 2

    return gabor


def noise_gabor_patch(size, p, theta, sf, phase, contrast, sigma, SD_1, SD_2):
    """

        Create a Noisy Gabor Patch

        size : int
            Image size n x n (pixel)
        sf : double
            Spatial Wavelength (pixel per cycle)
        theta : double
            Grating orientation starting from horizon (deg)
        sigma : double
            Standard deviation of gaussian envelope (pixel)
        phase : double
            0 to 2 pi
        scale : int
            if scale == 1, image is 0-1
            if scale == 255 image is 0-255
        SD_1

        """

    assert isinstance(size, int)
    assert sf > 0
    assert 0 <= contrast <= 1
    assert sigma > 0

    # Create a Gabor patch
    gabor = gabor_patch(size, theta, sf, phase, contrast, sigma)

    # Add noise
    if p is not None:
        tempsize = (size // 8 + 1, size // 8 + 1)

        # Gaussian additive noise
        additive_noise = (np.random.randn(*tempsize) * SD_1).repeat(repeats=8, axis=0).repeat(repeats=8, axis=1)
        additive_noise = additive_noise[0:size, 0:size]

        # Create substitute noise
        pure_noise = (0.5 + np.random.randn(*tempsize) * SD_2).repeat(repeats=8, axis=0).repeat(repeats=8, axis=1)
        pure_noise = pure_noise[0:size, 0:size]

        # Mix SN of Gabor patch and 1-SN with pure_noise
        idx_noise = np.random.rand(*tempsize).repeat(repeats=8, axis=0).repeat(repeats=8, axis=1)
        idx_noise = idx_noise[0:size, 0:size] < p/100
        noise_gabor = idx_noise * pure_noise + np.logical_not(idx_noise) * (gabor + additive_noise)
    else:
        noise_gabor = gabor

    return noise_gabor


def training_sample_generator(p, theta, contrast,
                     dtheta=1, size=227, sf=40, sigma=50, scale=255, SD_1=10/255, SD_2=15/100):

    assert isinstance(size, int)
    assert sf > 0
    assert 0 <= contrast <= 1
    assert sigma > 0

    batchsize = 2
    in_channels = 3

    X_batch = np.zeros((batchsize, in_channels, size, size), dtype=np.float32)
    Y_batch = np.zeros((batchsize, in_channels, size, size), dtype=np.float32)
    label_batch = np.zeros(batchsize, dtype=np.float32)

    for i_batchsize in range(batchsize):

        if i_batchsize == 0:
            X = noise_gabor_patch(size, p, theta+dtheta, sf, 0, contrast, sigma, SD_1=SD_1, SD_2=SD_2)
            Y = noise_gabor_patch(size, p, theta, sf, 0, contrast, sigma, SD_1=SD_1, SD_2=SD_2)
            label = 0
        else:
            X = noise_gabor_patch(size, p, theta-dtheta, sf, 0, contrast, sigma, SD_1=SD_1, SD_2=SD_2)
            Y = noise_gabor_patch(size, p, theta, sf, 0, contrast, sigma, SD_1=SD_1, SD_2=SD_2)
            label = 1

        if scale == 255:
            X = np.tile(X * 255, (in_channels, 1, 1))
            Y = np.tile(Y * 255, (in_channels, 1, 1))

        X_batch[i_batchsize] = X
        Y_batch[i_batchsize] = Y
        label_batch[i_batchsize] = label

    X_batch = torch.from_numpy(X_batch)
    Y_batch = torch.from_numpy(Y_batch)
    label_batch = torch.from_numpy(label_batch)
    return X_batch, Y_batch, label_batch

def test_sample_generator(batchsize, p, theta, contrast, flag,
                          dtheta=1, size=227, sf=40, sigma=50, scale=255, SD_1=10/255, SD_2=15/100):

    assert isinstance(batchsize, int)
    assert isinstance(size, int)
    assert sf > 0
    assert 0 <= contrast <= 1
    assert sigma > 0

    in_channels = 3
    X_batch = np.zeros((batchsize, in_channels, size, size), dtype=np.float32)
    Y_batch = np.zeros((batchsize, in_channels, size, size), dtype=np.float32)
    label_batch = np.zeros(batchsize, dtype=np.float32)

    for i_batchsize in range(batchsize):

        if flag == 0:
            X = noise_gabor_patch(size, p, theta + dtheta, sf, 0, contrast, sigma, SD_1=SD_1,
                                  SD_2=SD_2)
            Y = noise_gabor_patch(size, p, theta, sf, 0, contrast, sigma, SD_1=SD_1,
                                  SD_2=SD_2)
            label = 0
        else:
            X = noise_gabor_patch(size, p, theta - dtheta, sf, 0, contrast, sigma,
                                  SD_1=SD_1,
                                  SD_2=SD_2)
            Y = noise_gabor_patch(size, p, theta, sf, 0, contrast, sigma, SD_1=SD_1,
                                  SD_2=SD_2)
            label = 1
        if scale == 255:
            X = np.tile(X * 255, (in_channels, 1, 1))
            Y = np.tile(Y * 255, (in_channels, 1, 1))

        X_batch[i_batchsize] = X
        Y_batch[i_batchsize] = Y
        label_batch[i_batchsize] = label

    X_batch = torch.from_numpy(X_batch)
    Y_batch = torch.from_numpy(Y_batch)
    label_batch = torch.from_numpy(label_batch)
    return X_batch, Y_batch, label_batch

def test_sample_generator2(batchsize, p, theta, contrast,
                           size=227, sf=40, sigma=50, scale=255, SD_1=10/255, SD_2=15/100):

    assert isinstance(batchsize, int)
    assert isinstance(size, int)
    assert sf > 0
    assert 0 <= contrast <= 1
    assert sigma > 0

    in_channels = 3
    Y_batch = np.zeros((batchsize, in_channels, size, size), dtype=np.float32)

    for i_batchsize in range(batchsize):

        Y = noise_gabor_patch(size, p, theta, sf, 0, contrast, sigma,
                              SD_1=SD_1, SD_2=SD_2)
        if scale == 255:
            Y = np.tile(Y * 255, (in_channels, 1, 1))

        Y_batch[i_batchsize] = Y

    Y_batch = torch.from_numpy(Y_batch)

    return Y_batch



if __name__ == '__main__':
    # gabor = gabor_patch(size=227, theta=30, sf=40,
    #                     phase=0, contrast=1, sigma=50)
    # cv2.imwrite("gabor_temp.png", gabor * 255)
    # noise_gabor = noise_gabor_patch(size=227, p=75, theta=35, sf=40,
    #                                 phase=0, contrast=1, sigma=50, SD_1=10/255, SD_2=0.15)
    # cv2.imwrite("noise_gabor_temp.png", noise_gabor * 255)
    training_sample = training_sample_generator(p=75, theta=35, contrast=1)
    test_sample = test_sample_generator(batchsize=5, p=75, theta=35, contrast=1, flag=0)

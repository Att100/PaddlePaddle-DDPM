import paddle
import numpy as np


def make_grid(batch_imgs, nrow, ncol, hspace=0, wspace=0):
    assert batch_imgs.shape[0] == nrow * ncol
    h, w = batch_imgs.shape[1:3]
    hs, ws = hspace, wspace
    image = paddle.ones((nrow*(h+hs)+hs, ncol*(w+ws)+ws, 3)).astype('float32') * 255
    for i in range(nrow):
        for j in range(ncol):
            image[i*(h+hs)+hs:(i+1)*(h+hs), j*(w+ws)+ws:(j+1)*(w+ws), :] = batch_imgs[i*ncol+j, :, :, :].astype('float32')
    return image

def build_infinite_sampler(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

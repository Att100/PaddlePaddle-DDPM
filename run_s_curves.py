import paddle
import paddle.nn as nn
import paddle.optimizer
import numpy as np
import paddle.io
import matplotlib.pyplot as plt
import random
from tqdm import trange
from sklearn.datasets import make_s_curve
import argparse

from models.diffusion import DiffusionModel
from models.simple import SimpleNet


def build_data():
    s_curve,_ = make_s_curve(10**4,noise=0.1)
    s_curve = s_curve[:,[0,2]]/10.0
    return s_curve

def build_betas(num_steps):
    betas = np.linspace(-6, 6, num_steps)
    betas = (1/(1+np.exp(-betas)))*(0.5e-2 - 1e-5)+1e-5
    return betas

def show_target_distribution(data):
    data = data.T
    fig, ax = plt.subplots()
    ax.scatter(*data,color='blue',edgecolor='white');
    ax.axis('off')
    plt.show()


class _DiffusionModel(DiffusionModel):
    def __init__(self, denoise_model, options: dict, betas=None):
        super().__init__(denoise_model, options, betas)

    def train(self, data=None):
        # training sample for point data
        nb = 10000 // self.options['batchsize']  # drop last
        # Step 1. build grad clipper and optimizer
        clip = nn.ClipGradByNorm(clip_norm=1.0)
        optimizer = paddle.optimizer.Adam(parameters=self.denoise_model.parameters(), learning_rate=1e-3, grad_clip=clip)
        indexes = [i for i in range(data.shape[0])]

        # Step 2. training loop
        pbar = trange(self.options['epochs'], dynamic_ncols=True)
        for e in pbar:
            random.shuffle(indexes)  # random shuffle
            for idx in range(nb):
                batch_x = data[indexes[idx*self.options['batchsize']:(idx+1)*self.options['batchsize']]]
                # Step 3. sample t
                t = paddle.randint(0, self.num_steps, shape=(self.options['batchsize'],))
                # Step 4. calculate loss and optimize model
                loss = self.diffusion_loss(batch_x, t)
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                # Step 5. update ema
                self.update_ema()
            pbar.set_postfix(loss='%.4f' % loss.numpy()[0])

def build_from_pretrained(path):
    state_dict = paddle.load(path)
    ddpm = _DiffusionModel(
        SimpleNet(), state_dict['options'],
        state_dict['betas'])
    ddpm.load_denoising_model_ckpt(state_dict['model'])
    return ddpm

if  __name__ == "__main__":
    # options
    options = {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'num_diffusion_steps': 100,
        'loss_type': 'noisepred',
        'batchsize': 512,
        'epochs': 1000,
        'ema_decay': 0.99,
        'ckpt_dir': "./ckpts",
        'pretrained_path': "./ckpts/ddpm_s_curves_e1000.pdparam",
        'mode': 'train',  # 'denoise': generate, 'train': train model
        'num_points': 10000
    }

    parser = argparse.ArgumentParser()
    for k, val in options.items():
        parser.add_argument("--"+k, type=type(val), default=val)

    _option = vars(parser.parse_args())
    for k, val in options.items():
        options[k] = _option[k]

    if options['mode'] == 'train':
        # train
        betas = build_betas(options['num_diffusion_steps'])
        data = build_data()

        ddpm = _DiffusionModel(denoise_model=SimpleNet(), options=options, betas=betas)
        ddpm.train(data=paddle.to_tensor(data))
        ddpm.save_state_dict("{}/ddpm_s_curves_e{}.pdparam".format(options['ckpt_dir'], options['epochs']))
    else:
        # denoise
        ddpm = build_from_pretrained(options['pretrained_path'])
        gen_data = ddpm.p_sample_loop((options['num_points'], 2), True)
        plt.scatter(gen_data[:, 0], gen_data[:, 1])
        plt.show()


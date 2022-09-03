import paddle
from PIL import Image
import os
import numpy as np
import argparse
from tqdm import trange

from models.fpn import FPN
from models.diffusion import DiffusionModel
from utils.utils import make_grid

SIZE = 128

class _DiffusionModel(DiffusionModel):
    def __init__(self, denoise_model, options: dict, betas=None, ema_model=None):
        super().__init__(denoise_model, options, betas, ema_model)

    @paddle.no_grad()
    def p_sample_loop_ddim(self, shape, stride, eta=0, enable_pbar=False):
        # DDIM (https://github.com/ermongroup/ddim)
        alphas_bar = paddle.cumprod(1 - paddle.concat([paddle.zeros([1]), self.betas.astype('float32')], axis=0), dim=0)
        seq = range(0, self.num_steps, stride)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [paddle.normal(shape=shape)]
        if enable_pbar:
            pbar = trange(0, len(seq_next), dynamic_ncols=True)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            if enable_pbar:
                pbar.update()
            at = alphas_bar[i+1]
            at_next = alphas_bar[j+1]
            xt = xs[-1]
            et = self.denoise_fn(xt, paddle.to_tensor([i]*shape[0], dtype='int64'))
            x0_t = (xt - et * paddle.sqrt((1 - at))) / paddle.sqrt(at)
            x0_preds.append(x0_t)
            c1 = eta * paddle.sqrt(((1 - at / at_next) * (1 - at_next) / (1 - at)))
            c2 = paddle.sqrt((1 - at_next) - c1 ** 2)
            xt_next = paddle.sqrt(at_next) * x0_t + c1 * paddle.normal(shape=shape) + c2 * et
            xs.append(xt_next)
        return xs, x0_preds

    def sample_image(self, path, postfix, stride=1, n_samples=1, batch_size=1, nrow=1, ncol=1, enable_pbar=False):
        if n_samples == 1:
            img = self.p_sample_loop_ddim((1, 3, SIZE, SIZE), stride, eta=0, enable_pbar=enable_pbar)[0][-1]
            img = paddle.clip((img+1)/2, 0, 1).reshape((3, SIZE, SIZE)).transpose((1, 2, 0)) * 255
            _img = Image.fromarray(np.uint8(img.numpy()))
            _img.save(os.path.join(path, "sample_{}.png".format(postfix)))
        else:
            if n_samples <= batch_size:
                imgs = self.p_sample_loop_ddim((n_samples, 3, SIZE, SIZE), stride, eta=0, enable_pbar=enable_pbar)[0][-1]
            else:
                n_steps = n_samples//batch_size
                remain = n_samples - n_steps * batch_size
                imgs = []
                for i in range(n_steps):
                    print("Step {} ...".format(i+1))
                    imgs.append(self.p_sample_loop_ddim((batch_size, 3, SIZE, SIZE), stride, eta=0, enable_pbar=enable_pbar)[0][-1])
                if remain != 0:
                    print("Step {} ...".format(n_steps+1))
                    imgs.append(self.p_sample_loop_ddim((remain, 3, SIZE, SIZE), stride, eta=0, enable_pbar=enable_pbar)[0][-1])
                imgs = paddle.concat(imgs, axis=0)
            imgs = paddle.clip((imgs+1)/2, 0, 1).transpose((0, 2, 3, 1)) * 255
            image = make_grid(imgs, nrow, ncol, 2, 2)
            _img = Image.fromarray(np.uint8(image.numpy()))
            _img.save(os.path.join(path, "sample_{}.png".format(postfix)))    
        return _img   

    def sample_sequence(self, path, postfix, stride=1, n_samples=1, interval=1, enable_pbar=False):
        # n_samples == batch_size !!!
        _, x0_preds = self.p_sample_loop_ddim(
            (n_samples, 3, SIZE, SIZE), stride, eta=0, enable_pbar=enable_pbar)
        assert len(x0_preds) % interval == 0
        # b * (n_steps/(interval*stride) * h * w * c
        seq_imgs = paddle.concat([im.transpose((0, 2, 3, 1)).unsqueeze(1) for im in x0_preds[::interval]], axis=1)  
        seq_imgs = paddle.clip((seq_imgs+1)/2, 0, 1).reshape((-1, SIZE, SIZE, 3)) * 255
        image = make_grid(seq_imgs, n_samples, self.num_steps//(stride*interval), 2, 2)
        _img = Image.fromarray(np.uint8(image.numpy()))
        _img.save(os.path.join(path, "sample_{}.png".format(postfix)))    
        return _img   


def build_from_pretrained(path):
    state_dict = paddle.load(path)
    ddpm = _DiffusionModel(
        FPN(
                num_steps=state_dict['options']['num_diffusion_steps'],
                ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2
            ), 
        state_dict['options'],
        state_dict['betas'])
    ddpm.load_denoising_model_ckpt(state_dict['model'])
    return ddpm


if __name__ == "__main__":
    # options
    options = {
        'stride': 10,
        'batchsize': 8,
        'sample_dir': "./sample",
        'pretrained_path': "./ckpts/ddpm_celeba_hq_i1000000_ema.pdparam",
        'num_images': '16-4-4',  # 64 images with 8 rows and 8 cols
        'sequence': False,  
        'sample_postfix': "ddim_celeba_hq_test",
    }

    parser = argparse.ArgumentParser()
    for k, val in options.items():
        parser.add_argument("--"+k, type=type(val), default=val)

    _options = vars(parser.parse_args())
    for k, val in _options.items():
        options[k] = val

    ddpm = build_from_pretrained(options['pretrained_path'])
    n_im, n_row, n_col = tuple([int(i) for i in options['num_images'].split("-")])
    ddpm.denoise_model.eval()
    if not options['sequence']:
        _img = ddpm.sample_image(
            options['sample_dir'], options['sample_postfix'], options['stride'], 
            n_im, n_im if n_im<=options['batchsize'] else options['batchsize'], n_row, n_col, True)
    else:
        _img = ddpm.sample_sequence(
            options['sample_dir'], options['sample_postfix'], options['stride'], 
            n_im, 5, True
        )

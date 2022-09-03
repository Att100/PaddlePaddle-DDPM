import os
import paddle
import paddle.nn as nn
import paddle.optimizer
import numpy as np
from PIL import Image
import paddle.io
import paddle.vision.transforms as transforms
from tqdm import trange
from visualdl import LogWriter
import argparse
import random

from models.diffusion import DiffusionModel
from models.fpn import FPN
from utils.utils import make_grid, build_infinite_sampler

SIZE=128

def build_celeba_hq_dataloader(path: str, batchsize: int, num_workers: int=8):
    class CelebAHQ(paddle.io.Dataset):
        def __init__(self, path: str):
            super().__init__()

            self.files = self.merge_paths(path)
            self.transform=transforms.Compose([
                transforms.Resize((SIZE, SIZE)),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx])
            return self.transform(img), 0

        def merge_paths(self, path):
            sub1 = ['train', 'val']
            sub2 = ['male', 'female']
            paths = []
            for s1 in sub1:
                for s2 in sub2:
                    dir_ = os.path.join(path, s1, s2)
                    paths += [os.path.join(dir_, name) for name in os.listdir(dir_)]
            random.shuffle(paths)
            return paths

    dataset = CelebAHQ(path)
    return paddle.io.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=num_workers)

def build_lr_scheduler(lr, warmup_steps, total_steps):
    remaining = total_steps - warmup_steps
    def warm_up_onecycle(step):
        # linear OneCycleLR (climb to lr and then decrease to 0 linearly)
        recenter = step - warmup_steps
        decay_ratio = (remaining - recenter) / remaining
        return min(step, warmup_steps) / warmup_steps * (1 if recenter <=0 else decay_ratio)
    lr_sch = paddle.optimizer.lr.LambdaDecay(lr, warm_up_onecycle)
    return lr_sch

class _DiffusionModel(DiffusionModel):
    def __init__(self, denoise_model, options: dict, betas=None, ema_model=None):
        super().__init__(denoise_model, options, betas, ema_model)

    def train(self, sampler, ckpt=None):
        # Step 1. build grad clipper and optimizer
        clip = nn.ClipGradByNorm(clip_norm=1.0)
        lr_sched = build_lr_scheduler(self.options['lr'], self.options['warmup_steps'], self.options['training_iters'])
        if ckpt is not None:
            lr_sched.set_state_dict(ckpt['lr_sched'])
        optimizer = paddle.optimizer.Adam(parameters=self.denoise_model.parameters(), learning_rate=lr_sched, grad_clip=clip)
        if ckpt is not None:
            optimizer.set_state_dict(ckpt['optim'])

        logwriter = LogWriter(self.options['log_dir'])

        # Step 2. training loop
        self.denoise_model.train()
        pbar = trange(0 if ckpt is None else ckpt['ckpt_idx']+1, self.options['training_iters'], dynamic_ncols=True)
        for step_idx in pbar:
            batch_x = next(sampler)
            # Step 3. sample t
            t = paddle.randint(0, self.num_steps, shape=(self.options['batchsize'],))
            # Step 4. calculate loss and optimize model
            loss = self.deffusion_loss(batch_x, t)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()

            # Step 5. update ema
            self.update_ema()
            pbar.set_postfix(loss='%.4f' % loss.numpy()[0])
            logwriter.add_scalar("loss", loss.numpy()[0], step_idx+1)
            logwriter.add_scalar("learning rate", lr_sched.get_lr(), step_idx+1)
            
            # Step 6. sample results
            if (step_idx+1) % self.options['sample_interval'] == 0:
                self.denoise_model.eval()
                _img = self.sample_image(self.options['sample_dir'], "iter{}".format(step_idx+1), 16, 16, 4, 4)
                logwriter.add_image("sample", np.array(_img), step_idx+1)
                self.denoise_model.train()

            # Step 7. save checkpoints
            if (step_idx+1) % self.options['ckpt_interval'] == 0:
                paddle.save({
                    'model': self.denoise_model.state_dict(),
                    'model_ema': self.ema_model.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                    'optim': optimizer.state_dict(),
                    'options': self.options,
                    'betas': self.betas.numpy(),
                    'ckpt_idx': step_idx
                }, os.path.join(self.options['ckpt_dir'], "celeba_hq_ckpt_iter_{}.pdparam".format(step_idx+1)))

    def sample_image(self, path, postfix, n_samples=1, batch_size=1, nrow=1, ncol=1, enable_pbar=False):
        if batch_size == 1:
            img = self.p_sample_loop((1, 3, SIZE, SIZE), enable_pbar)
            img = paddle.clip((img+1)/2, 0, 1).reshape((3, SIZE, SIZE)).transpose((1, 2, 0)) * 255
            _img = Image.fromarray(np.uint8(img.numpy()))
            _img.save(os.path.join(path, "sample_{}.png".format(postfix)))
        else:
            if n_samples <= batch_size:
                imgs = self.p_sample_loop((n_samples, 3, SIZE, SIZE), enable_pbar)
            else:
                n_steps = n_samples//batch_size
                remain = n_samples - n_steps * batch_size
                imgs = []
                for i in range(n_steps):
                    print("Step {} ...".format(i+1))
                    imgs.append(self.p_sample_loop((batch_size, 3, SIZE, SIZE), enable_pbar))
                if remain != 0:
                    print("Step {} ...".format(n_steps+1))
                    imgs.append(self.p_sample_loop((remain, 3, SIZE, SIZE), enable_pbar))
                imgs = paddle.concat(imgs, axis=0)
            imgs = paddle.clip((imgs+1)/2, 0, 1).transpose((0, 2, 3, 1)) * 255
            image = make_grid(imgs, nrow, ncol, 2, 2)
            _img = Image.fromarray(np.uint8(image.numpy()))
            _img.save(os.path.join(path, "sample_{}.png".format(postfix)))    
        return _img    

    def sample_sequence(self, path, postfix, n_samples=1, interval=1, enable_pbar=False):
        # n_samples == batch_size !!!
        _, x0_preds = self.p_sample_loop((n_samples, 3, SIZE, SIZE), enable_pbar, track_x0pred=True)
        assert len(x0_preds) % interval == 0
        # b * (n_steps/interval) * h * w * c
        seq_imgs = paddle.concat([im.transpose((0, 2, 3, 1)).unsqueeze(1) for im in x0_preds[::interval]], axis=1)  
        seq_imgs = paddle.clip((seq_imgs+1)/2, 0, 1).reshape((-1, SIZE, SIZE, 3)) * 255
        image = make_grid(seq_imgs, n_samples, self.num_steps//interval, 2, 2)
        _img = Image.fromarray(np.uint8(image.numpy()))
        _img.save(os.path.join(path, "sample_{}.png".format(postfix)))    
        return _img   

def resume_training_from_ckpt(path, sampler):
    ckpt = paddle.load(path)
    d_m = FPN(
        num_steps=ckpt['options']['num_diffusion_steps'],
        ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2
    )
    d_m.set_state_dict(ckpt['model'])
    ema_m = FPN(
        num_steps=ckpt['options']['num_diffusion_steps'],
        ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2
    )
    ema_m.set_state_dict(ckpt['model_ema'])
    ddpm = _DiffusionModel(d_m, ckpt['options'], ckpt['betas'], ema_m)
    ddpm.train(sampler, ckpt)
    return ddpm

def save_model_from_ckpt(src_path, dest_path, use_ema=True):
    ckpt = paddle.load(src_path)
    paddle.save(
        {
            'betas': ckpt['betas'], 
            'model': ckpt['model_ema'] if use_ema else ckpt['model'],
            'options': ckpt['options']},
        dest_path
    )

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

if  __name__ == "__main__":
    # options
    options = {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'num_diffusion_steps': 1000,
        'loss_type': 'noisepred',
        'batchsize': 24,
        'training_iters': 1000000,
        'num_workers': 8,
        'ema_decay': 0.995,
        'lr': 2e-4,
        'warmup_steps': 5000,
        'sample_interval': 1500,
        'ckpt_interval': 40000,
        'data_dir': "./datasets/celeba_hq",
        'ckpt_dir': "./ckpts",
        'sample_dir': "./sample",
        'log_dir': "./log",
        'pretrained_path': "./ckpts/ddpm_celeba_hq_i1000000_ema.pdparam",
        'resume_ckpt_path': "./ckpts/celeba_hq_ckpt_iter_40000.pdparam",
        'resume': False,
        'mode': 'train',  # 'denoise': generate, 'convert': convert ckpt for training to ckpt for denosing, 'train': train model
        'num_images': '16-4-4',  # 16 images with 4 rows and 4 cols
        'sequence': False,  
        'sample_postfix': "ddpm_celeba_hq_test",
        'convert_ema': True
    }

    parser = argparse.ArgumentParser()
    for k, val in options.items():
        parser.add_argument("--"+k, type=type(val), default=val)

    _options = vars(parser.parse_args())
    for k, val in _options.items():
        options[k] = val

    if options['mode'] == 'train':
        # train
        dataloader = build_celeba_hq_dataloader(options['data_dir'], options['batchsize'], options['num_workers'])
        sampler = build_infinite_sampler(dataloader)

        if not options['resume']:
            ddpm = _DiffusionModel(denoise_model=FPN(
                num_steps=options['num_diffusion_steps'],
                ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2
            ), options=options)
            ddpm.train(sampler=sampler)
        else:
            print("Resume Training")
            ddpm = resume_training_from_ckpt(options['resume_ckpt_path'], sampler)

        ddpm.save_state_dict(
            "{}/ddpm_celeba_hq_i{}_ema.pdparam".format(options['ckpt_dir'], options['training_iters']))
        ddpm.save_state_dict(
            "{}/ddpm_celeba_hq_i{}.pdparam".format(options['ckpt_dir'], options['training_iters']), save_ema=False)
    elif options['mode'] == 'convert':
        # convert
        save_model_from_ckpt(options["resume_ckpt_path"], options['pretrained_path'], options['convert_ema'])
    else:
        # denoise
        ddpm = build_from_pretrained(options['pretrained_path'])
        ddpm.denoise_model.eval()
        n_im, n_row, n_col = tuple([int(i) for i in options['num_images'].split("-")])
        if not options['sequence']:
            _img = ddpm.sample_image(
                options['sample_dir'], options['sample_postfix'], 
                n_im, n_im if n_im<=options['batchsize'] else options['batchsize'], n_row, n_col, True)
        else:
            _img = ddpm.sample_sequence(
                options['sample_dir'], options['sample_postfix'], 
                n_im, 50, True
            )

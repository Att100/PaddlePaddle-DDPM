import paddle
import paddle.optimizer
import numpy as np
import paddle.io
import copy
from tqdm import trange


def _warmup_beta(start, end, num_steps, warmup_frac):
  betas = end * np.ones(num_steps, dtype=np.float64)
  warmup_time = int(num_steps * warmup_frac)
  betas[:warmup_time] = np.linspace(start, end, warmup_time, dtype=np.float64)
  return betas

def get_beta_schedule(beta_schedule, start, end, num_steps):
    if beta_schedule == 'quad':
        betas = np.linspace(start ** 0.5, end ** 0.5, num_steps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(start, end, num_steps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(start, end, num_steps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(start, end, num_steps, 0.5)
    elif beta_schedule == 'const':
        betas = end * np.ones(num_steps, dtype=np.float64)
    elif beta_schedule == 'jsd': 
        betas = 1. / np.linspace(num_steps, 1, num_steps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_steps,)
    return betas

class DiffusionModel:
    def __init__(self, denoise_model, options: dict, betas=None, ema_model=None):
        self.num_steps = options['num_diffusion_steps']
        self.loss_type = options['loss_type']
        self.options = options

        if betas is None:
            betas = get_beta_schedule(
                options['beta_schedule'], options['beta_start'], 
                options['beta_end'], options['num_diffusion_steps'])
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = paddle.to_tensor(betas)
        self.alphas_cumprod = paddle.to_tensor(alphas_cumprod)
        self.alphas_cumprod_prev = paddle.to_tensor(alphas_cumprod_prev)

        self.denoise_model = denoise_model
        if ema_model is None:
            self.ema_model = copy.deepcopy(self.denoise_model)
        else:
            self.ema_model = ema_model
        self.ema_state_dict = None

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = paddle.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = paddle.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = paddle.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = paddle.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = paddle.sqrt(1. / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = paddle.to_tensor(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = paddle.to_tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = paddle.to_tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = paddle.to_tensor(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    def _extract(self, a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs, bs
        out = paddle.gather(a, t)
        assert out.shape == [bs]
        return paddle.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def denoise_fn(self, x, t):
        return self.denoise_model(x, t)

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.normal(shape=x_start.shape)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def diffusion_loss(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn(shape=x_start.shape)
        assert noise.shape == x_start.shape
        x_noisy = self.q_sample(x_start, t, noise)
        out = self.denoise_model(x_noisy.astype('float32'), t)
        assert x_noisy.shape == x_start.shape
        return paddle.mean(paddle.square(noise - out))

    def p_mean_variance(self, x, t, clip_denoised: bool, return_x0pred=False):
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x.astype('float32'), t))
        else:
            raise NotImplementedError(self.loss_type)

        if clip_denoised:
            x_recon = paddle.clip(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        # assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
        if return_x0pred:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        return model_mean, posterior_variance, posterior_log_variance, None

    def p_sample(self, x, t, clip_denoised=True, return_x0pred=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, x0pred = self.p_mean_variance(x, t, clip_denoised, return_x0pred)
        noise = paddle.normal(shape=x.shape)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = paddle.reshape(1 - paddle.equal(t, 0).astype('float32'), [x.shape[0]] + [1] * (len(x.shape) - 1))
        if return_x0pred:
            return model_mean + nonzero_mask * paddle.exp(0.5 * model_log_variance) * noise, x0pred
        return model_mean + nonzero_mask * paddle.exp(0.5 * model_log_variance) * noise, None

    @paddle.no_grad()
    def p_sample_loop(self, shape, enable_pbar=False, track_x0pred=False):
        assert isinstance(shape, (tuple, list))
        cur_x = paddle.normal(shape=shape).astype('float32')
        x0preds = []
        if not enable_pbar:
            for i in reversed(range(self.num_steps)):
                cur_x, x0pred = self.p_sample(
                    cur_x, paddle.to_tensor([i]*shape[0], dtype='int64'), return_x0pred=track_x0pred)
                if track_x0pred:
                    x0preds.append(x0pred)
        else:
            pbar = trange(self.num_steps, dynamic_ncols=True)
            reversed_steps = [self.num_steps-i for i in range(1, self.num_steps+1)]
            for i in pbar:
                cur_x, x0pred = self.p_sample(
                    cur_x, paddle.to_tensor([reversed_steps[i]]*shape[0], dtype='int64'), return_x0pred=track_x0pred)
                if track_x0pred:
                    x0preds.append(x0pred)
        if track_x0pred:
            return cur_x, x0preds
        return cur_x

    @paddle.no_grad()
    def update_ema(self):
        m_ema = dict(self.ema_model.named_parameters())
        m_cur = dict(self.denoise_model.named_parameters())

        for k in m_ema.keys():
            m_ema[k][:] = m_ema[k] * self.options['ema_decay'] + m_cur[k] * (1 - self.options['ema_decay'])    

    def train(self, data=None):
        pass

    def save_state_dict(self, path, save_ema=True):
        state_dict = {}
        state_dict['options'] = self.options
        state_dict['betas'] = self.betas.numpy()
        if save_ema:
            state_dict['model'] = self.ema_model.state_dict()
        else:
            state_dict['model'] = self.denoise_model.state_dict()
        paddle.save(state_dict, path)

    def load_denoising_model_ckpt(self, ckpt):
        self.denoise_model.set_state_dict(ckpt)
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class UpSample(nn.Layer):
    def __init__(self, with_conv, inch=None, outch=None):
        super().__init__()
        if with_conv:
            self.conv1 = nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Identity()

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        return x


class DownSample(nn.Layer):
    def __init__(self, with_conv, inch=None, outch=None):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.down = nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2D(kernel_size=2, padding=1)

    def forward(self, x, temb):
        return self.down(x)


class AttnBlock(nn.Layer):
    def __init__(self, ch):
        super().__init__()

        self.norm = nn.GroupNorm(32, ch)
        self.proj_q = nn.Conv2D(in_channels=ch, out_channels=ch, kernel_size=1)
        self.proj_k = nn.Conv2D(in_channels=ch, out_channels=ch, kernel_size=1)
        self.proj_v = nn.Conv2D(in_channels=ch, out_channels=ch, kernel_size=1)
        self.proj_out = nn.Conv2D(in_channels=ch, out_channels=ch, kernel_size=1)

    def forward(self, x):
        b, c, H, W = x.shape

        h = self.norm(x)

        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.transpose((0, 2, 3, 1)).reshape((b, H*W, c))
        k = k.reshape((b, c, H*W))
        v = v.transpose((0, 2, 3, 1)).reshape((b, H*W, c))

        w = paddle.bmm(q, k) * (int(c) ** (-0.5))
        w = w.reshape((b, H*W, H*W))
        w = F.softmax(w, axis=-1)
        h = paddle.bmm(w, v).reshape((b, H, W, c)).transpose((0, 3, 1, 2))
        return self.proj_out(h)
    


class ResNetBlock(nn.Layer):
    def __init__(self, inch, outch, tdim, dropout, attn=False):
        super().__init__()

        self.temb_proj = nn.Sequential(
            nn.Swish(),
            nn.Linear(tdim, outch)
        )

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, inch),
            nn.Swish(),
            nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, outch),
            nn.Swish(),
            nn.Dropout2D(dropout),
            nn.Conv2D(in_channels=outch, out_channels=outch, kernel_size=3, padding=1)
        )

        if inch != outch:
            self.shortcut = nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(outch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = x
        h = self.conv1(h)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return self.attn(h)


class TimeEmbedding(nn.Layer):
    def __init__(self, num_steps, emb_dim, out_dim):
        super().__init__()

        half_dim = emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(0, half_dim) * -emb).astype('float32')
        pos = paddle.arange(num_steps).astype('float32')
        emb = pos[: None] * emb[:, None]
        emb = paddle.stack([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        emb = emb.reshape((num_steps, emb_dim))

        pretrained_emb = nn.Embedding(num_steps, emb_dim)
        pretrained_emb.weight.set_value(emb)
        self.embedding_time = nn.Sequential(
            pretrained_emb,
            nn.Swish(),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, t):
        return self.embedding_time(t)


class UNet(nn.Layer):
    def __init__(self, num_steps, ch, ch_mult, num_res_blocks, attn_levels, dropout=0):
        super().__init__()

        self.time_emb = TimeEmbedding(num_steps, ch, ch*4)
        self.in_conv = nn.Conv2D(in_channels=3, out_channels=ch, kernel_size=3, padding=1)

        cur_ch = ch
        chs = [cur_ch]

        # Downsample
        self.downs = nn.LayerList()
        for i_level, mult in enumerate(ch_mult):
            outch = ch * mult
            for i_block in range(num_res_blocks):
                self.downs.append(
                    ResNetBlock(
                        inch=cur_ch, outch=outch, tdim=ch*4, dropout=dropout, attn=(i_level in attn_levels))
                )
                cur_ch = outch
                chs.append(cur_ch)
            if i_level != len(ch_mult)-1:
                self.downs.append(DownSample(True, cur_ch, cur_ch))
                chs.append(cur_ch)

        # Middle
        self.middles = nn.Sequential(
            ResNetBlock(cur_ch, cur_ch, ch*4, dropout, True),
            ResNetBlock(cur_ch, cur_ch, ch*4, dropout, False)
        )

        # Upsample
        self.ups = nn.LayerList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            outch = ch * mult
            for i_block in range(num_res_blocks+1):
                self.ups.append(
                    ResNetBlock(
                        inch=chs.pop()+cur_ch, outch=outch, tdim=ch*4, dropout=dropout, attn=(i_level in attn_levels))
                )
                cur_ch = outch
            if i_level != 0:
                self.ups.append(UpSample(True, cur_ch, cur_ch))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            nn.Swish(),
            nn.Conv2D(cur_ch, 3, kernel_size=3, padding=1)
        )


    def forward(self, x, t):
        temb = self.time_emb(t)

        h = self.in_conv(x)
        hs = [h]
        
        # down sample
        for layer in self.downs:
            h = layer(h, temb)
            hs.append(h)

        # middle
        for layer in self.middles:
            h = layer(h, temb)

        # up sample
        for layer in self.ups:
            if isinstance(layer, ResNetBlock):
                h = paddle.concat([h, hs.pop()], axis=1)
            h = layer(h, temb)

        out = self.out_conv(h)
        return out


if __name__ == "__main__":
    batch_size = 8
    model = UNet(
        num_steps=1000, ch=128, ch_mult=[1, 2, 2, 2], attn_levels=[1],
        num_res_blocks=2, dropout=0.1)
    x = paddle.randn((batch_size, 3, 32, 32))
    t = paddle.randint(high=1000, shape=(batch_size, ))
    y = model(x, t)
    print(y.shape)

            

        

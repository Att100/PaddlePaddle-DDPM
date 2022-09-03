import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# This is a simpler implementation which can save training time

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
            self.down = nn.AvgPool2D(kernel_size=2)

    def forward(self, x, temb):
        return self.down(x)


class ResNetBlock(nn.Layer):
    def __init__(self, inch, outch, tdim):
        super().__init__()

        self.temb_proj = nn.Sequential(
            nn.Swish(),
            nn.Linear(tdim, outch)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=3, padding=1),
            nn.Swish(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=outch, out_channels=outch, kernel_size=3, padding=1),
            nn.Swish()
        )

        if inch != outch:
            self.shortcut = nn.Conv2D(in_channels=inch, out_channels=outch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.out_norm = nn.GroupNorm(32, outch)

    def forward(self, x, temb):
        h = x
        h = self.conv1(h)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return self.out_norm(h)


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


class FPN(nn.Layer):
    def __init__(self, num_steps, ch, ch_mult, num_res_blocks):
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
                        inch=cur_ch, outch=outch, tdim=ch*4)
                )
                cur_ch = outch
                chs.append(cur_ch)
            if i_level != len(ch_mult)-1:
                self.downs.append(DownSample(False, cur_ch, cur_ch))

        # Middle
        self.middle = ResNetBlock(cur_ch, cur_ch, ch*4)

        # Upsample
        chs = [chs[0]] + chs
        cur_ch = chs.pop()
        self.ups = nn.LayerList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            if i_level != len(ch_mult)-1:
                self.ups.append(UpSample(False, cur_ch, cur_ch))
            for i_block in range(num_res_blocks):
                outch = chs.pop()
                self.ups.append(
                    ResNetBlock(
                        inch=cur_ch, outch=outch, tdim=ch*4)
                )
                cur_ch = outch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
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
        h = self.middle(h, temb)
        hs.pop()

        # up sample
        for layer in self.ups:
            h = layer(h, temb)
            h = h + hs.pop()

        out = self.out_conv(h)
        return out


if __name__ == "__main__":
    batch_size = 8
    model = FPN(
        num_steps=1000, ch=128, ch_mult=[1, 1, 2, 2, 4, 4],
        num_res_blocks=2)
    x = paddle.randn((batch_size, 3, 128, 128))
    t = paddle.randint(high=1000, shape=(batch_size, ))
    y = model(x, t)
    print(y.shape)

            

        

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ASFBLite(nn.Module):
    """
    
    Cross-Gated Attention Fusion

    """
    def __init__(self, dim, ratio=4):
        super(ASFBLite, self).__init__()
        reduced_dim = max(dim // ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, reduced_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(reduced_dim, dim * 2, kernel_size=1, bias=False),
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.cross_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.output = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x, y):
        B, C, H, W = x.shape
        concat = torch.cat([x, y], dim=1)
        
        avg_out = self.channel_mlp(self.avg_pool(concat))
        max_out = self.channel_mlp(self.max_pool(concat))
        channel_att = torch.sigmoid(avg_out + max_out)
        concat_ch = concat * channel_att
        
        avg_spatial = concat_ch.mean(dim=1, keepdim=True)
        max_spatial = concat_ch.max(dim=1, keepdim=True)[0]
        x_avg, y_avg = x.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_spatial, max_spatial, x_avg, y_avg], dim=1)
        spatial_att = self.spatial_conv(spatial_feat)  # [B, 2, H, W]
        
        x_att = spatial_att[:, 0:1, :, :]
        y_att = spatial_att[:, 1:2, :, :]
        
        cross_att = self.cross_gate(concat)
        concat_gated = concat_ch * cross_att
        
        x_out, y_out = concat_gated.chunk(2, dim=1)
        fused = x_out * x_att + y_out * y_att
        
        return self.output(torch.cat([fused, x + y], dim=1)[:, :C*2]) + (x + y) * 0.5

class AHEB(nn.Module):
    """
    Adaptive Hybrid Enhancement Block (AHEB) - Lightweight Version
    Cross-Frequency Enhancement Block
    轻量化改进:
    1. Channel Splitting: 输入通道减半分别处理局部和全局特征，显著降低计算量
    2. 简化局部分支: 使用轻量级 3x3 卷积替代复杂的 CB 模块
    3. 简化门控机制: 降低中间层通道数，移除空间自适应调制
    4. 移除冗余投影: 优化卷积通道配置
    """
    def __init__(self, dim, ratio_spatial=1, window_sizes=[16, 32], sigmas=[4, 8]):
        super(AHEB, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.other_dim = dim - self.half_dim
        self.num_scales = len(window_sizes)
        
        self.conv_in = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, kernel_size=3, padding=1, groups=self.half_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.half_dim, self.half_dim, kernel_size=1, bias=False) 
        )
        
        for i, (ws, sigma) in enumerate(zip(window_sizes, sigmas)):
            kernel = self._gaussian_kernel(ws, sigma).repeat(self.other_dim, 1, 1, 1)
            self.register_buffer(f'kernel_{i}', kernel, persistent=False)
        
        self.window_sizes = window_sizes
        
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.other_dim, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.global_refine = nn.Conv2d(self.other_dim, self.other_dim, kernel_size=3, padding=1, groups=self.other_dim, bias=False)
        
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= torch.sum(kernel_2d)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _apply_gaussian(self, x, kernel_idx):
        """应用指定尺度的高斯卷积"""
        kernel = getattr(self, f'kernel_{kernel_idx}')
        k = kernel.shape[-1]
        pad_size = k - 1
        pad_l = pad_size // 2
        pad_r = pad_size - pad_l
        pad_t = pad_size // 2
        pad_b = pad_size - pad_t
        x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        return F.conv2d(x_padded, kernel, padding=0, groups=x.shape[1])

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        x_proj = self.conv_in(x)
        x_local, x_global = torch.split(x_proj, [self.half_dim, self.other_dim], dim=1)
        
        x_local = self.local_branch(x_local)
        
        scale_weights = self.scale_attention(x_global)
        
        x_global_fused = 0
        for i in range(self.num_scales):
            w = scale_weights[:, i:i+1, :, :]
            x_global_fused = x_global_fused + w * self._apply_gaussian(x_global, i)
            
        x_global_fused = self.global_refine(x_global_fused)
        
        combined = torch.cat([x_local, x_global_fused], dim=1)
        gate = self.freq_gate(combined)
        gate_local, gate_global = torch.split(gate, [self.half_dim, self.other_dim], dim=1)
        
        x_local_gated = x_local * gate_global
        x_global_gated = x_global_fused * gate_local
        
        out = self.conv_out(torch.cat([x_local_gated, x_global_gated], dim=1))
        
        return identity + out

class CB(nn.Module):
    def __init__(self, dim, ratio = 1, ks = 3):
        super(CB, self).__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(dim, dim * ratio, kernel_size=ks, stride=1, padding="same", groups=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim * ratio, dim * ratio, kernel_size=ks, stride=1, padding="same", groups=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim * ratio, dim, kernel_size=ks, stride=1, padding="same", groups=1, bias=False))
    def forward(self,x):
        return self.cb(x) + x


class DUC(nn.Module):
    """
    轻量级动态更新单元 (Dynamic Update Cell, DUC)
    
    适用于计算资源受限场景，保留核心动态特性的同时降低计算量
    
    Args:
        dim: 输入/输出通道数
    """
    def __init__(self, dim):
        super(DUC, self).__init__()
        
        self.gate_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)
        self.gate_pw = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False)
        
        self.candidate_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)
        self.candidate_pw = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        
        hidden_dim = max(1, dim // 4)
        self.step_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h, x):
        concat = torch.cat([h, x], dim=1)
        
        gate_feat = self.gate_pw(self.gate_dw(concat))
        z, r = gate_feat.chunk(2, dim=1)
        z = self.sigmoid(z)
        r = self.sigmoid(r)
        
        h_reset = torch.cat([r * h, x], dim=1)
        h_candidate = self.tanh(self.candidate_pw(self.candidate_dw(h_reset)))
        
        step = self.step_fc(concat).view(-1, 1, 1, 1)
        
        h_new = z * h + (1 - z) * h_candidate
        h_new = h + step * (h_new - h)
        
        return h_new
class UNet(nn.Module):
    def __init__(self, in_dim, feature_dim, ratio_spatial, ratio_fusion, middle_blk_num=2, enc_blk_nums=[1,1], dec_blk_nums=[1,1]):
        super(UNet,self).__init__()
        self.to_hidden = nn.Conv2d(in_dim, feature_dim, kernel_size=3, stride=1, padding="same", groups=1, bias=False)
        self.project_out = nn.Conv2d(feature_dim, in_dim, kernel_size=3, stride=1, padding="same", groups=1, bias=False)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        dim = feature_dim
        self.encoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(enc_blk_nums[0])]))
        self.downs.append(nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, bias=False,padding="same"), nn.PixelUnshuffle(2)))

        dim = dim * 2
        self.encoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(enc_blk_nums[1])]))
        self.downs.append(nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, bias=False,padding="same"), nn.PixelUnshuffle(2)))

        dim = dim * 2
        self.middle_blks = nn.Sequential(*[AHEB(dim, ratio_spatial) for _ in range(middle_blk_num)])


        self.ups.append(nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, bias=False,padding="same"), nn.PixelShuffle(2)))
        dim = dim // 2
        self.fusions.append(ASFBLite(dim,ratio_fusion))
        self.decoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(dec_blk_nums[0])]))

        self.ups.append(nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, bias=False,padding="same"), nn.PixelShuffle(2)))
        dim = dim // 2
        self.fusions.append(ASFBLite(dim, ratio_fusion))
        self.decoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(dec_blk_nums[1])]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.to_hidden(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, fusion, enc_skip in zip(self.decoders, self.ups, self.fusions, encs[::-1]):
            
            x = up(x)
            x = fusion(x,enc_skip)
            x = decoder(x)
        x = self.project_out(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
class CDB(nn.Module):
    def __init__(self, in_dim, kernel_size = 3):
        super(CDB, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_maker = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, bias=False,padding="same"),
                                          nn.LeakyReLU(),
                                          nn.Conv2d(in_dim, in_dim, kernel_size=3, bias=False,padding="same"),
                                          nn.Conv2d(in_dim, in_dim * kernel_size * kernel_size, kernel_size=1, bias=False,padding="same"))
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self,x,y):
        b, c, h, w = x.shape
        kernel = self.kernel_maker(x + y)
        kernel = kernel.reshape([b, c, self.kernel_size * self.kernel_size, h, w])
        x_unfold = self.unfold(x).reshape(b, c, -1, h, w)
        x = (x_unfold * kernel).sum(2) + x
        return x


class ST(nn.Module):
    def __init__(self, control_nums):
        super(ST, self).__init__()
        self.weight_a = nn.Parameter(torch.randn(control_nums))
        self.weight_b = nn.Parameter(torch.randn(control_nums))
        self.control_nums = control_nums

    def forward(self,x):
        freq = torch.arange(1, self.control_nums + 1, device=x.device) * math.pi
        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.control_nums)
        Fourier_sin = torch.sin(freq * x)
        Fourier_cos = torch.cos(freq * x)
        x = torch.sum(self.weight_a * Fourier_sin + self.weight_b * Fourier_cos, dim=-1)
        return x

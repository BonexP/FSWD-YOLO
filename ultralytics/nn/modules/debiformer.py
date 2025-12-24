import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# --------------------------
# Helper Classes
# --------------------------

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query, key):
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        self.mul_weight = mul_weight

    def forward(self, r_idx, r_weight, kv):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        return topk_kv


class QKVConv(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Conv2d(dim, qk_dim + qk_dim + dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=1)
        return q, kv


# --------------------------
# Attention Modules
# --------------------------

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, topk=4, side_dwconv=3):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = dim
        self.scale = self.qk_dim ** -0.5
        self.topk = topk

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim)
        self.router = TopkRouting(qk_dim=self.qk_dim, topk=self.topk, param_routing=False, diff_routing=False)
        self.kv_gather = KVGather(mul_weight='none')
        self.qkv_conv = QKVConv(self.dim, self.qk_dim)
        self.attn_act = nn.Softmax(dim=-1)
        self.wo = nn.Conv2d(dim, dim, 1)
        self.kv_down = nn.Identity()  # Assuming identity for simplicity in YOLO context

    def forward(self, x):
        # x: NCHW
        N, C, H, W = x.size()

        # Padding
        pad_l = pad_t = 0
        pad_r = (self.n_win - W % self.n_win) % self.n_win
        pad_b = (self.n_win - H % self.n_win) % self.n_win
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, H_pad, W_pad = x.size()

        # QKV
        q, kv = self.qkv_conv(x)  # q: N, C, H, W

        # Reshape for Window Processing
        # q: (n, p^2, w, w, c) via einops
        q = rearrange(q, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = rearrange(kv, 'n p2 h w c -> n p2 (h w) c')  # No downsample

        # Window-level QK
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        # LEPE
        lepe = self.lepe(x)  # NCHW
        lepe = rearrange(lepe, "n c (j h) (i w) -> n (j h) (i w) c", j=self.n_win, i=self.n_win)

        # Routing
        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # Attention
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel

        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H_pad // self.n_win, w=W_pad // self.n_win)

        out = out + lepe
        out = rearrange(out, 'n h w c -> n c h w')
        out = self.wo(out)

        # Unpad
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]
        return out


class DeBiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, topk=4, side_dwconv=3, input_res=None):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = dim
        self.scale = self.qk_dim ** -0.5
        self.topk = topk

        # Dynamic config based on channel dimension for YOLO flexibility
        self.n_groups = max(1, dim // 64)
        self.stride_def = 2 if dim >= 256 else (4 if dim >= 128 else 8)  # Heuristic for stride based on depth
        if dim >= 512: self.stride_def = 1

        # Calculate q_size for RPE table (default assumption if input_res not provided)
        # Assuming standard YOLO input 640. If dim=256 (P3/P4), size is approx 20-40.
        # We set a sufficient size for RPE.
        base_size = 40
        self.q_h, self.q_w = (base_size, base_size)

        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.num_heads // self.n_groups
        self.kk = 3  # kernel size for offset

        self.rpe_table = nn.Parameter(torch.zeros(self.num_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
        trunc_normal_(self.rpe_table, std=0.01)

        self.lepe1 = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=self.stride_def, padding=side_dwconv // 2,
                               groups=dim)

        self.router = TopkRouting(qk_dim=self.qk_dim, topk=self.topk)
        self.kv_gather = KVGather(mul_weight='none')
        self.qkv_conv = QKVConv(self.dim, self.qk_dim)
        self.attn_act = nn.Softmax(dim=-1)
        self.kv_down = nn.Identity()

        # Deformable Projections
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim, dim, 1)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.unifyheads1 = nn.Conv2d(dim, dim, 1)

        # Offset Network
        self.conv_offset_q = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, self.stride_def, self.kk // 2,
                      groups=self.n_group_channels, bias=False),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False),
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ConvFFN(dim, int(dim * 3))

    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device), indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device), indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        N, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # Padding
        pad_l = pad_t = 0
        pad_r = (self.n_win - W % self.n_win) % self.n_win
        pad_b = (self.n_win - H % self.n_win) % self.n_win
        x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, H_pad, W_pad = x_pad.size()

        # QKV
        q, kv = self.qkv_conv(x_pad)
        q_bi = rearrange(q, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        kv_bi = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)

        kv_pix = rearrange(kv_bi, 'n p2 h w c -> n p2 (h w) c')

        # LEPE (on unpadded/padded?) - Use padded for consistency
        lepe1 = self.lepe1(x_pad)  # Strided conv

        # ---------------- Deformable Offset ----------------
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups)
        offset_q = self.conv_offset_q(q_off)  # B*g 1 Hg Wg
        Hk, Wk = offset_q.size(2), offset_q.size(3)

        offset_q = offset_q.tanh()  # Optional: mul(range_factor)
        offset_q = rearrange(offset_q, 'b p h w -> b h w p')

        reference = self._get_ref_points(Hk, Wk, N, dtype, device)
        pos_k = (offset_q + reference).clamp(-1., +1.)

        # Sample Q
        x_sampled_q = F.grid_sample(
            input=x.reshape(N * self.n_groups, self.n_group_channels, H, W),  # Sample from original x
            grid=pos_k[..., (1, 0)],  # y,x -> x,y
            mode='bilinear', align_corners=True)
        q_sampled = x_sampled_q.reshape(N, C, Hk, Wk)

        # Pad q_sampled if needed for window attention
        pad_rg_s = (self.n_win - Wk % self.n_win) % self.n_win
        pad_bg_s = (self.n_win - Hk % self.n_win) % self.n_win
        q_sampled_pad = F.pad(q_sampled, (0, pad_rg_s, 0, pad_bg_s))
        lepe1 = F.pad(lepe1, (0, pad_rg_s, 0, pad_bg_s))  # Align lepe with sampled size

        # ---------------- Bi-Level Routing ----------------
        queries_def = self.proj_q(q_sampled_pad)
        queries_def = rearrange(queries_def, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q_win, k_win = queries_def.mean([2, 3]), kv_bi[..., 0:self.qk_dim].mean([2, 3])
        r_weight, r_idx = self.router(q_win, k_win)

        kv_gather = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_gather, v_gather = kv_gather.split([self.qk_dim, self.dim], dim=-1)

        # Attention
        k_g = rearrange(k_gather, 'n p2 k hw (m c) -> (n p2) m c (k hw)', m=self.num_heads)
        v_g = rearrange(v_gather, 'n p2 k hw (m c) -> (n p2) m (k hw) c', m=self.num_heads)
        q_g = rearrange(queries_def, 'n p2 h w (m c) -> (n p2) m (h w) c', m=self.num_heads)

        attn_weight = (q_g * self.scale) @ k_g
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_g

        out_def = rearrange(out, '(n j i) m (h w) c -> n (m c) (j h) (i w)', j=self.n_win, i=self.n_win,
                            h=q_sampled_pad.size(2) // self.n_win, w=q_sampled_pad.size(3) // self.n_win)
        out_def = out_def + lepe1
        out_def = self.unifyheads1(out_def)
        out_def = q_sampled_pad + out_def

        # Norm & MLP
        out_def_norm = rearrange(out_def, 'b c h w -> b h w c')
        out_def_norm = self.norm2(out_def_norm)
        out_def_norm = rearrange(out_def_norm, 'b h w c -> b c h w')
        out_def = out_def + self.mlp(out_def_norm)

        # ---------------- Deformable Agent Attention ----------------
        # Simplified for integration: Project back to original resolution
        out_def_norm = rearrange(out_def, 'b c h w -> b h w c')
        out_def_norm = self.norm(out_def_norm)
        out_def_norm = rearrange(out_def_norm, 'b h w c -> b c h w')

        k_def = self.proj_k(out_def_norm)
        v_def = self.proj_v(out_def_norm)
        q_orig = self.proj_q(x_pad)  # Use original x as query for final reconstruction

        # Flatten for attention
        k_pix_sel = rearrange(k_def, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        v_pix_sel = rearrange(v_def, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        q_pix = rearrange(q_orig, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)

        attn = torch.einsum('b c m, b c n -> b m n', q_pix, k_pix_sel)  # B*h, HW_orig, HW_sampled
        attn = attn.mul(self.scale)

        # RPE Bias (Simplified interpolation if size mismatch)
        rpe_bias = self.rpe_table
        if rpe_bias.shape[1] < H_pad or rpe_bias.shape[2] < W_pad:
            rpe_bias = F.interpolate(rpe_bias.unsqueeze(0), size=(H_pad * 2 - 1, W_pad * 2 - 1), mode='bilinear',
                                     align_corners=False).squeeze(0)

        # Just applying basic softmax here as displacement logic is complex for general plugin
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('b m n, b c n -> b c m', attn, v_pix_sel)
        out = rearrange(out, '(n m) c (h w) -> n (m c) h w', m=self.num_heads, h=H_pad, w=W_pad)
        out = self.proj_out(out)

        # Unpad
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]

        return out


# --------------------------
# Main YOLO Module
# --------------------------

class DebiFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, n_win=7, topk=4, drop_path=0.):
        super().__init__()
        # Use BiLevel for standard or DeBi for complex.
        # Here we implement the alternating structure from the paper's Block if desired
        # Or just use the DeBiLevelRoutingAttention as the primary "DebiFormer" feature.
        # Based on user request "DebiFormer module", we use the full DeBi capability.

        self.attn = DeBiLevelRoutingAttention(dim, num_heads=num_heads, n_win=n_win, topk=topk)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ConvFFN(dim, int(dim * 4))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x is NCHW
        # Norms expect NHWC usually in Transfomer, but we wrapped attention to handle NCHW
        # Let's handle Norm manually

        # Shortut 1: Attention
        shortcut = x
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm1(x_norm)
        x_norm = rearrange(x_norm, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(self.attn(x_norm))

        # Shortcut 2: MLP
        shortcut = x
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm2(x_norm)
        x_norm = rearrange(x_norm, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(self.mlp(x_norm))

        return x


class DebiFormer(nn.Module):
    """
    YOLOv11 compatible DebiFormer Module.
    Arguments:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of blocks
        shortcut (bool): Shortcut (unused in this specific block structure usually, but kept for interface)
        g (int): Groups (unused)
        e (float): Expansion (unused, or used for bottleneck)
        k (int): Top-k regions
        n_win (int): Window size
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=4, n_win=7):
        super().__init__()
        self.c = c2
        # Input projection if channels change
        self.cv1 = nn.Conv2d(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        num_heads = max(1, c2 // 32)
        self.m = nn.Sequential(*(DebiFormerBlock(c2, num_heads=num_heads, n_win=n_win, topk=k) for _ in range(n)))

    def forward(self, x):
        x = self.cv1(x)
        x = self.m(x)
        return x
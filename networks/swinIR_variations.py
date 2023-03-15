from networks.swin_transformer_sr import *
from networks.common import default_conv, MeanShift, UpSampler
from networks.meta_upscale import MetaUpSampler
import torch.nn.functional as F

"""
Under working:
    1. Patch Embedding/UnEmbedding -> wavelet xfm and ifm;
    2.0: seems no need of basic STL layer.
    x 2. Basic STL layer (n STL + conv or no conv, w/o residual connection)
    3. Based on the basic STL Layer -> 
        3.1 Dense Block
        3.2 RDN
        3.3 RRDB
        3.4 DBPN block
    4. Meanshift
    5. Upsampler -> MetaSR
    6. GAN -> Change the loss
"""


class DenseSTLayer(nn.Module):
    """ A basic dense layer consists of two basic Swin Transformer layer:
            x -> Swin (window_shift = 0) -> Swin (window_shift = half window_size) -> * dense scale, cat with x
        input:
            x: tensor N x P x T -> N x P x (T+G), where G is the growth rate (hidden dim)
        The depth is suggested as 2, so that two swin transformer blocks will be used,
            the first without shift operation
            while the second do the shift window operation.
        To build very deep networks, the window_size is set as 2 (or 4) by default.
        All the above settings need to be verified in the experiments.
        Args:
            input_dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
            growth_rate (int): Number of channels (G) of the final output
            dense_scale (float): the scale factor for the main pathway
            dim_modify_mode: where to add mlp layer to modify the dims of inputs to outputs
        """

    def __init__(self, input_dim, input_resolution, depth=2, num_heads=6, window_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=30., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 growth_rate=60, dense_scale=1., dim_modify_mode='tail'):

        super(DenseSTLayer, self).__init__()

        assert growth_rate % num_heads == 0, 'growth_rate % num_heads should be 0'
        assert input_dim % num_heads == 0, 'token dim % num_heads should be 0'

        if dim_modify_mode == 'head':
            if input_dim != growth_rate:
                self.head = nn.Sequential(
                    nn.Linear(input_dim, growth_rate),
                    norm_layer(growth_rate)
                )
            else:
                self.head = nn.Identity()
            hidden_dim = growth_rate
            self.tail = nn.Identity()

        elif dim_modify_mode == 'tail':
            self.head = nn.Identity()
            hidden_dim = input_dim
            if hidden_dim != growth_rate:
                self.tail = nn.Sequential(
                    nn.Linear(hidden_dim, growth_rate),
                    norm_layer(growth_rate)
                )
            else:
                self.tail = nn.Identity()

        self.body = BasicLayer(
            hidden_dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, norm_layer, downsample, use_checkpoint
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.dense_scale = dense_scale
        self.growth_rate = hidden_dim
        self.depth = depth

    def forward(self, x, x_size):
        short_cut = x
        x = self.head(x)
        x = self.body(x, x_size)
        x = self.tail(x)
        x = torch.cat((short_cut, x.mul(self.dense_scale)), 2)
        return x

    # def extra_repr(self) -> str:
    #     return f"hidden_dim={self.hidden_dim}, input_resolution={self.input_resolution}, depth={self.depth}, " \
    #         f"growth_rate={self.growth_rate}, dense_scale={self.dense_scale}, input_dim={self.input_dim}"

    def flops(self):
        # todo: modify here, add the flops of linear layers (if there are)
        flops = 0
        flops += self.body.flops()
        return flops


class RDSTB(nn.Module):
    """Residual Dense Swin Transformer Block (RDSTB).
    DSTL + DSTL + DSTL + Conv (residual connection)
    Args:
        growth_rate:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.


    """

    def __init__(self, input_dim, input_resolution, layer_depth, num_heads=6, window_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 growth_rate=0, dense_scale=1., dim_modify_mode='tail',
                 num_blocks=3, residual_scale=1.):
        super(RDSTB, self).__init__()

        self.input_dim = input_dim
        self.input_resolution = input_resolution
        self.residual_scale = residual_scale

        idim = input_dim
        self.body = nn.ModuleList([])
        for i in range(int(num_blocks)):
            self.body.append(
                DenseSTLayer(
                    input_dim=idim,
                    input_resolution=input_resolution,
                    depth=layer_depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                    growth_rate=growth_rate,
                    dense_scale=dense_scale,
                    dim_modify_mode=dim_modify_mode
                )
            )
            idim += growth_rate

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(idim, input_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(idim, idim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(idim // 4, idim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(idim // 4, input_dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=input_dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=idim,
            norm_layer=None)

    def forward(self, x, x_size):
        short_cut = x

        for m in self.body:
            x = m(x, x_size)

        x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))).mul(self.residual_scale)
        return x + short_cut

    # def extra_repr(self) -> str:
    #     s = f"hidden_dim={self.hidden_dim}, input_resolution={self.input_resolution}, depth={self.depth}, " \
    #         f"growth_rate={self.growth_rate}, dense_scale={self.dense_scale}, input_dim={self.input_dim}"
    #     return s

    def flops(self):
        # todo: modify here
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class RRDSTB(nn.Module):
    """Residual in Residual Dense Swin Transformer Block (RRDSTB).
    RDSTB + RDSTB + RDSTB (residual connection)
    Args:
        growth_rate:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.


    """

    def __init__(self, input_dim, input_resolution, layer_depth, num_heads=6, window_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 growth_rate=30, dense_scale=1., dim_modify_mode='tail',
                 num_blocks_in_rdb=3, rdb_residual_scale=1.,
                 num_blocks_in_rrdb=3, rrdb_residual_scale=1.):
        super(RRDSTB, self).__init__()

        self.input_dim = input_dim
        self.input_resolution = input_resolution
        self.residual_scale = rrdb_residual_scale

        self.body = nn.ModuleList([])
        for i in range(int(num_blocks_in_rrdb)):
            self.body.append(
                RDSTB(
                    input_dim=input_dim,
                    input_resolution=input_resolution,
                    layer_depth=layer_depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                    growth_rate=growth_rate,
                    dense_scale=dense_scale,
                    dim_modify_mode=dim_modify_mode,
                    num_blocks=num_blocks_in_rdb,
                    residual_scale=rdb_residual_scale
                )
            )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(input_dim, input_dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(input_dim // 4, input_dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(input_dim // 4, input_dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=input_dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=input_dim,
            norm_layer=None)

    def forward(self, x, x_size):
        short_cut = x

        for m in self.body:
            x = m(x, x_size)

        x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))).mul(self.residual_scale)
        return x + short_cut

    # def extra_repr(self) -> str:
    #     s = f"hidden_dim={self.hidden_dim}, input_resolution={self.input_resolution}, depth={self.depth}, " \
    #         f"growth_rate={self.growth_rate}, dense_scale={self.dense_scale}, input_dim={self.input_dim}"
    #     return s

    def flops(self):
        # todo: modify here
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


# class ESTSR(nn.Module):
#     """
#     A SR Network consists of Residual in Residual Dense Swin Transformer Blocks (RRDSTB).
#         Meanshift + Head_conv + n * RRDSTB + conv tail + Meanshift (residual connection)
#     Args:
#         growth_rate:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#         img_size: Input image size.
#         patch_size: Patch size.
#         resi_connection: The convolutional block before residual connection.
#
#
#     """
#     # todo: modify this parameters, modify the comments.
#     # todo: add multi-scale upsampler?
#     # todo: once complete this, try ST-MetaRDN?
#     def __init__(self, img_size=48, patch_size=1, in_chans=1, sr_scale=2, embed_dim=60,
#                  dense_layer_depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6],
#                  window_size=[4, 4, 4, 4], rdb_depths=[3, 3, 3, 3],
#                  rrdb_depths=[3, 3, 3, 3], num_rrdb_blocks=4,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop=0., drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, resi_connection='1conv',
#                  growth_rate=30, dense_scale=1., dim_modify_mode='tail',
#                  rdb_residual_scale=1.,
#                  rrdb_residual_scale=1.,
#                  global_res_scale=1.,
#                  mean=None, std=None,
#                  act_in_conv='leaky_relu', bn_in_conv=None,
#                  scale_free=False):
#         super(ESTSR, self).__init__()
#
#         # ## basic information
#         self.input_resolution = img_size
#         self.patch_size = patch_size
#         self.input_channel = in_chans
#
#         # ## networks
#         self.num_blocks = num_rrdb_blocks
#         self.n_feats = embed_dim
#         self.patch_norm = patch_norm
#         self.ape = ape
#         self.use_checkpoint = use_checkpoint
#         self.drop_path_rate = drop_path_rate
#         self.drop_rate = drop_rate
#         self.resi_connection = resi_connection
#         self.global_res_scale = global_res_scale
#         # swin transformer block
#         self.window_size = window_size
#         self.mlp_ratio = mlp_ratio
#         self.qkv_bias = qkv_bias
#         self.qk_scale = qk_scale
#         self.norm_layer = norm_layer
#         self.num_heads = num_heads
#         # DSTL
#         self.dense_layer_depths = dense_layer_depths
#         self.dense_scale = dense_scale
#         self.growth_rate = growth_rate
#         self.dim_modify_mode = dim_modify_mode
#         # RDSTB
#         self.rdb_depths = rdb_depths
#         self.rdb_residual_scale = rdb_residual_scale
#         # RRDSTB
#         self.rrdb_depths = rrdb_depths
#         self.rrdb_residual_scale = rrdb_residual_scale
#
#         # conv
#         if act_in_conv == 'relu':
#             self.act_in_conv = nn.ReLU(True)
#         elif act_in_conv == 'leaky_relu':
#             self.act_in_conv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         elif act_in_conv == 'prelu':
#             self.act_in_conv = nn.PReLU()
#         else:
#             raise ValueError('Invalid activation {}, should be one of [relu, leaky_relu, prelu]'.format(act_in_conv))
#         self.bn_in_conv = bn_in_conv
#         # ## up-scale
#         self.sr_scale = sr_scale
#         self.scale_free = scale_free
#
#         # ## mean shift
#         if mean is None:
#             mean = [0. for _ in range(self.input_channel)]
#         if std is None:
#             std = [1. for _ in range(self.input_channel)]
#         if len(mean) != len(std) or len(mean) != self.input_channel:
#             raise ValueError(
#                 'Dimension of mean {} / std {} should fit input channels {}'.format(
#                     len(mean), len(std), self.input_channel
#                 )
#             )
#         self.mean = mean
#         self.std = std
#         self.add_mean = MeanShift(mean, std, 'add')
#         self.sub_mean = MeanShift(mean, std, 'sub')
#
#         # ## shallow feature extraction
#         self.head = default_conv(in_chans, embed_dim, 3)
#
#         # ## Swin TR based deep feature extraction
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # build the main body with RRDSTB
#         self.body = nn.ModuleList()
#         for i_block in range(self.num_blocks):
#             layer = RRDSTB(
#                 input_dim=embed_dim,
#                 input_resolution=(patches_resolution[0],
#                                   patches_resolution[1]),
#                 layer_depth=self.dense_layer_depths[i_block],
#                 num_heads=self.num_heads[i_block],
#                 window_size=self.window_size[i_block],
#                 mlp_ratio=self.mlp_ratio,
#                 qkv_bias=self.qkv_bias,
#                 qk_scale=self.qk_scale,
#                 drop=self.drop_rate,
#                 attn_drop=attn_drop,
#                 img_size=self.input_resolution,
#                 patch_size=self.patch_size,
#                 resi_connection=self.resi_connection,
#                 growth_rate=self.growth_rate,
#                 dense_scale=self.dense_scale,
#                 dim_modify_mode=self.dim_modify_mode,
#                 num_blocks_in_rdb=self.rdb_depths[i_block],
#                 rdb_residual_scale=self.rdb_residual_scale,
#                 num_blocks_in_rrdb=self.rrdb_depths[i_block],
#                 rrdb_residual_scale=self.rrdb_residual_scale
#             )
#             self.body.append(layer)
#
#         self.norm = norm_layer(self.n_feats)
#
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             # to save parameters and memory
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         # ## high quality image reconstruction
#         # upsample layers + reconstruciton layer
#         if self.scale_free:
#             self.tail = MetaUpSampler(self.n_feats, self.input_channel, 3)
#         else:
#             if self.sr_scale > 1:
#                 m_tail = [UpSampler(default_conv, self.sr_scale, self.n_feats, act=None, bn=self.bn_in_conv)]
#             else:
#                 m_tail = []
#             m_tail.append(default_conv(self.n_feats, self.input_channel, 3))
#             self.tail = nn.Sequential(*m_tail)
#
#         # ## initial weights
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def forward_features(self, x):
#         # N x C x H x W -> N x P x T (because patch_size = 1 so T = C)
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.body:
#             x = blk(x, x_size)
#
#         x = self.norm(x)  # B L C
#         x = self.patch_unembed(x, x_size)
#
#         return x
#
#     def forward(self, x, sr_scale=None):
#         x = self.sub_mean(x)
#         x = self.head(x)
#
#         # lr feature maps
#         res = self.forward_features(x).mul(self.global_res_scale)
#         res += x
#
#         # up-sample
#         if self.scale_free:
#             x = self.tail(res, sr_scale)
#         else:
#             x = self.tail(res)
#
#         x = self.add_mean(x)
#
#         return x
#
#     def extra_repr(self):
#         return ''
#
#     def flops(self):
#         return None


class RDSTSR(nn.Module):
    """
    A SR Network consists of Residual Dense Swin Transformer Blocks (RDSTB).
        Meanshift + Head_conv + n * RDSTB + conv tail + Meanshift (residual connection)
    Args:
        growth_rate:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.


    """
    def __init__(self, img_size=48, patch_size=1, in_chans=1, sr_scale=2, embed_dim=60,
                 dense_layer_depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6],
                 window_size=[4, 4, 4, 4], rdb_depths=[3, 3, 3, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 growth_rate=30, dense_scale=1., dim_modify_mode='tail',
                 rdb_residual_scale=1.,
                 global_res_scale=1.,
                 mean=None, std=None,
                 act_in_conv='leaky_relu', bn_in_conv=None,
                 scale_embedding=False,
                 feature_maps_only=False):
        super(RDSTSR, self).__init__()

        # ## basic information
        self.input_resolution = img_size
        self.patch_size = patch_size
        self.input_channel = in_chans

        # ## networks
        self.num_blocks = len(rdb_depths)
        assert len(rdb_depths) == len(window_size) == len(num_heads) == len(dense_layer_depths)
        self.n_feats = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.use_checkpoint = use_checkpoint
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.resi_connection = resi_connection
        self.global_res_scale = global_res_scale
        # swin transformer block
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.norm_layer = norm_layer
        self.num_heads = num_heads
        # DSTL
        self.dense_layer_depths = dense_layer_depths
        self.dense_scale = dense_scale
        self.growth_rate = growth_rate
        self.dim_modify_mode = dim_modify_mode
        # RDSTB
        self.rdb_depths = rdb_depths
        self.rdb_residual_scale = rdb_residual_scale

        # conv
        if act_in_conv == 'relu':
            self.act_in_conv = nn.ReLU(True)
        elif act_in_conv == 'leaky_relu':
            self.act_in_conv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif act_in_conv == 'prelu':
            self.act_in_conv = nn.PReLU()
        else:
            raise ValueError('Invalid activation {}, should be one of [relu, leaky_relu, prelu]'.format(act_in_conv))
        self.bn_in_conv = bn_in_conv
        # ## up-scale
        self.sr_scale = sr_scale
        # for multi-scale tasks, scale embedding
        self.scale_embedding = scale_embedding
        if scale_embedding:
            # this has not been done, probably should be done in the SwinBlock
            self.scale_embedding = nn.Identity()
            self.scale_embed_layer = nn.Linear(1, 1)

        # ## feature maps only
        self.feature_maps_only = feature_maps_only

        # ## mean shift
        if mean is None:
            mean = [0. for _ in range(self.input_channel)]
        if std is None:
            std = [1. for _ in range(self.input_channel)]
        if len(mean) != len(std) or len(mean) != self.input_channel:
            raise ValueError(
                'Dimension of mean {} / std {} should fit input channels {}'.format(
                    len(mean), len(std), self.input_channel
                )
            )
        self.mean = mean
        self.std = std
        self.add_mean = MeanShift(mean, std, 'add')
        self.sub_mean = MeanShift(mean, std, 'sub')

        # ## shallow feature extraction
        self.head = default_conv(in_chans, embed_dim, 3)

        # ## Swin TR based deep feature extraction
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build the main body with RRDSTB
        self.body = nn.ModuleList()
        for i_block in range(self.num_blocks):
            layer = RDSTB(
                input_dim=embed_dim,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                layer_depth=self.dense_layer_depths[i_block],
                num_heads=self.num_heads[i_block],
                window_size=self.window_size[i_block],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                img_size=self.input_resolution,
                patch_size=self.patch_size,
                resi_connection=self.resi_connection,
                growth_rate=self.growth_rate,
                dense_scale=self.dense_scale,
                dim_modify_mode=self.dim_modify_mode,
                num_blocks=self.rdb_depths[i_block],
                residual_scale=self.rdb_residual_scale,
            )
            self.body.append(layer)

        self.norm = norm_layer(self.n_feats)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ## high quality image reconstruction
        # upsample layers + reconstruciton layer
        if self.sr_scale > 1:
            m_tail = [UpSampler(default_conv, self.sr_scale, self.n_feats, act=None, bn=self.bn_in_conv)]
        else:
            m_tail = []
        m_tail.append(default_conv(self.n_feats, self.input_channel, 3))
        self.tail = nn.Sequential(*m_tail)

        # ## initial weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # N x C x H x W -> N x P x T (because patch_size = 1 so T = C)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for blk in self.body:
            x = blk(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        if not self.feature_maps_only:
            x = self.sub_mean(x)

        # ## pad the inputs to fit window_size
        x, ori_size = self.pad(x)
        x = self.head(x)

        # lr feature maps
        res = self.forward_features(x).mul(self.global_res_scale)
        res += x

        # ## ipad
        res = self.ipad(res, ori_size)

        if self.feature_maps_only:
            return res
        else:
            # up-scample
            x = self.tail(res)
            x = self.add_mean(x)
            return x

    def pad(self, x):
        window_size = self.window_size[0]
        h, w = x.shape[-2:]
        ph = int(math.ceil(h / window_size) * window_size - h)
        pw = int(math.ceil(w / window_size) * window_size - w)
        x = F.pad(x, [0, pw, 0, ph], 'replicate')
        return x, [h, w]

    def ipad(self, x, ori_size):
        ori_h, ori_w = ori_size
        return x[:, :, :ori_h, :ori_w]

    def extra_repr(self):
        return ''

    def flops(self):
        return None


def make_RDSTSR(paras, mean=None, std=None, feature_maps_only=False):
    """
    Get parameters of RDSTSR net, build the model and return
    """
    # ## basic information
    img_size = paras.patch_size   # lr_patch_size
    in_chans = paras.input_channel
    sr_scale = paras.sr_scale
    patch_size = paras.swin_patch_size

    # Swin Transformer block
    mlp_ratio = paras.swin_hidden_ratio
    qkv_bias = paras.swin_qkv_bias
    qk_scale = paras.swin_qk_scale
    drop_rate = paras.swin_drop_rate
    attn_drop_rate = paras.swin_attn_drop_rate
    drop_path_rate = paras.swin_drop_path_rate

    # ## RDSTSR parameters
    embed_dim = paras.rdst_embed_dim
    dense_layer_depths = paras.rdst_dense_layer_depths
    num_heads = paras.rdst_num_heads
    window_size = paras.rdst_window_size
    rdb_depths = paras.rdst_rdb_depths
    layer_norm = paras.rdst_layer_norm
    norm_layer = nn.LayerNorm if layer_norm else nn.Identity
    ape = paras.rdst_ape
    patch_norm = paras.rdst_patch_norm
    use_checkpoint = paras.rdst_use_checkpoint
    resi_connection = paras.rdst_res_connection
    growth_rate = paras.rdst_growth_rate
    dense_scale = paras.rdst_dense_scale
    dim_modify_mode = paras.rdst_dim_modify_mode
    rdb_residual_scale = paras.rdst_rdb_residual_scale
    global_res_scale = paras.rdst_global_res_scale
    act_in_conv = paras.rdst_act_in_conv
    bn_in_conv = paras.rdst_bn_in_conv

    # img_size = int(img_size // sr_scale // window_size + 1) * window_size
    sr_scale = int(sr_scale)

    img_size = int(img_size / 4)

    # build model
    model = RDSTSR(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        sr_scale=sr_scale, embed_dim=embed_dim,
        dense_layer_depths=dense_layer_depths, num_heads=num_heads,
        window_size=window_size, rdb_depths=rdb_depths,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        drop_rate=drop_rate, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
        norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
        use_checkpoint=use_checkpoint, resi_connection=resi_connection,
        growth_rate=growth_rate, dense_scale=dense_scale, dim_modify_mode=dim_modify_mode,
        rdb_residual_scale=rdb_residual_scale, global_res_scale=global_res_scale,
        mean=mean, std=std,
        act_in_conv=act_in_conv, bn_in_conv=bn_in_conv,
        feature_maps_only=feature_maps_only)

    return model



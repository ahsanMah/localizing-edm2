import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


def build_flows(
    latent_size, num_flows=4, num_blocks=2, hidden_units=128, context_size=64
):
    # Define flows

    flows = []
    for i in range(num_flows):
        flows += [
            nf.flows.MaskedAffineAutoregressive(
                latent_size,
                hidden_units,
                context_features=context_size,
                num_blocks=num_blocks,
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribution
    q0 = nf.distributions.DiagGaussian(2, trainable=False)

    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    return model


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class SpatialNormer(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv.weight.data.fill_(1)  # all ones weights
        self.conv.weight.requires_grad = False  # freeze weights

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x.square()).pow_(0.5)


class PatchFlow(torch.nn.Module):
    def init(
        self,
        input_size,
        patch_size=3,
        context_embedding_size=128,
        num_blocks=2,
        hidden_units=128,
    ):
        c, h, w = input_size
        self.local_pooler = SpatialNormer(in_channels=c, kernel_size=patch_size)
        self.flow = build_flows()
        self.position_encoding = PositionalEncoding2D(channels=context_embedding_size)

        # caching pos encs
        self.position_encoding(self.local_pooler(torch.empty((1,c,h,w))))

    def init_weights(self):
        # Initialize weights with Xavier
        linear_modules = list(
            filter(lambda m: isinstance(m, nn.Linear), self.flow.modules())
        )
        total = len(linear_modules)
        # pdb.set_trace()
        for idx, m in enumerate(linear_modules):
            # Last layer gets init w/ zeros
            if idx == total - 1:
                nn.init.zeros_(m.weight.data)
            else:
                nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, x, chunk_size=32):

        x_norm = self.local_pooler(x)
        context = self.position_encoding(x_norm)

        # (Patches * batch) x channels
        local_ctx = rearrange(context, "b c h w -> (h w) b c")
        patches = rearrange(x, "b c h w -> (h w) b c")

        nchunks = (patches.shape[0] + chunk_size - 1) // chunk_size
        patches = patches.chunk(nchunks, dim=0)
        ctx_chunks = local_ctx.chunk(nchunks, dim=0)
        zs, jacs = [], []

        # gc = repeat(global_ctx, "b c -> (n b) c", n=self.patch_batch_size)

        for p, ctx in zip(patches, ctx_chunks):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            ctx = rearrange(ctx, "n b c -> (n b) c")
            p = rearrange(p, "n b c -> (n b) c")

            z, ldj = self.flow.inverse_and_log_det(p, context=ctx)

            zs.append(z)
            jacs.append(ldj)

            del ctx, p

        return zs, jacs

    @staticmethod
    def stochastic_step(
        scores, x_batch, flow_model, opt=None, train=False, n_patches=1
    ):
        if train:
            flow_model.train()
            opt.zero_grad(set_to_none=True)
        else:
            flow_model.eval()

        patches, context = PatchFlow.get_random_patches(
            scores, x_batch, flow_model, n_patches
        )

        patch_feature = patches.to(flow_model.device)
        context_vector = context.to(flow_model.device)
        patch_feature = rearrange(patch_feature, "n b c -> (n b) c")
        context_vector = rearrange(context_vector, "n b c -> (n b) c")

        global_pooled_image = flow_model.global_pooler(x_batch)
        global_context = flow_model.global_attention(global_pooled_image)
        gctx = repeat(global_context, "b c -> (n b) c", n=n_patches)

        # Concatenate global context to local context
        context_vector = torch.cat([context_vector, gctx], dim=1)

        z, ldj = flow_model.flow.inverse_and_log_det(
            patch_feature,
            context=context_vector,
        )

        loss = flow_model.nll(z, ldj) * n_patches

        if train:
            loss.backward()
            opt.step()

        return loss.item() / n_patches

    @staticmethod
    def get_random_patches(scores, x_batch, flow_model, n_patches):
        h = flow_model.local_pooler(scores).cpu()
        flow_model.position_encoder = flow_model.position_encoder.cpu()
        local_patches = rearrange(h, "b c h w -> (h w) b c")
        context = rearrange(flow_model.position_encoder(h), "b c h w d -> (h w) b c")

        # Get random patches
        total_patches = local_patches.shape[0]
        shuffled_idx = torch.randperm(total_patches)
        rand_idx_batch = shuffled_idx[:n_patches]

        return local_patches[rand_idx_batch], context[rand_idx_batch]

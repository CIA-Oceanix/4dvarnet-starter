import torch
import torch.nn.functional as F
import torchvision
import kornia.filters as kfilts

# U-Net block
class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size,padding=1)
        self.relu  = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size,padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(torch.nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024),
                       pools=(2,2,2,2,2),
                       kernel_size=3):
        super().__init__()
        self.enc_blocks = torch.nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)])
        self.pools      = torch.nn.ModuleList([torch.nn.MaxPool2d(pools[i]) for i in range(len(pools))])

    def forward(self, x):
        ftrs = []
        for i in range(len(self.pools)):
            x = self.enc_blocks[i](x)
            ftrs.append(x)
            x = self.pools[i](x)
        return ftrs

class Decoder(torch.nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), pools=(2,2,2,2), pads=(0,0,0,0)):
        super().__init__()
        self.chs        = chs
        self.upconvs    = torch.nn.ModuleList([torch.nn.ConvTranspose2d(chs[i], chs[i+1], pools[i], pools[i]) for i in range(len(chs)-1)])
        self.dec_blocks = torch.nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(torch.nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), pools=(2,2,2,2,2),
                       dec_chs=(1024, 512, 256, 128, 64), pads = (0,0,0,0),
                       retain_dim=False, out_sz=(572,572), kernel_size=3):
        super().__init__()
        self.encoder = Encoder(enc_chs, pools, kernel_size)
        self.decoder = Decoder(dec_chs, pools[:-1], pads)
        self.num_class = 8*enc_chs[0] # the SPDE parameters in high-dim space
        self.head = torch.nn.Conv2d(dec_chs[-1], self.num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out

class PriorNet(torch.nn.Module):
    def __init__(self, dim_in,  dim_hidden, nparam, kernel_size=3, downsamp=None):

        super().__init__()

        self.nt = dim_in
        self.nparam = nparam
        self.dim_param = self.nt * nparam
        #self.dim_hidden_param = dim_hidden*self.dim_param
        self.dim_hidden_param = 100
        self.bilin_quad = False

        # state estimation
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # parameter estimation
        self.conv_in_param = torch.nn.Conv2d(
            self.dim_param, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden_param = torch.nn.Conv2d(
            self.dim_hidden_param, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.add_grad_info = torch.nn.Conv2d(
            self.dim_hidden_param + 3*self.nt, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # rebuild the augmented state
        self.aug_conv_out = torch.nn.Conv2d(
            dim_in + self.dim_hidden_param, dim_in + self.dim_param, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward(self, state):

        n_b, _, n_y, n_x = state.shape
        state = self.down(state)

        # state estimation
        x = self.conv_in(state[:,:self.nt,:,:])
        x = self.conv_hidden(F.relu(x))
        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )

        # parameter estimation
        theta = self.conv_in_param(state[:,self.nt:,:,:])
        theta = self.conv_hidden_param(F.relu(theta))
        theta = self.add_grad_info(torch.cat([x, 
                                              torch.reshape(kfilts.spatial_gradient(x,normalized=True),
                                                            (n_b, 2*self.nt, n_y, n_x)),
                                              theta], dim=1))

        # final state
        state = self.aug_conv_out(
            torch.cat([x, theta], dim=1)
        )
        state = self.up(state)
        return state


class UNetPriorCost(torch.nn.Module):
    def __init__(self, dim_in, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):

        super().__init__()
        self.dim_in = dim_in
        self.nt = nt
        self.unet = UNet(enc_chs=(self.dim_in,self.dim_in*2,self.dim_in*4,self.dim_in*8),
                              pools=(2,2,2),
                              dec_chs=(self.dim_in*8,self.dim_in*4,self.dim_in*2),
                              pads=(0,0),
                              kernel_size=3)

    def forward_ae(self, x):
        x = self.unet(x)
        return x

    def forward(self, state, exclude_params=False):
        if not exclude_params:
            return F.mse_loss(state, self.forward_ae(state))
        else:
            return F.mse_loss(state[:,:self.nt,:,:], self.forward_ae(state)[:,:self.nt,:,:])

class AugPriorCost(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, nparam, kernel_size=3, downsamp=None):

        super().__init__()
        self.nt = dim_in
        self.priornet = PriorNet(dim_in, dim_hidden, nparam, kernel_size, downsamp)

    def forward_ae(self, x):
        x = self.priornet(x)
        return x

    def forward(self, state, exclude_params=False):
        if not exclude_params:
            return F.mse_loss(state, self.forward_ae(state))
        else:
            return F.mse_loss(state[:,:self.nt,:,:], self.forward_ae(state)[:,:self.nt,:,:])


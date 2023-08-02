import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, cin, cout, ks=3):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=(1, ks, ks), padding=(0, ks//2, ks//2)),
            nn.ReLU(),
            nn.BatchNorm3d(cout),
        )

        self.skip_mod = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, kernel_size=(1,3,3), padding=(0, 1, 1))

    def forward(self, x):
       return F.relu(self.mod(x) + self.skip_mod(x)) 

class DownBlock(nn.Module):
    def __init__(self, cin,  cout, ks=3):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=(1,ks,ks), stride= (1, 2, 2), padding=(0, ks//2, ks//2)),
            nn.ReLU(),
            nn.BatchNorm3d(cout),
        )

    def forward(self, x):
        return self.mod(x)


class CNNEncoder(nn.Module):
    def __init__(self, dim_in=1, kernel_size=3):
        super().__init__()
        self.mod = nn.Sequential(
            DownBlock(dim_in, 16, ks=kernel_size),
            ResBlock(16, 16, ks=kernel_size),
            DownBlock(16, 32, ks=kernel_size),
            ResBlock(32, 32, ks=kernel_size),
            DownBlock(32, 32, ks=kernel_size),
            ResBlock(32, 32, ks=kernel_size),
        )

    def forward(self, x):
        return self.mod(x)

class CNNDecoder(nn.Module):
    def __init__(self, dim_in=32, kernel_size=3):
        super().__init__()
        self.mod = nn.Sequential(
            ResBlock(32, 32, ks=kernel_size),
            nn.Upsample(scale_factor=(1, 2, 2)),
            ResBlock(32, 16, ks=kernel_size),
            nn.Upsample(scale_factor=(1, 2, 2)),
            ResBlock(16, 8, ks=kernel_size),
            nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=(1,1,1)),
        )

    def forward(self, x):
        return self.mod(x)

class ConvLstm(nn.Module):
    def __init__(self, dim_in=32, dim_hidden=16, kernel_size=3):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x, hidden, cell):
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)

        hidden = out_gate * torch.tanh(cell)

        out = self.conv_out(hidden)
        return out, hidden, cell

class BidirectionalConvLstm2d(nn.Module):
    def __init__(self, encoder, decoder, convlstm):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.convlstm = convlstm

    def forward(self, x):
        x = self.encoder(x)
        x = self.bidirectional(x)
        x = self.decoder(x)
        return x

    def bidirectional(self, x):
        inps = x.unbind(2) 

        b, _, nx, ny = inps[0].shape
        # init
        out = None
        hidden, cell = (
            torch.randn((b, self.convlstm.dim_hidden, nx, ny), device=x.device),
            torch.randn((b, self.convlstm.dim_hidden, nx, ny), device=x.device),
        )

        # forward 
        for inp in inps:
            out, hidden, cell = self.convlstm(inp, hidden, cell)


        outs = [out]
        for inp in inps[:-1][::-1]:
            out, hidden, cell = self.convlstm(inp, hidden, cell)
            outs.append(out)

        return torch.stack(outs[::-1], dim=2)

class Interp2dWrapper(nn.Module):
    def __init__(self, mod, inp_shape, inp_pattern='...', out_pattern='...'):
        super().__init__()
        self.mod = mod
        self.inp_shape=tuple(inp_shape)
        self.inp_pattern=inp_pattern
        self.out_pattern=out_pattern

    def prior_cost(self, x, *args, **kwargs):
        return 0.

    def init_state(self, x, *args, **kwargs):
        return 0.

    def forward(self, batch):
        x = batch.input.nan_to_num()
        out_shape = x.shape[-2:]
        x = F.interpolate(x, self.inp_shape, mode='bilinear')
        x = einops.rearrange(x, self.inp_pattern + '->' + self.out_pattern)
        x = self.mod(x)
        x = einops.rearrange(x, self.out_pattern + '->' + self.inp_pattern)
        x = F.interpolate(x, out_shape, mode='bilinear')
        return x



if __name__ == '__main__':
    import hydra
    from omegaconf import OmegaConf
    from hydra.core.config_store import ConfigStore
    import config
    with hydra.initialize('4dvarnet-starter/config', version_base="1.3"):
        # print(ConfigStore().list('/'))

        cfg = hydra.compose('main.yaml', overrides=['xp=base', '+params=convlstm'])

        OmegaConf.register_new_resolver("hydra", lambda p: p, replace=True)
        dm = hydra.utils.call(cfg.datamodule)
        mod = hydra.utils.call(cfg.model)
        trainer = hydra.utils.call(cfg.trainer)

    dm.setup()
    _batch = next(iter(dm.train_dataloader()))
    mod = mod.to('cuda')
    mod._norm_stats = dm.norm_stats()
    batch = mod.transfer_batch_to_device(_batch, mod.device, 0)
    opt = torch.optim.Adam(mod.parameters())

    opt.zero_grad()
    loss = mod.training_step(batch, 0)
    print(mod.solver.mod.encoder.mod[0].mod[0].weight.grad)
    loss.backward()
    print(mod.solver.mod.encoder.mod[0].mod[0].weight.grad)
    print(mod.solver.mod.decoder.mod[-1].weight.grad)
    opt.step()
    print(loss)


    
    print('toto')
    x = torch.rand(2, 30, 240, 240)
    _mod = BidirectionalConvLstm2d(
        encoder=CNNEncoder(1),
        convlstm=ConvLstm(32),
        decoder=CNNDecoder(32),
    )
    mod = Interp2dWrapper(mod=_mod, inp_shape=(128, 128), inp_pattern='batch time lat lon', out_pattern='batch () time lat lon')
    mod = mod.to('cuda')
    x = x.to('cuda')
    out=mod(x)
    print(out.shape)


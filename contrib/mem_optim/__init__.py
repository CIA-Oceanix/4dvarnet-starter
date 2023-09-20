import torch
import src.models


def generator(path):
    all_files = sorted(path.glob('*.t'))
    idxes = set([f.name.split('_')[0] for f in all_files])
    nodes = set([f.name.split('_')[1] for f in all_files])
    for idx in idxes:
        batches = [torch.load(path / f'{idx}_{node}.t') for node in sorted(list(nodes))]
        for 

    

class LitModel(src.models.Lit4dVarNet):
    def test_step(self, batch, batch_idx):
        out = super().test_step(batch, batch_idx)
        if self.logger:
            data_path = Path(self.logger.log_dir)/'test_data'
        else:
            data_path = Path(f'{self.save_dir}/test_data')
        data_path.mkdir(exist_ok=True, parents=True)
        torch.save(out, data_path / f'{batch_idx:06}_{self.global_rank:02}.t')

    def on_test_epcoh


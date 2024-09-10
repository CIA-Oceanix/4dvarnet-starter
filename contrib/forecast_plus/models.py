import pandas as pd
import numpy as np
from pathlib import Path

from src.models import Lit4dVarNetForecast


class Plus4dVarNetForecast(Lit4dVarNetForecast):
    def __init__(
            self,
            *args,
            rec_weight_fn,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn

    def on_test_epoch_end(self):
        dims = self.rec_weight.size()
        dT = dims[0]
        metrics = []
        output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
        for i in range(output_start, 7):
            forecast_weight = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, forecast_weight
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data_leadtime = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            if self.logger:
                test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                

            metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
            metrics_leadtime = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            metrics.append(metrics_leadtime)

        print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())
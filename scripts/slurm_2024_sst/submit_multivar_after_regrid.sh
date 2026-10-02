#!/bin/bash
# submit the two multivariate SST inferences once both SLA files are on the SST grid
cd /Odyssey/private/d21botvy
until grep -q "REGRID EXIT" regrid_sla_to_sst_grid.log; do sleep 30; done
grep -q "REGRID EXIT 0" regrid_sla_to_sst_grid.log || { echo "regrid failed, nothing submitted"; exit 1; }
sbatch launch_SST_UNet_multivar_inference_2024_nadir.sh
sbatch launch_SST_UNet_multivar_inference_2024_nadir_swot.sh

#!/bin/bash
# ./launch_sst_wgeo.sh dm3 all_baltic_wgeo_linweight_dt15

diro=/DATASET/mbeauchamp/DMI/results/lightning_logs

dm=$1
xpname=$2

list_outputs=()
# run 4DVarNet sequentially for the 12 months
for month in {1..12} ; do
  echo "Run 4DVarNet on month"${month}
  if [ $dm = "all" ] ; then 
    CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 python main.py xp=dmi/dmi_sst_test_${month}_oibench_wgeo
  else
    CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 python main.py xp=dmi/dmi_sst_test_${month}_oibench_wgeo_${dm}
  fi
  N=`ls -Art ${diro} | tail -n 1 | cut -f2 -d'_'`
  mm1=$((month-1))
  mp1=$((month+1))
  if [ ${mm1} == 0 ]; then
    mm1=12
  fi
  if [ ${mp1} == 13 ]; then
    mp1=1
  fi
  # extract only relevant days from NetCDF files
  cdo delete,month=$mm1 ${diro}/version_${N}/test_data.nc ${diro}/version_${N}/test_data_${month}_tmp.nc
  cdo delete,month=$mp1 ${diro}/version_${N}/test_data_${month}_tmp.nc ${diro}/version_${N}/test_data_${month}.nc
  rm -rf ${diro}/version_${N}/test_data_${month}_tmp.nc
  # append to the list of NetCDF files to merge
  list_outputs+=("${diro}/version_${N}/test_data_${month}.nc")
done

echo ${list_outputs[*]}
# merge the NetCDF files
ncrcat ${list_outputs[*]} -O ${diro}/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_${dm}.nc
mv -f ${diro}/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_${dm}.nc /DATASET/mbeauchamp/DMI/4DVarNet_outputs/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_${xpname}_${dm}.nc

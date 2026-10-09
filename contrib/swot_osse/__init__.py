"""SWOT + nadir OSSE training with an error model for the SWOT swaths (new, separate from the existing SWOT OSSE code).

- prepare_cls_masks_2024: daily nadir and SWOT-swath masks from the 2024 CLS gridded products
- data.open_glorys12_data_sla_swot_errors: GLORYS SLA sampled with those masks, with noise on the swaths and SWOT dropout
- train.finetune_from_weights: fine-tuning that loads only the weights of an existing checkpoint
"""

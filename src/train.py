
def base_training(trainer, dm, lit_mod, test_fn=None):
    if trainer.logger is not None:
        print('Logdir:', trainer.logger.log_dir)

    lit_mod.norm_stats = dm.norm_stats()
    trainer.fit(lit_mod, datamodule=dm)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')

    if test_fn is not None:
        test_fn(lit_mod)




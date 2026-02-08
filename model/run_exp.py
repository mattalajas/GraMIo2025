import torch
from gramio import Gramio
from gramio_filler import GramioFiller
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tsl import logger
from tsl.data import ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.experiment import Experiment
from tsl.metrics import torch as torch_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy
from utils import (AirCross, CrossSpatioTemporalDataset, StandardScalerSplit, SpatioTemporalDataModule,
                   CrossGPVARDataset, add_missing_sensors_cross, test_wise_eval)


def get_model_class(model_str):
    if model_str == 'gramio':
        model = Gramio
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model

def get_dataset(dataset_name: str, p_fault=0., p_noise=0., masked_s=None, connectivity=None, 
                spatial_shift=False, order=0, node_features='CC', test_months=[5],
                years = [], include_exog=False, exog='traffic', synth_params=None):
    if dataset_name == 'synthetic':
        return add_missing_sensors_cross(CrossGPVARDataset(include_exog=include_exog, **synth_params),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  masked_sensors=masked_s,
                                  connect=connectivity,
                                  spatial_shift=spatial_shift,
                                  order=order,
                                  node_features=node_features)
    if dataset_name == 'carb_sf':
        return add_missing_sensors_cross(AirCross(root='data/CARB-SF', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)
    if dataset_name == 'carb_la':
        return add_missing_sensors_cross(AirCross(root='data/CARB-LA', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)
    if dataset_name == 'madrid':
        return add_missing_sensors_cross(AirCross(root='data/MADRID', test_months=test_months,
                                                  years=years, include_exog=include_exog, exog=exog),
                                        p_fault=p_fault,
                                        p_noise=p_noise,
                                        min_seq=12,
                                        max_seq=12 * 4,
                                        masked_sensors=masked_s,
                                        connect=connectivity,
                                        spatial_shift=spatial_shift,
                                        order=order,
                                        node_features=node_features)

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")

def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    torch.set_float32_matmul_precision('high')
    # Load configuration
    
    seed_everything(cfg.seed, workers=True)

    dataset, masked_sensors = get_dataset(cfg.dataset.name,
                            p_fault=cfg.dataset.get('p_fault'),
                            p_noise=cfg.dataset.get('p_noise'),
                            masked_s=cfg.dataset.get('masked_sensors'),
                            connectivity=cfg.dataset.get('connectivity'),
                            spatial_shift=cfg.dataset.get('spatial_shift'),
                            order=cfg.dataset.get('order'),
                            node_features=cfg.dataset.get('node_features'),
                            test_months=cfg.dataset.get('test_months', (3, 6, 9, 12)),
                            years=cfg.dataset.get('years', []),
                            include_exog=cfg.dataset.get('include_exog', False),
                            exog=cfg.dataset.get('exog', 'traffic'),
                            synth_params=cfg.dataset.get('synth_params', None))

    print(f'Masked sensors: {masked_sensors}')

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity, layout='dense')

    # instantiate dataset
    if cfg.dataset.get('include_exog', False):
        covariates = {'modality': dataset.modality}
        torch_dataset = CrossSpatioTemporalDataset(target=dataset.dataframe(),
                                                    mask=dataset.training_mask,
                                                    eval_mask=dataset.eval_mask,
                                                    covariates=covariates,
                                                    transform=MaskInput(),
                                                    connectivity=adj,
                                                    window=cfg.window,
                                                    stride=cfg.stride)
    else:
        torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                        mask=dataset.training_mask,
                                        eval_mask=dataset.eval_mask,
                                        transform=MaskInput(),
                                        connectivity=adj,
                                        window=cfg.window,
                                        stride=cfg.stride)

    if cfg.scaler == 'StandardScaler':
        scalers = {'target': StandardScaler(axis=(0, 1))}
    elif cfg.scaler == 'StandardScalerSplit':
        scalers = {'target': StandardScalerSplit(axis=(0, 1), split=dataset.air_max_nodes)}
    else:
        raise ValueError(f"Scaler {cfg.scaler} not available in this setting.")

    val_len = cfg.dataset.splitting.get('val_len')
    test_len = cfg.dataset.splitting.get('test_len')

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        generator=g)
    dm.setup(stage='fit')

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    model_cls.filter_model_args_(model_kwargs)

    loss_fn = torch_metrics.MaskedMAE()
    model_kwargs.update(cfg.model.hparams)

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE()
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    imputer = GramioFiller(model_class=model_cls,
                        model_kwargs=model_kwargs,
                        optim_class=getattr(torch.optim, cfg.optimizer.name),
                        optim_kwargs=dict(cfg.optimizer.hparams),
                        loss_fn=loss_fn,
                        scaled_target=cfg.scale_target,
                        metrics=log_metrics,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        gradient_clip_val=cfg.grad_clip_val,
                        gradient_clip_algorithm=cfg.grad_clip_alg,
                        known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                        sampling = cfg.model.hparams.sampling,
                        **cfg.model.regs)

    ########################################
    # logging options                      #
    ########################################

    if 'wandb' in cfg:
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.wandb.offline,
                                 project=cfg.wandb.project)
    elif cfg.logger == 'tensorboard':
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir,
                                       name='tensorboard')
    else: 
        exp_logger = None

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.device,
        callbacks=[early_stop_callback, checkpoint_callback],
        detect_anomaly=False)
    trainer.fit(imputer, datamodule=dm, ckpt_path=cfg.call_path)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    
    res = test_wise_eval(y_hat, y_true, mask, 
                        known_nodes=[i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                        adj=adj[:dataset.air_max_nodes, :dataset.air_max_nodes],
                        mode='test',
                        num_groups=cfg.num_groups,
                        features=y_true)

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    
    res.update(test_wise_eval(y_hat, y_true, mask, 
                known_nodes=[i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                adj=adj[:dataset.air_max_nodes, :dataset.air_max_nodes], mode='val', 
                num_groups=cfg.num_groups, features=y_true))
    
    res.update(
        dict(model=cfg.model.name,
             db=cfg.dataset.name,
             seed=cfg.seed,
             spatial=cfg.dataset.spatial_shift,
             node_f=cfg.dataset.node_features)
    )
    return res

if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)
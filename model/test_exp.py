import argparse
import torch
from gramio import Gramio
from gramio_filler import GramioFiller
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from tsl.data import ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy
from utils import (AirCross, CrossSpatioTemporalDataset, StandardScalerSplit, SpatioTemporalDataModule,
                   CrossGPVARDataset, add_missing_sensors_cross, test_wise_eval)


def get_dataset(dataset_name: str, p_fault=0., p_noise=0., masked_s=None, connectivity=None, 
                spatial_shift=False, order=0, node_features='CC', test_months=[5],
                years = [], include_exog=False, exog='traffic', synth_params=None):
    if dataset_name == 'syntheticCross':
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
    if dataset_name == 'aircross':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossSF', test_months=test_months,
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
    if dataset_name == 'aircross_la':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossLA', test_months=test_months,
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
    if dataset_name == 'aircross_sp':
        return add_missing_sensors_cross(AirCross(root='data/AirCrossSpain', test_months=test_months,
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

def load_model_and_infer(config_path: str, checkpoint_path: str):
    torch.set_float32_matmul_precision('high')   
    # Load configuration
    cfg = OmegaConf.load(config_path)
    seed_everything(cfg.seed, workers=True)
    
    # Load dataset
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

    adj = dataset.get_connectivity(**cfg.dataset.connectivity, layout='dense')
    
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
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=val_len, test_len=test_len),
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        generator=g)
    dm.setup(stage='test')
    
    # Load model
    model_cls = Gramio
    model_kwargs = dict(adj=adj, input_size=dm.n_channels, output_size=dm.n_channels, horizon=cfg.window)
    model_cls.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.device,
        gradient_clip_val=cfg.get('grad_clip_val', None),
        gradient_clip_algorithm=cfg.get('grad_clip_alg', None),
        logger=False)
    
    imputer = GramioFiller.load_from_checkpoint(checkpoint_path,
                                                model_class=model_cls,
                                                model_kwargs=model_kwargs,
                                                gradient_clip_val=cfg.grad_clip_val,
                                                gradient_clip_algorithm=cfg.grad_clip_alg,
                                                known_set = [i for i in range(dataset.air_max_nodes) if i not in masked_sensors],
                                                **cfg.model.regs)
    imputer.eval()
    
    # Run inference
    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)

    y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('eval_mask', None))
    res = test_wise_eval(y_hat, y_true, mask, 
                            known_nodes=[i for i in range(adj.shape[0]) if i not in masked_sensors],
                            adj=adj,
                            mode='test',
                            num_groups=cfg.num_groups,
                            features=y_true)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model inference with config and checkpoint.')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint.')
    args = parser.parse_args()

    # Assuming this function is defined elsewhere
    results = load_model_and_infer(args.config, args.checkpoint)
    print("Inference results:", results)
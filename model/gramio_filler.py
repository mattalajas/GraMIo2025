import inspect
import warnings
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.utilities import move_data_to_device
from torchmetrics import MetricCollection
from tsl import logger
from tsl.metrics.torch import MaskedMetric
from utils import cmd

warnings.filterwarnings("ignore")

def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]

class Filler(pl.LightningModule):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 known_set=None):
        """
        PL module to implement hole fillers.

        :param model_class: Class of pytorch nn.Module implementing the imputer.
        :param model_kwargs: Model's keyword arguments.
        :param optim_class: Optimizer class.
        :param optim_kwargs: Optimizer's keyword arguments.
        :param loss_fn: Loss function used for training.
        :param scaled_target: Whether to scale target before computing loss using batch processing information.
        :param metrics: Dictionary of type {'metric1_name':metric1_fn, 'metric2_name':metric2_fn ...}.
        :param scheduler_class: Scheduler class.
        :param scheduler_kwargs: Scheduler's keyword arguments.
        """
        super(Filler, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn'], logger=False)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.automatic_optimization = False
        self.known_set = known_set

        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        self.scaled_target = scaled_target

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)
        # instantiate model
        self.model = self.model_cls(**self.model_kwargs)

    def reset_model(self):
        self.model = self.model_cls(**self.model_kwargs)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)\
    
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_mean/{name}', param.grad.mean(), on_step=True)
                self.log(f'grad_max/{name}', param.grad.max(), on_step=True)
                self.log(f'grad_min/{name}', param.grad.min(), on_step=True)      
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=True)      

    def collate_prediction_outputs(self, outputs):
        """
        Collate the outputs of the :meth:`predict_step` method.

        Args:
            outputs: Collated outputs of the :meth:`predict_step` method.

        Returns:
            The collated outputs.
        """
        # iterate over results
        processed_res = dict()
        keys = set()
        # iterate over outputs for each batch
        for res in outputs:
            if res:
                for k, v in res.items():
                    if k in keys:
                        processed_res[k].append(v)
                    else:
                        processed_res[k] = [v]
                    keys.add(k)
        # concatenate results
        for k, v in processed_res.items():
            processed_res[k] = torch.cat(v, 0)
        return processed_res
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        batch_data["training"] = False

        modality = batch_data.pop("modality", None)
        split = torch.where(modality>0)[0][0]

        eval_mask = batch_data.pop('eval_mask', None)[:, :, :split]
        known_set = torch.tensor(self.known_set).to(dtype=int)
        unknown_set = torch.tensor([i for i in range(eval_mask.shape[2]) if i not in known_set]).to(dtype=int)
        arrange = torch.cat((known_set, unknown_set))
        reverse = torch.empty_like(arrange)
        reverse[arrange] = torch.arange(len(arrange)).to(arrange.device)
        arrange = arrange.detach().cpu().numpy().tolist()
        known_set = known_set.detach().cpu().numpy().tolist()

        if known_set == []:
            return None
        
        batch_data["sub_entry_num"] = len(unknown_set)
        batch_data["masked_set"] = unknown_set.detach().cpu().numpy().tolist()
        batch_data["known_set"] = known_set
        batch_data["x_exog"] = batch_data["x"][:, :, split:]

        x = batch_data["x"][:, :, :split]
        batch_data["x"] = x[:, :, known_set, :]
        batch_data["split"] = split

        mask = batch_data["mask"][:, :, :split]
        mask = mask[:, :, arrange, :]
        batch_data["mask"] = mask

        # Extract mask and target
        y = batch_data.pop('y')[:, :, :split]
        batch_data.pop("edge_index", None)

        # Compute outputs and rescale
        res, _, _ = self.predict_batch(batch, split, preprocess=False, postprocess=True)
        finpreds, finsim = res[0], res[1]
        finpreds = finpreds[:, :, reverse, :]
        mask = mask[:, :, reverse, :]
        
        output = dict(y=y,
                      y_hat=finpreds,
                      mask=mask,
                      eval_mask=eval_mask,
                      finsim=finsim)
        return output

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='train_')
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='val_', compute_groups=False)
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='test_')

    def _preprocess(self, data, split, batch_preprocessing):
        """
        Perform preprocessing of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: preprocessed data
        """
        for key, trans in batch_preprocessing.items():
            if key in data:
                data[key] = trans.transform(data[key], split)
        return data

    def _postprocess(self, data, split, batch_preprocessing):
        """
        Perform postprocessing (inverse transform) of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to trasform
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: inverse transformed data
        """
        trans = batch_preprocessing.get('y')
        if trans is not None:
            data = trans.inverse_transform(data, split)
        return data

    def predict_batch(self, batch, split=None, preprocess=False, postprocess=True, return_target=False):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop('x')
            x = self._preprocess(x, x.shape[2], batch_preprocessing)
            y_hat = self.forward(x, **batch_data)
        else:
            y_hat = self.forward(**batch_data)
        # Rescale outputs
        if postprocess:
            y_pred = y_hat[0]
            y_pred = self._postprocess(y_pred, y_pred.shape[2], batch_preprocessing)
            y_hat = [y_pred, y_hat[1]]
        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, y_hat, mask
        return y_hat, None, None

    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []
        for batch in loader:
            batch = move_data_to_device(batch, self.device)
            batch_data, batch_preprocessing = self._unpack_batch(batch)

            modality = batch_data.pop("modality", None)
            split = torch.where(modality>0)[0][0]

            eval_mask = batch_data.pop('eval_mask', None)[:, :, :split]
            known_set = torch.tensor(self.known_set).to(dtype=int)
            unknown_set = torch.tensor([i for i in range(eval_mask.shape[2]) if i not in known_set]).to(dtype=int)
            arrange = torch.cat((known_set, unknown_set))
            reverse = torch.empty_like(arrange)
            reverse[arrange] = torch.arange(len(arrange)).to(arrange.device)
            arrange = arrange.detach().cpu().numpy().tolist()
            known_set = known_set.detach().cpu().numpy().tolist()
            # Extract mask and target

            batch_data["sub_entry_num"] = len(unknown_set)
            batch_data["masked_set"] = unknown_set.detach().cpu().numpy().tolist()
            batch_data["known_set"] = known_set
            x = batch_data["x"][:, :, :split]
            batch_data["x"] = x[:, :, known_set, :]
            mask = batch_data["mask"][:, :, :split]
            mask = mask[:, :, arrange, :]
            batch_data["mask"] = mask

            # Extract mask and target
            y = batch_data.pop('y')[:, :, :split]
            batch_data.pop("edge_index", None)

            y_hat, _, _ = self.predict_batch(batch, split, preprocess=preprocess, postprocess=postprocess)
            y_hat = y_hat[0]

            if isinstance(y_hat, (list, tuple)):
                y_hat = y_hat[0]

            targets.append(y)
            imputations.append(y_hat)
            masks.append(eval_mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """
        batch_preprocessing = batch.get('transform')
        return batch, batch_preprocessing

    def on_train_epoch_start(self) -> None:
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg

class GramioFiller(Filler):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn=None,
                 scaled_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 inductive=True,
                 gradient_clip_val=None,
                 gradient_clip_algorithm=None,
                 known_set=None,
                 sampling ='partition',
                 y1 = 1,
                 y2 = 1):
        super(GramioFiller, self).__init__(model_class=model_class,
                                                  model_kwargs=model_kwargs,
                                                  optim_class=optim_class,
                                                  optim_kwargs=optim_kwargs,
                                                  loss_fn=loss_fn,
                                                  scaled_target=scaled_target,
                                                  metrics=metrics,
                                                  scheduler_class=scheduler_class,
                                                  scheduler_kwargs=scheduler_kwargs,
                                                  known_set=known_set)

        self.known_set = known_set
        self.inductive = inductive
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.y1 = y1
        self.y2 = y2
        self.sampling = sampling
    
    def load_model(self, filename: str):
        """Load model's weights from checkpoint at :attr:`filename`.

        Differently from
        :meth:`~pytorch_lightning.core.LightningModule.load_from_checkpoint`,
        this method allows to load the state_dict also for models instantiated
        outside the predictor, without checking that hyperparameters of the
        checkpoint's model are the same of the predictor's model.
        """
        storage = torch.load(filename, lambda storage, loc: storage, weights_only=False)
        # if predictor.model has been instantiated inside predictor
        if self.model_cls is not None:
            model_cls = storage['hyper_parameters']['model_class']
            model_kwargs = storage['hyper_parameters']['model_kwargs']
            # check model class and hyperparameters are the same
            assert model_cls == self.model_cls
        else:
            logger.warning("Predictor with already instantiated model is "
                           f"loading a state_dict from {filename}. Cannot "
                           " check if model hyperparameters are the same.")
        self.load_state_dict(storage['state_dict'])
    
    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=True,
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False,
                 **kwargs)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        opt1 = self.optimizers()
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        batch_data.pop("eval_mask")
        batch_data.pop("edge_index", None)

        modality = batch_data.pop("modality", None)
        split = torch.where(modality>0)[0][0]

        y = batch_data.pop("y")
        y = y[:, :, :split]
        x = batch_data["x"][:, :, :split]
        mask = batch_data["mask"][:, :, :split]
        mask_r = rearrange(mask, "b s n 1 -> (b s) n")
        mask_sum = mask_r.sum(0)  # n

        if self.known_set is None:
            known_set = torch.where(mask_sum > 0)[0].detach().cpu().numpy().tolist()
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio
        else:
            known_set = self.known_set
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio
        if self.sampling == 'half':
            self.ratio/=2

        batch_data["known_set"] = known_set

        sub_entry_num = 0
        batch_data["reset"] = self.inductive

        # Create randomised model here
        cur_entry_num = mask.size(2)

        if self.sampling != 'empty':
            dynamic_ratio = self.ratio + 0.1 * np.random.random()  # ratio + 0.1
            aug_entry_num = max(int(cur_entry_num / dynamic_ratio), cur_entry_num + 1)
            sub_entry_num = aug_entry_num - cur_entry_num  # n2 - n1

        train_ratio = (1 - self.ratio) + 0.1 * np.random.random() 
        trn_entry_num = min(max(int(train_ratio * cur_entry_num), 1), len(known_set)//2)

        # assert sub_entry_num > 0, "The augmented data should have more entries than original data."
        self.sub_entry_num = sub_entry_num

        arrange = torch.randperm(len(known_set))
        t_set = torch.tensor(known_set)
        masked_indx = t_set[arrange[-trn_entry_num:]].numpy().tolist()
        seened_indx = t_set[arrange[:-trn_entry_num]].numpy().tolist()
        full = seened_indx + masked_indx

        # Arrange the indexes to b s (seen, masked) d
        x = x[:, :, seened_indx, :]
        y = y[:, :, full, :]
        mask = mask[:, :, full, :]
        b, s, n, d = mask.size()

        if self.inductive:
            sub_entry = torch.zeros(b, s, sub_entry_num, d).to(x.device)
            mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
            y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

        # Mask the training masks too
        batch_data["seened_set"] = seened_indx
        batch_data["masked_set"] = masked_indx
        batch_data["x_exog"] = batch_data["x"][:, :, split:]

        eval_mask = mask  # eval_mask = mask, during training
        eval_mask[:, :, :len(seened_indx)] = 0.

        batch_data["x"] = x  # b s seen d
        batch_data["mask"] = mask  # b s n' 1
        batch_data["sub_entry_num"] = sub_entry_num  # number
        batch_data["training"] = True
        batch_data["split"] = split

        # Compute predictions and compute loss
        res, _, _ = self.predict_batch(batch, preprocess=False, postprocess=False)
        finpreds, fincross, finsim = res[0], res[1], res[2]

        b = x.shape[0]
        if self.scaled_target:
            target = batch.transform['y'].transform(y, split)
        else:
            target = y
            finpreds = self._postprocess(finpreds, finpreds.shape[2], batch_preprocessing)
            fincross = self._postprocess(fincross, fincross.shape[2], batch_preprocessing)

        opt1.zero_grad()

        # Cross Loss
        if self.y1 != 0:
            cross_loss = self.loss_fn(fincross, target, eval_mask.bool())
        else:
            cross_loss = 0

        if self.y2 != 0:
            cmds = torch.tensor([]).to(x.device)
            for i in range(finsim[0].shape[0]):
                    inv_emb_air = rearrange(finsim[0][i], 'b t d -> (b t) d')
                    inv_emb_cro = rearrange(finsim[1][i], 'b t d -> (b t) d')

                    og_air = inv_emb_air.size(0) // (b)
                    og_cro = inv_emb_cro.size(0) // (b)

                    batches = torch.arange(0, b).to(device=x.device)
                    air_batch = torch.repeat_interleave(batches, repeats=(og_air))
                    cro_batch = torch.repeat_interleave(batches, repeats=(og_cro))

                    cmds = torch.cat([cmds, torch.clamp(cmd(inv_emb_air, inv_emb_cro, \
                                                            air_batch, cro_batch, n_moments=3).mean(), min=0).unsqueeze(0)])

            cmd_loss = cmds.mean()
        else:
            cmd_loss = 0

        main_loss = self.loss_fn(finpreds, target, eval_mask.bool()) 
        loss = main_loss + self.y1 * cross_loss + self.y2 * cmd_loss
        
        self.manual_backward(loss)

        if self.gradient_clip_algorithm and self.gradient_clip_val:
            self.clip_gradients(opt1, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.gradient_clip_algorithm)

        opt1.step()

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(finpreds, finpreds.shape[2], batch_preprocessing)
        else:
            imputation = finpreds

        self.log('Main loss', 
                 main_loss,
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)

        self.log('Reconstruction Loss', 
                 cmd_loss,
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)
    
        self.log('IRM error', 
                 cross_loss,                  
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False)

        # Store every randomised graphs here
        self.train_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        batch_data["training"] = False
        batch_data.pop("edge_index", None)

        modality = batch_data.pop("modality", None)
        split = torch.where(modality>0)[0][0]

        y = batch_data.pop("y")
        y = y[:, :, :split]

        if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            mask = batch_data["mask"][:, :, :split]
            mask = rearrange(mask, "b s n 1 -> (b s) n")
            mask_sum = mask.sum(0)  # n
            known_set = torch.where(mask_sum > 0)[0].detach().cpu().numpy().tolist()
        else:
            known_set = self.known_set

        batch_data["x_exog"] = batch_data["x"][:, :, split:]
        x = batch_data["x"][:, :, :split]
        mask = batch_data["mask"][:, :, :split]
        eval_mask = batch_data.pop("eval_mask")[:, :, :split]

        unknown_set = [i for i in range(mask.shape[2]) if i not in known_set]
        full = known_set + unknown_set

        batch_data["x"] = x[:, :, known_set, :]
        batch_data["split"] = split
        batch_data["sub_entry_num"] = len(unknown_set)
        batch_data["known_set"] = known_set
        batch_data["masked_set"] = unknown_set
        
        mask = mask[:, :, full, :]
        eval_mask = eval_mask[:, :, full, :]
        y = y[:, :, full, :]
        batch_data["mask"] = mask

        # Compute predictions and compute loss
        res, _, _ = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation = res[0]

        if self.scaled_target:
            target = batch.transform['y'].transform(y, split)
        else:
            target= y
            imputation = self._postprocess(imputation, imputation.shape[2], batch_preprocessing)
            
        val_loss = self.loss_fn(imputation, target, eval_mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, imputation.shape[2], batch_preprocessing)

        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        batch_data["training"] = False
        batch_data.pop("edge_index", None)

        modality = batch_data.pop("modality", None)
        split = torch.where(modality>0)[0][0]
        
        mask = batch_data["mask"][:, :, :split]
        # mask_sum = mask.sum(0)  # n
        known_set = torch.tensor(self.known_set).to(dtype=int)
        unknown_set = torch.tensor([i for i in range(mask.shape[2]) if i not in known_set]).to(dtype=int)
        arrange = torch.cat((known_set, unknown_set))
        reverse = torch.empty_like(arrange)
        reverse[arrange] = torch.arange(len(arrange)).to(arrange.device)
        arrange = arrange.detach().cpu().numpy().tolist()
        known_set = known_set.detach().cpu().numpy().tolist()

        if known_set == []:
            return None

        batch_data["sub_entry_num"] = len(unknown_set)
        batch_data["known_set"] = known_set
        batch_data["x_exog"] = batch_data["x"][:, :, split:]

        x = batch_data["x"][:, :, :split]
        batch_data["x"] = x[:, :, known_set, :]
        batch_data["masked_set"] = unknown_set.detach().cpu().numpy().tolist()

        mask = batch_data["mask"][:, :, :split]
        mask = mask[:, :, arrange, :]
        batch_data["mask"] = mask
        batch_data["split"] = split

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)[:, :, :split]
        y = batch_data.pop('y')[:, :, :split]

        # Compute outputs and rescale
        res, _, _ = self.predict_batch(batch, split, preprocess=False, postprocess=True)
        imputation = res[0]

        imputation = imputation[:, :, reverse, :]
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss
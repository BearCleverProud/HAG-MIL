import argparse
import os
import yaml

# internal imports
from datasets.dataset_module import PLDataModule
from models.model import IATModel

# pytorch imports
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, AUROC

# pytorch lightning imports
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule, Trainer, utilities
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# loss function imports
from topk.svm import SmoothTop1SVM

def expand_indices(indices):
    expanded = []
    for index in indices:
        expanded.append(index.item() * 4)
        expanded.append(index.item() * 4 + 1)
        expanded.append(index.item() * 4 + 2)
        expanded.append(index.item() * 4 + 3)
    expanded = torch.tensor(expanded).long().to(indices.device)
    return expanded

class PLModule(LightningModule):
    def __init__(self, config_dict):
        super(PLModule, self).__init__()
        self.config_dict = config_dict
        self.automatic_optimization = False

        if config_dict['model_arguments']['inst_loss'] == 'svm':
            instance_loss_fn = SmoothTop1SVM(n_classes=config_dict['model_arguments']['n_classes'])
        elif config_dict['model_arguments']['inst_loss'] == 'ce':
            instance_loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        if config_dict['model_arguments']['bag_loss'] == 'ce':
            self.criterion_level_2 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_level_1 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_level_0 = nn.CrossEntropyLoss()
        elif config_dict['model_arguments']['bag_loss'] == 'svm':
            self.criterion = SmoothTop1SVM(n_classes=config_dict['model_arguments']['n_classes'])
        else:
            raise NotImplementedError
        
        self.model0 = IATModel(config_dict['model_arguments'], instance_loss_fn=instance_loss_fn)
        self.model1 = IATModel(config_dict['model_arguments'], instance_loss_fn=instance_loss_fn)
        self.model2 = IATModel(config_dict['model_arguments'], instance_loss_fn=instance_loss_fn)
        
        self.acc = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.auc = AUROC(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.f1 = F1Score(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
    
    def training_step(self, batch, batch_index):
        x2, x1, x0, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        x2, x1, x0, y = batch
        logits, Y_prob, _, A_raw, results_dict = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss = config_dict['model_arguments']['bag_weight'] * self.criterion_level_2(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        indices_level_1 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['t_patch']]
        expanded_indices_level_1 = expand_indices(indices_level_1)
        logits, Y_prob, _, A_raw, results_dict = self.model1(x1[expanded_indices_level_1], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss = config_dict['model_arguments']['bag_weight'] * self.criterion_level_1(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        indices_level_0 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['t_patch']]
        indices_level_0 = expanded_indices_level_1[indices_level_0]
        expanded_indices_level_0 = expand_indices(indices_level_0)
        logits, Y_prob, _, A_raw, results_dict = self.model0(x0[expanded_indices_level_0], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
        self.manual_backward(loss)
        opt.step()

        self.log('t_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1, logger=True)
        self.log('t_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}
    
    def epoch_end_log(self, outputs, prefix):
        
        assert prefix == 't_' or prefix == 'v_' or prefix == 'test_'

        y = torch.cat([x["true"] for x in outputs])
        preds = torch.cat([x['pred'] for x in outputs])
        auc = self.auc(preds, y)
        f1 = self.f1(preds, y)
        self.log(prefix+'auc', auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log(prefix+'f1', f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)

    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 't_')

    def validation_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'v_')
    
    def test_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'test_')

    def validation_step(self, batch, batch_index):
        x2, x1, x0, y = batch
        
        logits, Y_prob, _, A_raw, results_dict = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        
        indices_level_1 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        expanded_indices_level_1 = expand_indices(indices_level_1)
        logits, Y_prob, _, A_raw, results_dict = self.model1(x1[expanded_indices_level_1], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        
        indices_level_0 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        indices_level_0 = expanded_indices_level_1[indices_level_0]
        expanded_indices_level_0 = expand_indices(indices_level_0)
        logits, Y_prob, _, A_raw, results_dict = self.model0(x0[expanded_indices_level_0], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            
        self.log('v_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log('v_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}

    def test_step(self, batch, batch_index):
        x2, x1, x0, y = batch
        
        logits, Y_prob, _, A_raw, results_dict = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])

        indices_level_1 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        expanded_indices_level_1 = expand_indices(indices_level_1)
        logits, Y_prob, _, A_raw, results_dict = self.model1(x1[expanded_indices_level_1], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        
        indices_level_0 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        indices_level_0 = expanded_indices_level_1[indices_level_0]
        expanded_indices_level_0 = expand_indices(indices_level_0)
        logits, Y_prob, _, A_raw, results_dict = self.model0(x0[expanded_indices_level_0], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']

        self.log('test_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.config_dict['hyperparams_arguments']['lr'], 
                                weight_decay=self.config_dict['hyperparams_arguments']['reg'])
        return [optimizer]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Configurations for IAT Training')
    parser.add_argument('--config-path', type=str, default='configs/camelyon.yaml',
                        help='Configuration path used in training the model')
    args = parser.parse_args()

    config_dict = yaml.safe_load(open(args.config_path, 'r'))

    utilities.seed.seed_everything(config_dict['hyperparams_arguments']['seed'])

    model_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                config_dict['storage_arguments']['task'],
                                config_dict['storage_arguments']['model_dir'],
                                config_dict['storage_arguments']['exp_code'])
    log_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                config_dict['storage_arguments']['task'],
                                config_dict['storage_arguments']['log_dir'])
                            
    logger = pl_loggers.TensorBoardLogger(log_dir, config_dict['storage_arguments']['exp_code'])
    model = PLModule(config_dict)
    all_datasets = PLDataModule(config_dict)

    model_checkpoint = ModelCheckpoint(monitor='v_loss', 
                                        dirpath=os.path.join(model_dir,f'version_{logger.version}'),
                                        filename=config_dict['model_arguments']['model_type'] + '-{epoch:02d}-{v_loss:.3f}',
                                        save_top_k=config_dict['storage_arguments']['save_top_k'], mode='min'
                                        )
    early_stopping = EarlyStopping(monitor="v_loss", min_delta=0.00, 
                                patience=config_dict['hyperparams_arguments']['early_stop'], verbose=False, mode="min")

    trainer = Trainer(
                    gpus=config_dict['hyperparams_arguments']['gpus'],
                    max_epochs=config_dict['hyperparams_arguments']['max_epoch'],
                    callbacks=[model_checkpoint, early_stopping],
                    logger=logger,
                    strategy=DDPPlugin(find_unused_parameters=True),
                    log_every_n_steps=10,
                    )

    trainer.fit(model, all_datasets)
    trainer.test(ckpt_path="best",datamodule=all_datasets)

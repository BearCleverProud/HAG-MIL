
from main import PLModule, expand_indices
from datasets.dataset_module import PLDataModule
import yaml
from pytorch_lightning import utilities
import torch
from torchmetrics import AUROC, F1Score, Accuracy
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Configurations for IAT Training')
parser.add_argument('--config-path', type=str, default='configs/camelyon.yaml',
                    help='Configuration path used in training the model')
parser.add_argument('--ckpt-path', type=str,
                    help='Path to the checkpoint')
args = parser.parse_args()
config_dict = yaml.safe_load(open(args.config_path, 'r'))

utilities.seed.seed_everything(config_dict['hyperparams_arguments']['seed'])
model = PLModule.load_from_checkpoint(args.ckpt_path, config_dict=config_dict).to('cuda:0')
auc = AUROC(task='multiclass', num_classes=config_dict['model_arguments']['n_classes']).to('cuda:0')
f1 = F1Score(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro').to('cuda:0')
acc = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes']).to('cuda:0')
tnt_datasets = PLDataModule(config_dict)

preds = []
ys = []
total = 0
correct = 0
for sample in tqdm(tnt_datasets.test_dataloader()):
    with torch.no_grad():
        x2, x1, x0, y = sample
        x2 = x2.to('cuda:0')
        x1 = x1.to('cuda:0')
        x0 = x0.to('cuda:0')
        y = y.to('cuda:0')
        
        logits, Y_prob, Y_hat, A_raw, results_dict = model.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        
        indices_level_1 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        expanded_indices_level_1 = expand_indices(indices_level_1)
        logits, Y_prob, Y_hat, A_raw, results_dict = model.model1(x1[expanded_indices_level_1], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])

        indices_level_0 = torch.argsort(A_raw, descending=True).squeeze().detach()[:config_dict['hyperparams_arguments']['v_patch']]
        indices_level_0 = expanded_indices_level_1[indices_level_0]
        expanded_indices_level_0 = expand_indices(indices_level_0)
        logits, Y_prob, Y_hat, A_raw, results_dict = model.model0(x0[expanded_indices_level_0], label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        preds.append(Y_prob.detach().cpu())
        if torch.argmax(Y_prob).item() == y.item():
            correct += 1
        total += 1
        ys.append(y)

preds = torch.cat(preds, dim=0).to('cuda:0')
ys = torch.cat(ys, dim=0).to('cuda:0')
auc_score = auc(preds, ys)
print(f'AUC: {auc_score}')
print(f'Accuracy: {acc(preds, ys)}')
print(f'F1 Score: {f1(preds, ys)}')

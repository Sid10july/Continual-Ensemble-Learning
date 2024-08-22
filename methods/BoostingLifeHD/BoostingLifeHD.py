from __future__ import print_function

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_tsne, plot_confusion_matrix
from methods.LifeHD.LifeHD import LifeHD

class BoostingLifeHD(LifeHD):
    def __init__(self, opt, train_loader, val_loader, num_classes, model, logger, device, num_learners=5, lr=1e-3):
        super(BoostingLifeHD, self).__init__(opt, train_loader, val_loader, num_classes, model, logger, device)
        self.num_learners = num_learners
        self.lr = lr
        self.ensemble = [LifeHD(opt, train_loader, val_loader, num_classes, model, logger, device) for _ in range(num_learners)]
        self.sample_weights = torch.ones(len(train_loader.dataset)) / len(train_loader.dataset)

    def _update_sample_weights(self, predictions, targets, batch_indices):
        errors = predictions != targets
        weighted_error = torch.sum(self.sample_weights[batch_indices] * errors) / torch.sum(self.sample_weights[batch_indices])
        learner_weight = self.lr * torch.log((1 - weighted_error) / weighted_error)

        new_weights = self.sample_weights[batch_indices] * torch.exp(learner_weight * errors)
        new_weights /= torch.sum(new_weights)
        self.sample_weights[batch_indices] = new_weights

    def start(self):
        for model in self.ensemble:
            model.start()

    def warmup(self):
        for model in self.ensemble:
            model.warmup()

    def train(self, epoch):
        for model_idx, model in enumerate(self.ensemble):
            # Update DataLoader with current sample weights
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
            weighted_train_loader = DataLoader(self.train_loader.dataset, sampler=sampler, batch_size=self.train_loader.batch_size)
        
            # Train the model with the weighted data
            model.train(epoch, train_loader=weighted_train_loader)
        
            # Get predictions and update sample weights
            all_predictions = []
            all_targets = []
            for images, labels in tqdm(weighted_train_loader, desc=f"Training Model {model_idx}"):
                images = images.to(model.device)
                outputs, _ = model.model(images)
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())

            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            self._update_sample_weights(all_predictions, all_targets, np.arange(len(all_predictions)))

    def validate(self, epoch, loader_idx, plot, mode):
        all_scores = []
        all_test_labels = []
        for model in self.ensemble:
            with torch.no_grad():
                scores = []
                for images, labels in tqdm(model.val_loader, desc="Testing"):
                    images = images.to(model.device)
                    outputs, _ = model.model(images)
                    scores.append(outputs.detach().cpu().numpy())
                all_scores.append(np.array(scores))
                all_test_labels.append(np.array([label.cpu().numpy() for _, label in model.val_loader]))

        
        all_scores = np.array(all_scores)
        learner_weights = np.array([model.sample_weight for model in self.ensemble])
        weighted_scores = np.average(all_scores, axis=0, weights=learner_weights)
        majority_vote = np.argmax(weighted_scores, axis=-1)
        # averaged_scores = np.mean(np.array(all_scores), axis=0)
        # majority_vote = np.argmax(averaged_scores, axis=-1)
        
        
        flat_test_labels = np.concatenate(all_test_labels[0])
        
        self._log_metrics(majority_vote, flat_test_labels, epoch, loader_idx, plot, mode)

    def _log_metrics(self, pred_labels, test_labels, epoch, loader_idx, plot, mode):
        acc, purity, cm = eval_acc(test_labels, pred_labels)
        print(f'Acc: {acc}, purity: {purity}')
        nmi = eval_nmi(test_labels, pred_labels)
        print(f'NMI: {nmi}')
        ri = eval_ri(test_labels, pred_labels)
        print(f'RI: {ri}')

        with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
            f.write(f'{epoch},{loader_idx},{acc},{purity},{nmi},{ri},{self.model.cur_classes},{self.trim},{self.merge}\n')

        self.logger.log_value('accuracy', acc, loader_idx)
        self.logger.log_value('purity', purity, loader_idx)
        self.logger.log_value('nmi', nmi, loader_idx)
        self.logger.log_value('ri', ri, loader_idx)
        self.logger.log_value('num of clusters', self.model.cur_classes, loader_idx)

    def add_sample_hv_to_exist_class(self, sample_hv):
        for model in self.ensemble:
            model.add_sample_hv_to_exist_class(sample_hv)

    def merge_clusters(self):
        for model in self.ensemble:
            model.merge_clusters()

    def trim_clusters(self):
        for model in self.ensemble:
            model.trim_clusters()

    def add_sample_hv_to_novel_class(self, sample_hv):
        for model in self.ensemble:
            model.add_sample_hv_to_novel_class(sample_hv)




    def trim_clusters(self):
        for model in self.ensemble:
            model.trim_clusters()

    def add_sample_hv_to_novel_class(self, sample_hv):
        for model in self.ensemble:
            model.add_sample_hv_to_novel_class(sample_hv)
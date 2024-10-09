from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms.functional import rotate
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from tqdm import tqdm
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_tsne, plot_tsne_graph, \
    plot_novelty_detection, plot_confusion_matrix
from methods.LifeHD.LifeHD import LifeHD 
from sklearn.metrics.pairwise import cosine_similarity



class BaggingLifeHD(LifeHD):
    def __init__(self, opt, train_loader, val_loader, num_classes, model, logger, device, num_learners):
        super(BaggingLifeHD, self).__init__(opt, train_loader, val_loader, num_classes, model, logger, device)
        self.num_learners = num_learners
        self.ensemble = [LifeHD(opt, self._bootstrap_sample(train_loader), val_loader, num_classes, model, logger, device)
                         for _ in range(self.num_learners)]
        self.unified_centroids = None

    def _bootstrap_sample(self, data_loader):
        dataset = data_loader.dataset
        sample_size = int(.85 * len(dataset))
        indices = torch.randperm(len(dataset))[:sample_size].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        new_data_loader = DataLoader(subset, batch_size=data_loader.batch_size, 
                                     shuffle=True, num_workers=data_loader.num_workers)
        return new_data_loader

    def start(self):
        for model in self.ensemble:
            model.start()
        final_acc, final_purity = self.validate_bagging('final')
        print(f'Final Bagging Validation Accuracy: {final_acc}, Purity: {final_purity}')


    def validate_bagging(self, mode):
        all_test_labels = []
        all_predictions = []


        for model in self.ensemble:
            with torch.no_grad():
                for images, labels in tqdm(model.val_loader, desc="Collecting Labels and Predictions"):
                    images = images.to(model.device)
                    outputs, _ = model.model(images)
                    predictions = torch.argmax(outputs, dim=-1).cpu().tolist()  
                    all_predictions.append(predictions)  
                    all_test_labels.extend(labels.cpu().tolist())  

        flat_test_labels = np.array(all_test_labels)
        

        all_predictions_flattened = np.array([pred for learner_preds in all_predictions for pred in learner_preds])


        acc, purity = self._log_metrics(all_predictions_flattened, flat_test_labels, mode)
        return acc, purity

    def _log_metrics(self, pred_labels, test_labels, mode):
        acc, purity, cm = eval_acc(test_labels, pred_labels)
        nmi = eval_nmi(test_labels, pred_labels)
        ri = eval_ri(test_labels, pred_labels)

        print(f'Final Validation Accuracy: {acc}, Purity: {purity}, NMI: {nmi}, RI: {ri}')
        with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
            f.write(f'{mode},{acc},{purity},{nmi},{ri}\n')

        self.logger.log_value('accuracy', acc)
        self.logger.log_value('purity', purity)
        self.logger.log_value('nmi', nmi)
        self.logger.log_value('ri', ri)

        return acc, purity



    # def validate_bagging(self, mode):
    #     all_test_labels = []
    #     all_predictions = []

    #     for model in self.ensemble:
    #         with torch.no_grad():
    #             for images, labels in tqdm(model.val_loader, desc="Collecting Labels"):
    #                 images = images.to(model.device)
    #                 outputs, _ = model.model(images)
    #                 predictions = torch.argmax(outputs, dim=-1).cpu().tolist()  
    #                 all_predictions.extend(predictions)  
    #                 all_test_labels.extend(labels.cpu().tolist())  

    #     flat_test_labels = np.array(all_test_labels)
    #     flat_predictions = np.array(all_predictions)

    #     learners_centroids = [model.model.classify_weights.detach().cpu().numpy() for model in self.ensemble]

    #     averaged_centroids = self.average_similar_centroids(learners_centroids, flat_predictions)

    #     all_predictions = []  

    #     for model in self.ensemble:
    #         with torch.no_grad():
    #             for images, labels in tqdm(model.val_loader, desc="Testing"):
    #                 images = images.to(model.device)
    #                 outputs, _ = model.model(images)

    #                 # Compute cosine similarity with the averaged centroids
    #                 cosine_sim = F.normalize(outputs, dim=-1) @ F.normalize(torch.from_numpy(averaged_centroids).to(model.device).float())
    #                 predictions = torch.argmax(cosine_sim, dim=-1).cpu().tolist()
    #                 all_predictions.extend(predictions)

    #     flat_predictions = np.array(all_predictions)

    #     acc, purity = self._log_metrics(flat_predictions, flat_test_labels, mode)
    #     return acc, purity

    # def _log_metrics(self, pred_labels, test_labels, mode):
    #     acc, purity, cm = eval_acc(test_labels, pred_labels)
    #     nmi = eval_nmi(test_labels, pred_labels)
    #     ri = eval_ri(test_labels, pred_labels)

    #     print(f'Final Validation Accuracy: {acc}, Purity: {purity}, NMI: {nmi}, RI: {ri}')
    #     with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
    #         f.write(f'{mode},{acc},{purity},{nmi},{ri}\n')

    #     self.logger.log_value('accuracy', acc)
    #     self.logger.log_value('purity', purity)
    #     self.logger.log_value('nmi', nmi)
    #     self.logger.log_value('ri', ri)

    #     return acc, purity








# These were some methods I used to average
    # def _map_centroid_to_predicted_labels(self, centroids, predictions):
        
    #     D1 = predictions.max() + 1  

    #     centroid_to_pred_label = np.zeros(centroids.shape[0], dtype=int)

        
    #     for i in range(centroids.shape[0]):
            
    #         class_predictions = predictions == i
    #         if np.sum(class_predictions) > 0:
    #             centroid_to_pred_label[i] = np.argmax(np.bincount(predictions[class_predictions]))

    #     return centroid_to_pred_label

    # def average_similar_centroids(self, learners_centroids, all_predictions):
    #     """
    #     Averages centroids from multiple learners that map to the same predicted class.
    #     """
    #     num_learners = len(learners_centroids)
    #     num_classes = max([centroids.shape[0] for centroids in learners_centroids])

    #     s
    #     averaged_centroids = np.zeros((num_classes, learners_centroids[0].shape[1]))

        
    #     centroid_label_mappings = [self._map_centroid_to_predicted_labels(centroids, all_predictions)
    #                                for centroids in learners_centroids]

        
    #     for class_label in range(num_classes):
    #         centroids_to_average = []
    #         for learner_idx in range(num_learners):
    #             matching_centroids = learners_centroids[learner_idx][centroid_label_mappings[learner_idx] == class_label]
    #             centroids_to_average.extend(matching_centroids)

    #         if len(centroids_to_average) > 0:
    #             averaged_centroids[class_label] = np.mean(centroids_to_average, axis=0)

    #     return averaged_centroids

    #SECOND
    #   def average_similar_centroids(self, learners_centroids):
    #     num_learners = len(learners_centroids)
    #     num_classes_per_learner = len(learners_centroids[0])

    #     averaged_centroids = []
    
       
    #     used_centroids_per_learner = [set() for _ in range(num_learners)]
    
       
    #     matched_centroids_log = {learner: [] for learner in range(num_learners)}

    #     for i in range(num_classes_per_learner):
    #         base_centroid = learners_centroids[0][i]
    #         all_similar_centroids = [base_centroid]
    #         matched_centroids_log[0].append(i)  

    #         for j in range(1, num_learners):
    #             other_centroids = learners_centroids[j]
            
                
    #             similarities = cosine_similarity([base_centroid], other_centroids)
            
                
    #             sorted_indices = np.argsort(-similarities[0])  
            
                
    #             for most_similar_index in sorted_indices:
    #                 if most_similar_index not in used_centroids_per_learner[j]:
    #                     most_similar_centroid = other_centroids[most_similar_index]
    #                     all_similar_centroids.append(most_similar_centroid)
    #                     used_centroids_per_learner[j].add(most_similar_index)  
    #                     matched_centroids_log[j].append(most_similar_index)  
    #                     break  
        
            
    #         averaged_centroids.append(np.mean(all_similar_centroids, axis=0))

    #     averaged_centroids = np.array(averaged_centroids)
    
       
    #     print("Matched centroids for each learner before merging:")
    #     for learner, matched_centroids in matched_centroids_log.items():
    #         print(f'Learner {learner}: {matched_centroids}')
    
    #     return averaged_centroids

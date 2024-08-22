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
        sample_size = int(.55 * len(dataset))
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

  
#---------Method 1-----------------

    # def average_similar_centroids(self, learners_centroids):
    #
    #     num_learners = len(learners_centroids)
    #     num_classes_per_learner = len(learners_centroids[0])
    
    #     
    #     averaged_centroids = []

    #     for i in range(num_classes_per_learner):
    #         # Take the i-th centroid from the first learner
    #         base_centroid = learners_centroids[0][i]
    #         all_similar_centroids = [base_centroid]
    #         used_indices = set()

    #         
    #         for j in range(1, num_learners):
    #             other_centroids = learners_centroids[j]
            
    #             
    #             similarities = cosine_similarity([base_centroid], other_centroids)
    #             most_similar_index = np.argmax(similarities)
            
    #             
    #             if (j, most_similar_index) not in used_indices: #this doesn't re-eval if found but in my testing it didn't matter
    #                 most_similar_centroid = other_centroids[most_similar_index]
    #                 all_similar_centroids.append(most_similar_centroid)
    #                 used_indices.add((j, most_similar_index))
                
    #         
    #         averaged_centroids.append(np.mean(all_similar_centroids, axis=0))
    
    #     averaged_centroids = np.array(averaged_centroids)
    
    #     return averaged_centroids

    # def validate_bagging(self, mode):
    #     all_test_labels = []
    #     all_predictions = []

    #     # Collect centroids from all learners
    #     learners_centroids = [model.model.classify_weights.detach().cpu().numpy() for model in self.ensemble]
    
    #     # Average centroids by merging similar ones
    #     averaged_centroids = self.average_similar_centroids(learners_centroids)

    #     # Calculate cosine similarity with the averaged centroids
    #     for model in self.ensemble:
    #         with torch.no_grad():
    #             for images, labels in tqdm(model.val_loader, desc="Testing"):
    #                 images = images.to(model.device)
    #                 outputs, _ = model.model(images)
    #                 all_test_labels.extend(labels.cpu().tolist())
                
    #                 # Calculate cosine similarity with the averaged centroids
    #                 cosine_sim = F.normalize(outputs, dim=-1) @ F.normalize(torch.from_numpy(averaged_centroids).to(model.device))
    #                 predictions = torch.argmax(cosine_sim, dim=-1).cpu().tolist()
    #                 all_predictions.extend(predictions)

    #     # Evaluate with the averaged centroids
    #     flat_test_labels = np.array(all_test_labels)  # Convert to NumPy array
    #     flat_predictions = np.array(all_predictions)  # Convert to NumPy array

    #     acc, purity = self._log_metrics(flat_predictions, flat_test_labels, mode)
    #     return acc, purity


#-------Method 2 K-Means ------------------
    def KMEANSmethod(self, learners_centroids, desired_num_classes):
        """
        Aggressively merge centroids from multiple learners to increase accuracy.

        Args:
            learners_centroids (list of numpy arrays): List where each element is an array of centroids for one learner.
            desired_num_classes (int): The number of classes to reduce the centroids to.

        Returns:
            averaged_centroids (numpy array): The averaged centroids after merging similar ones.
        """
        num_learners = len(learners_centroids)
        num_classes_per_learner = len(learners_centroids[0])
    
        # Initialize arrays to keep track of averaged centroids
        all_centroids = []
    
        # Collect all centroids from all learners
        for centroids in learners_centroids:
            all_centroids.extend(centroids)
    
        all_centroids = np.array(all_centroids)
    
        # Perform KMeans clustering to reduce the number of centroids to desired_num_classes
        kmeans = KMeans(n_clusters=desired_num_classes)
        kmeans.fit(all_centroids)
    
        # Compute the new centroids for each cluster
        averaged_centroids = np.zeros((desired_num_classes, all_centroids.shape[1]))
        for i in range(desired_num_classes):
            cluster_centroids = all_centroids[kmeans.labels_ == i]
            if cluster_centroids.size > 0:
                averaged_centroids[i] = np.mean(cluster_centroids, axis=0)
    
        return averaged_centroids
    

    def validate_bagging(self, mode):
        all_centroids = [model.model.classify_weights.detach().cpu().numpy() for model in self.ensemble]
    
        # Aggressively merge centroids
        num_classes = len(all_centroids[0])
        desired_num_classes = num_classes
        averaged_centroids = self.KMEANS_method(all_centroids, desired_num_classes)
    
        # Calculate cosine similarity with the averaged centroids
        all_predictions = []
        all_test_labels = []
        for model in self.ensemble:
            with torch.no_grad():
                for images, labels in model.val_loader:
                    images = images.to(model.device)
                    outputs, _ = model.model(images)

                    averaged_centroids_tensor = torch.from_numpy(averaged_centroids).float().to(self.device)
                    outputs = outputs.float()
                
                    # Calculate cosine similarity with the averaged centroids
                    normalized_outputs = F.normalize(outputs, dim=-1)
                    normalized_centroids = F.normalize(averaged_centroids_tensor, dim=-1)
                    cosine_sim = normalized_outputs @ normalized_centroids
                    # cosine_sim = F.normalize(outputs, dim=-1) @ F.normalize(torch.from_numpy(averaged_centroids_tensor))
                    predictions = torch.argmax(cosine_sim, dim=-1).cpu().tolist()
                    all_predictions += predictions
                    all_test_labels += labels.cpu().tolist()
    
        # Evaluate with the averaged centroids
        flat_test_labels = np.array(all_test_labels)
        flat_predictions = np.array(all_predictions)

        acc, purity = self._log_metrics(flat_predictions, flat_test_labels, mode)
        return acc, purity


#-----------------------------------------------------

    def _log_metrics(self, pred_labels, test_labels, mode):
        print("test", len(test_labels))
        print("pred", len(pred_labels))
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

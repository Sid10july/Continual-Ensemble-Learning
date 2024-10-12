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
from methods.LifeHD.LifeHD import LifeHD, get_nc_laplacian, get_nc 
from sklearn.metrics.pairwise import cosine_similarity

novelty_detect = []
class_shift = []
VAL_CNT = 10

class BaggingBaseLifeHD(LifeHD):
    def train(self, epoch):
        """Training of one epoch on single-pass of data"""
        """Unsupervised method. Should not use the labels"""
        # Set validation frequency
        val_freq = np.floor(len(self.train_loader) / VAL_CNT).astype('int')
        batchs_per_class = np.floor(len(self.train_loader) / self.num_classes).astype('int')

        with torch.no_grad():
            class_batch_idx = 0  # batch index in the current class
            cur_class = -1

            for idx, (image, label) in enumerate(self.train_loader):

                # Batch Bagging:
                bootstrap_ratio = 1.0
                batch_size = image.size(0)
                bootstrap_size = int(batch_size * bootstrap_ratio)
                indices = torch.randint(0, batch_size, (bootstrap_size,))
                bootstrap_image = image[indices]
                bootstrap_label = label[indices]


                # Online Bagging:
                # bootstrap_image = []
                # bootstrap_label = []
                # for i in range(image.size(0)):
                #     k = np.random.poisson(lam = 1.0)
                #     img = image[i]
                #     lab = label[i]
                #     if k > 0:
                #         for _ in range(k):
                #             bootstrap_image.append(img)
                #             bootstrap_label.append(lab)
                # bootstrap_image = torch.stack(bootstrap_image)
                # bootstrap_label = torch.tensor(bootstrap_label)


                # Validation
                if idx > self.opt.warmup_batches and idx % val_freq == 0:
                    # Trick: trim the clusters that have samples less than 10
                    if idx > self.opt.warmup_batches + 1 and self.opt.merge_mode != 'no_trim':
                        self.trim_clusters()

                    # acc, purity = self.validate(epoch, idx, False, 'before')
                    #################################################
                    # 3. Cluster merging
                    #################################################
                    if self.opt.merge_mode != 'no_merge':
                        pair_simil, class_hvs = self.model.extract_pair_simil(self.mask)  # numpy array
                        thres = self.model.dist_mean[:self.model.cur_classes].mean().cpu().numpy()
                        # thres = thres * self.cur_mask_dim / self.opt.dim
                        # print(self.model.dist_mean[:self.model.cur_classes])
                        # print('thres: ', thres)
                        nc, _, U = get_nc(class_hvs, pair_simil, thres,
                                          idx, self.opt, self.warmup_done)

                        # Merge clusters
                        if self.opt.k_merge_min < nc < self.model.max_classes:
                            self.merge_clusters(U, nc, class_hvs, idx)

                    acc, purity = self.validate(epoch, idx+1, False, 'after')
                    print('Validate stream: [{}][{}/{}]\tacc: {} purity: {}'.format(
                        epoch, idx + 1, len(self.train_loader), acc, purity))
                    sys.stdout.flush()

                # Adjust the mask dimension to lower dimension
                if self.opt.mask_mode == 'adaptive' and idx - self.last_novel > 3:
                    weight_sum = torch.abs(self.model.classify_weights[:self.model.cur_classes].sum(dim=0))
                    sort_idx = torch.argsort(weight_sum, descending=True)
                    self.mask = torch.zeros(self.opt.dim, device=self.device).type(torch.bool)
                    self.mask[sort_idx[:self.opt.mask_dim]] = 1
                    self.cur_mask_dim = self.opt.mask_dim

                if bootstrap_label[0] > cur_class:
                    class_shift.append(idx)
                    class_batch_idx = 0
                    cur_class = bootstrap_label[0]
                
                if self.opt.rotation > 0.0:
                    rot_degrees = self.opt.rotation / batchs_per_class * class_batch_idx
                    bootstrap_image = rotate(bootstrap_image, rot_degrees)

                bootstrap_image = bootstrap_image.to(self.device)
                bootstrap_label = bootstrap_label.to(self.device)
                outputs, sample_hv = self.model(bootstrap_image, self.mask)

                # Check if warmup has ended
                if not self.warmup_done:
                    self.warmup(idx, sample_hv, bootstrap_label)

                else:
                    # Normal session after warmup
                    #################################################
                    # 1. predict the nearest centroid
                    #################################################
                    simil_to_class, pred_class = torch.max(outputs, dim=-1)
                    pred_class_samples = self.model.classify_sample_cnt[pred_class]

                    #print(
                    #    '\n\nidx: {}/{}\ncur_label: {}\nmin_dist: {}\npred_class: {}'.format(
                    #        idx, len(self.train_loader),
                    #        label.cpu().numpy(),
                    #        simil_to_class.cpu().numpy(),

                    assert pred_class_samples.min() > 0, \
                        'Predicted class {} has zero sample!'

                    #################################################
                    # 2. add sample to cluster or novelty detection
                    #################################################
                    # Novelty detection
                    # Compare the new max cosine similarity with the 95-percentile
                    # (given by opt.beta, mean - 3 * standard difference)
                    # in the pred_class's distance distribution
                    simil_threshold = self.model.dist_mean[pred_class] - \
                                        self.opt.beta * self.model.dist_std[pred_class]
                    # simil_threshold = simil_threshold * self.cur_mask_dim / self.opt.dim
                    #print('\tmean {}\n\tstd {}\n\tdist_thres {}'.format(
                    #    self.model.dist_mean[pred_class].cpu().numpy(),
                    #    self.model.dist_std[pred_class].cpu().numpy(),
                    #    simil_threshold.cpu().numpy()))
                    #print('threshold: ', simil_threshold.cpu().numpy())
                    #print('simil to class: ', simil_to_class.cpu().numpy())

                    # To show as a novelty, we require the samples in the existing cluster
                    # is larger than a fixed number (default is 10), so we have sufficient
                    # confidence
                    novel_detect_mask = (simil_to_class < simil_threshold) & \
                          (pred_class_samples > 10)  # (batch_size, D)
                    # print(simil_to_class < simil_threshold)
                    # print(pred_class_samples > 10)
                    # print(novel_detect_mask)

                    # Add the new sample to the predicted class
                    self.add_sample_hv_to_exist_class(sample_hv[~novel_detect_mask], 
                                                      pred_class[~novel_detect_mask], 
                                                      simil_to_class[~novel_detect_mask], 
                                                      idx)

                    # A novelty is detected, need to create new classes
                    if novel_detect_mask.sum() > 0:
                        #print('Novelty detected!')
                        novelty_detect.append(idx)
                        #print('pred class ', pred_class[novel_detect_mask].cpu().numpy())
                        #print('simil to class ', simil_to_class[novel_detect_mask].cpu().numpy())
                        #print('simil threshold ', simil_threshold[novel_detect_mask].cpu().numpy())
                        self.add_sample_hv_to_novel_class(sample_hv[novel_detect_mask], idx)

                        # Revert the mask dim to dim
                        if self.opt.mask_mode == 'adaptive':
                            self.mask = torch.ones(self.opt.dim, device=self.device).type(torch.bool)
                            self.cur_mask_dim = self.opt.dim
                            self.last_novel = idx

                self.model.classify.weight[:] = F.normalize(self.model.classify_weights)

                #print('sample cnt', self.model.classify_sample_cnt[:self.model.cur_classes].cpu().numpy().astype('int'))
                #print('mean', self.model.dist_mean[:self.model.cur_classes].cpu().numpy())
                #print('std', self.model.dist_std[:self.model.cur_classes].cpu().numpy())

                self.logger.log_value('mask_dim', self.cur_mask_dim, idx)

                class_batch_idx += 1

            if self.opt.merge_mode != 'no_trim':
                self.trim_clusters()
            print(self.model.classify_sample_cnt)
            plot_novelty_detection(class_shift, novelty_detect, self.opt.save_folder)


class BaggingLifeHD:
    def __init__(self, opt, train_loader, val_loader,
                 num_classes, model, logger, device, num_learners):
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.model = model
        self.logger = logger
        self.device = device
        self.num_learners = num_learners
        self.ensemble = [BaggingBaseLifeHD(opt, self.train_loader, self.val_loader, self.num_classes, model, logger, device)
                         for _ in range(self.num_learners)]
        self.unified_centroids = None
    
    def start(self):
        # train for one epoch
        time1 = time.time() 
        for model in self.ensemble:
            model.train(1)
        time2 = time.time()
        print('Total time {:.2f}'.format( time2 - time1))

        # final validation
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

# class BaggingLifeHD(LifeHD):
#     def __init__(self, opt, train_loader, val_loader, num_classes, model, logger, device, num_learners):
#         super(BaggingLifeHD, self).__init__(opt, train_loader, val_loader, num_classes, model, logger, device)
#         self.num_learners = num_learners
#         self.ensemble = [LifeHD(opt, self._bootstrap_sample(train_loader), val_loader, num_classes, model, logger, device)
#                          for _ in range(self.num_learners)]
#         self.unified_centroids = None

#     def _bootstrap_sample(self, data_loader):
#         dataset = data_loader.dataset
#         sample_size = int(.85 * len(dataset))
#         indices = torch.randperm(len(dataset))[:sample_size].tolist()
#         subset = torch.utils.data.Subset(dataset, indices)
#         new_data_loader = DataLoader(subset, batch_size=data_loader.batch_size, 
#                                      shuffle=True, num_workers=data_loader.num_workers)
#         return new_data_loader

#     def start(self):
#         for model in self.ensemble:
#             model.start()
#         final_acc, final_purity = self.validate_bagging('final')
#         print(f'Final Bagging Validation Accuracy: {final_acc}, Purity: {final_purity}')


#     def validate_bagging(self, mode):
#         all_test_labels = []
#         all_predictions = []


#         for model in self.ensemble:
#             with torch.no_grad():
#                 for images, labels in tqdm(model.val_loader, desc="Collecting Labels and Predictions"):
#                     images = images.to(model.device)
#                     outputs, _ = model.model(images)
#                     predictions = torch.argmax(outputs, dim=-1).cpu().tolist()  
#                     all_predictions.append(predictions)  
#                     all_test_labels.extend(labels.cpu().tolist())  

#         flat_test_labels = np.array(all_test_labels)
        

#         all_predictions_flattened = np.array([pred for learner_preds in all_predictions for pred in learner_preds])


#         acc, purity = self._log_metrics(all_predictions_flattened, flat_test_labels, mode)
#         return acc, purity

#     def _log_metrics(self, pred_labels, test_labels, mode):
#         acc, purity, cm = eval_acc(test_labels, pred_labels)
#         nmi = eval_nmi(test_labels, pred_labels)
#         ri = eval_ri(test_labels, pred_labels)

#         print(f'Final Validation Accuracy: {acc}, Purity: {purity}, NMI: {nmi}, RI: {ri}')
#         with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
#             f.write(f'{mode},{acc},{purity},{nmi},{ri}\n')

#         self.logger.log_value('accuracy', acc)
#         self.logger.log_value('purity', purity)
#         self.logger.log_value('nmi', nmi)
#         self.logger.log_value('ri', ri)

#         return acc, purity



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

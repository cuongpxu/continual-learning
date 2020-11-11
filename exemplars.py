import abc
import torch
import utils
import copy
import numpy as np
from torch import nn
from torch.nn import functional as F
from data import OnlineExemplarDataset


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

        # Proposed method
        self.online_exemplar_sets = {}  # --> exemplar_set is a dictionary contains an <np.array> of N images
                                        # with shape (N, Ch, H, W) and its corresponding true label
        self.online_memory_budget = 1000
        self.online_classes_so_far = []
        self.online_exemplar_means = {}
        # self.online_exemplar_features = {}

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass


    ####----MANAGING ONLINE EXEMPLAR SETS----####
    def check_full_memory(self):
        current_size = self.get_online_exemplar_size()
        return current_size >= self.online_memory_budget

    def get_online_exemplar_size(self):
        total = 0
        for m in self.online_classes_so_far:
            total += self.online_exemplar_sets[m][0].shape[0]
        return total

    def check_online_budget(self, n_new):
        return self.get_online_exemplar_size() + n_new < self.online_memory_budget

    def check_online_budget_for_each_class(self, n_new, m):
        # print(m, n_new)
        n_exemplar_of_m = self.online_exemplar_sets[m][0].shape[0]
        class_budget = (self.online_memory_budget // len(self.online_classes_so_far))
        return (n_exemplar_of_m + n_new) < class_budget

    def drop_old_instances(self, n_new, m):
        # print('Drop old instances to make memory available for current instances')
        class_budget = self.online_memory_budget // len(self.online_classes_so_far)
        n_exemplar_of_m = self.online_exemplar_sets[m][0].shape[0]
        memory_left = class_budget - n_exemplar_of_m
        self.online_exemplar_sets[m][0] = np.delete(self.online_exemplar_sets[m][0], np.arange(n_new - memory_left),
                                                    axis=0)
        self.online_exemplar_sets[m][1] = np.delete(self.online_exemplar_sets[m][1], np.arange(n_new - memory_left),
                                                    axis=0)

    def drop_instances_out_of_class_budget(self, x, y, n_classes):
        class_budget = self.online_memory_budget // n_classes
        # Reduce adding size when it is bigger than class budget
        if x.size(0) > class_budget:
            x = x[x.size(0) - class_budget:]
            y = y[y.size(0) - class_budget:]
        return x, y

    def reduce_memory_for_new_classes(self):
        # print('Reduce memory for new class')
        for m in self.online_classes_so_far:
            n_exemplar_of_m = self.online_exemplar_sets[m][0].shape[0]
            class_budget = self.online_memory_budget // (len(self.online_classes_so_far) + 1)
            if class_budget == 0:
                raise ValueError('Memory budget is too small, increase it to cover all classes so far!!!')

            if class_budget < n_exemplar_of_m:
                self.online_exemplar_sets[m][0] = np.delete(self.online_exemplar_sets[m][0],
                                                            np.arange(n_exemplar_of_m - class_budget), axis=0)
                self.online_exemplar_sets[m][1] = np.delete(self.online_exemplar_sets[m][1],
                                                            np.arange(n_exemplar_of_m - class_budget), axis=0)

    def add_instances_to_online_exemplar_sets(self, x, y, m):
        # print('Exemplar size: {}, adding size {}, class {}'.format(self.get_online_exemplar_size(), x.size(0), m))
        if m not in self.online_classes_so_far:
            x, y = self.drop_instances_out_of_class_budget(x, y, len(self.online_classes_so_far) + 1)
            # Reduce instances in each class to make room for new classes
            if not self.check_online_budget(x.size(0)):
                self.reduce_memory_for_new_classes()
            self.online_classes_so_far.append(m)
            self.online_exemplar_sets[m] = [x.cpu().detach().numpy(), y.cpu().detach().numpy()]
        else:
            if self.check_online_budget_for_each_class(y.size(0), m):
                self.online_exemplar_sets[m][0] = np.concatenate(
                    (self.online_exemplar_sets[m][0], x.cpu().detach().numpy()), axis=0)
                self.online_exemplar_sets[m][1] = np.concatenate(
                    (self.online_exemplar_sets[m][1], y.cpu().detach().numpy()), axis=0)
            else:
                x, y = self.drop_instances_out_of_class_budget(x, y, len(self.online_classes_so_far))
                # Drop old instances in exemplar to make available memory for very last instances
                self.drop_old_instances(x.size(0), m)

                self.online_exemplar_sets[m][0] = np.concatenate(
                    (self.online_exemplar_sets[m][0], x.cpu().detach().numpy()), axis=0)
                self.online_exemplar_sets[m][1] = np.concatenate(
                    (self.online_exemplar_sets[m][1], y.cpu().detach().numpy()), axis=0)

    def compute_class_means(self):
        # compute features for each class
        for i in range(len(self.online_exemplar_sets)):
            dataset = OnlineExemplarDataset(self.online_exemplar_sets[i])
            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())

                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            if i not in self.online_exemplar_means:
                self.online_exemplar_means[i] = class_mean.squeeze()
            else:
                self.online_exemplar_means[i] = 0.5 * (self.online_exemplar_means[i] + class_mean.squeeze())

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        self.train(mode=mode)

    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

    def classify_with_online_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        ex_means = []
        for k in self.online_exemplar_means:
            ex_means.append(self.online_exemplar_means[k])

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = ex_means if allowed_classes is None else [
            ex_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

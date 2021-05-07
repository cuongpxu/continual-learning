import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size())==4)
        for t, m, s in zip(tensor.permute(1,0,2,3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def _rotate_image(image, rotation):
    '''Rotating the pixels of an image according to [rotation].

        [image]         3D-tensor containing the image
        [rotation]   <float> rotation angle'''
    if rotation is None:
        return image
    else:
        img = transforms.ToPILImage()(image).rotate(rotation)
        image = transforms.ToTensor()(img)
        return image


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_augmentation(name, augment=False):
    dataset_transform = None
    if name in ['CIFAR10', 'CIFAR100'] and augment:
        dataset_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

    return dataset_transform


def get_dataset(name, type='train', download=True, capacity=None, dir='./datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    if type == 'train':
        transforms_list = [*AVAILABLE_TRANSFORMS['imagenet_augment']] if (name == 'imagenet' and augment) else \
            [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    else:
        transforms_list = [*AVAILABLE_TRANSFORMS['imagenet_test_augment']] if name == 'imagenet' else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name + "_norm"]]
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    if name == 'CUB2011':
        dataset = Cub2011(dir, train=False if type == 'test' else True,
                          transform=dataset_transform, target_transform=target_transform, download=download)
    elif name != 'imagenet':
        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                                download=download, transform=dataset_transform, target_transform=target_transform)
    else:
        dataset = dataset_class('{dir}/{type}'.format(dir=dir, type=type), transform=dataset_transform,
                                target_transform=target_transform)
    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class OnlineExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).
    '''
    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        images, labels = exemplar_sets[0], exemplar_sets[1]
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.target_transform = target_transform

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index] if self.target_transform is None else self.target_transform(self.labels[index])
        return image.float(), label.long()


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
                exemplar_id = index % total
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                break
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://drive.google.com/u/0/open?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = default_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
#----------------------------------------------------------------------------------------------------------#


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'CUB2011': None,
    'imagenet': datasets.ImageFolder,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'mnist_norm': [
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ],
    'cifar10_norm': [
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ],
    'cifar100_norm': [
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ],
    'cifar10_denorm': UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'cifar100_denorm': UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'CUB2011': [
        transforms.ToTensor(),
    ],
    'CUB2011_norm': [
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ],
    'CUB2011_denorm': UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'augment': [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ],
    'imagenet': [
        transforms.ToTensor()
    ],
    'imagenet_norm': [
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ],
    'imagenet_denorm': UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'imagenet_augment': [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ],
    'imagenet_test_augment': [
        transforms.Resize((224, 224))
    ]
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'CUB2011': {'size': 32, 'channels': 3, 'classes': 200},
    'imagenet': {'size': 224, 'channels': 3, 'classes': 1000}
}


#----------------------------------------------------------------------------------------------------------#

def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", normalize=False, augment=False,
                             only_config=False, verbose=False, exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    ## NOTE: option 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            train_dataset = get_dataset('mnist', type="train", dir=data_dir,
                                        target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(config['classes'] / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(config['classes']))) if exception \
                else np.random.permutation(list(range(config['classes'])))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'rotMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            train_dataset = get_dataset('mnist', type="train", dir=data_dir,
                                        target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate rotations
            if exception:
                rotations = [None] + np.random.choice(180, tasks - 1, replace=False).tolist()
            else:
                rotations = np.random.choice(180, tasks, replace=False).tolist()
            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, rot in enumerate(rotations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, transform=transforms.Lambda(lambda x, p=rot: _rotate_image(x, p)),
                    target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=rot: _rotate_image(x, p)),
                    target_transform=target_transform
                ))

    elif name == 'CIFAR10':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'CIFAR10' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar10']
        classes_per_task = int(np.floor(config['classes'] / tasks))

        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(config['classes'])))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            cifar10_train = get_dataset('cifar10', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar10_test = get_dataset('cifar10', type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                train_datasets.append(SubDataset(cifar10_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar10_test, labels, target_transform=target_transform))

    elif name == 'CIFAR100':
        # check for number of tasks
        if tasks>100:
            raise ValueError("Experiment 'CIFAR100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(config['classes'] / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(config['classes'])))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))

    elif name == 'CUB2011':
        print(tasks)
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'CUB-200-2011' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['CUB2011']
        classes_per_task = int(np.floor(config['classes'] / tasks))

        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(config['classes'])))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            cub2011_train = get_dataset('CUB2011', type="train", dir=data_dir, normalize=normalize,
                                        augment=augment, target_transform=target_transform, verbose=verbose)
            cub2011_test = get_dataset('CUB2011', type="test", dir=data_dir, normalize=normalize,
                                       target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario == 'domain' else None
                train_datasets.append(SubDataset(cub2011_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cub2011_test, labels, target_transform=target_transform))

    elif name == 'ImageNet':
        # check for number of tasks
        if tasks > 1000:
            raise ValueError("Experiment 'ImageNet' cannot have more than 1000 tasks!")
        # configurations
        config = DATASET_CONFIGS['imagenet']
        classes_per_task = int(np.floor(1000 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(1000)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            imagenet_train = get_dataset('imagenet', type="train", dir=data_dir, normalize=normalize,
                                         augment=augment, target_transform=target_transform, verbose=verbose)
            imagenet_test = get_dataset('imagenet', type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if scenario == 'domain' else None
                train_datasets.append(SubDataset(imagenet_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(imagenet_test, labels, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario == 'domain' else classes_per_task*tasks
    config['normalize'] = normalize if name in ['CIFAR10', 'CIFAR100', 'ImageNet'] else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["{}_denorm".format(name.lower())]

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


if __name__ == '__main__':
    image = torch.rand((1, 28, 28))
    print(image.size())
    img = transforms.ToPILImage()(image)
    # rot_img = F.rotate(img, 180)
    plt.imshow(img)
    plt.show()
    # plt.imshow(rot_img)
    # plt.show()
    # print(type(img), img.shape)
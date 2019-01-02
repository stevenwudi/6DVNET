import torch


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, img_dir, transforms=None):
        self.root = root
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


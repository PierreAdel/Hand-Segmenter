
import glob
from torch.utils.data import Dataset
import cv2
from torch.utils.data.dataloader import DataLoader


class EGATEDataset(Dataset):
    """
    Extended GTEA Gaze+ Hand Dataset from <http://cbs.ic.gatech.edu/fpv/>

    Args:
        dataset_path: should include an image and its annotation.
        transform: list of pytorch transforms to apply on the image.
        target_transform: list of pytorch transforms to apply on the target.
    """

    def __init__(self, dataset_path, transform=None, target_transform=None):

        self.dataset_path = dataset_path

        self.transform = transform
        self.target_transform = target_transform

        # get dir of all images
        self.img_list = glob.glob(self.dataset_path)

    # print number of imgs in dataset
    def __len__(self):
        return len(self.img_list)

    # get image and target by index
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is masks of the target classes.
        """

        img_file = self.img_list[idx]
        img = cv2.imread(img_file)

        target_file = img_file.replace('Images', 'Masks')
        target_file = target_file.replace('.jpg', '.png')
        target = cv2.imread(target_file)

        # not implemented yet
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.transform(target)

        return img, target


# Testing
if __name__ == "__main__":
    path = "F:\EGE\Images\*.jpg" # change this
    eEGATEDataset = EGATEDataset(path)
    print(len(eEGATEDataset))
    data_loader = DataLoader(dataset=eEGATEDataset, batch_size=3, shuffle=True, num_workers=4, drop_last=True)
    data_iter = iter(data_loader)
    tensor_image, tensor_label = next(data_iter)
    print(tensor_image, tensor_label)

# todo
# 1- from google drive
# 2- transforms
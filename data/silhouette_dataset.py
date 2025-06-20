import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from data.base_dataset import BaseDataset

SILHOUETTE_IMAGE_SIZE = (320, 320)

def normalize_silhouette_image(image, out_image_size=SILHOUETTE_IMAGE_SIZE):
    """
    :param image: input silhouette image
    :return: silhouette image cropped and resized to output dimensions
    """

    # Convert PIL Image to tensor
    tensor = torch.tensor(np.array(image))

    # Get indices of foreground pixels
    threshold = 0.5*255
    foreground_indices = torch.nonzero(tensor > threshold)

    # Calculate minimum and maximum indices in each dimension
    min_indices = foreground_indices.min(0)[0]
    max_indices = foreground_indices.max(0)[0]

    # Crop image using bounding box coordinates
    cropped_image = image.crop((min_indices[1].item(), min_indices[0].item(), max_indices[1].item(), max_indices[0].item()))

    # Resize silhouette to take up full output size height
    resized_silhouette_height = out_image_size[1]
    scale_factor = resized_silhouette_height/cropped_image.size[1]
    resized_image = cropped_image.resize((int(scale_factor*cropped_image.size[0]), int(resized_silhouette_height)), resample=Image.NEAREST)

    # Center image and pad to silhoutte image size
    normalized_image = Image.new('L', (out_image_size[0], out_image_size[1]))
    normalized_image.paste(resized_image, (int((out_image_size[0]-resized_image.size[0])/2), int((out_image_size[1]-resized_image.size[1])/2)))

    # Convert to tensor, normalize values from -1 to 1
    normalize_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    return normalize_transform(normalized_image)

class SilhouetteDataset(BaseDataset):
    """A custom dataset to load grayscale silhouette images from image paths stored in a CSV."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--dataset_csv_A', type=str, required=True, help='input CSV of image paths for dataset A')
        parser.add_argument('--dataset_csv_B', type=str, required=True, help='input CSV of image paths for dataset B')
        parser.add_argument('--dataroot_B', type=str, default=None, help='image root for dataset B')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        random.seed(432)    # Set constant seed to get consistent results every time
        self.A_paths = pd.read_csv(opt.dataset_csv_A, header=None).to_numpy().squeeze()
        self.B_paths = pd.read_csv(opt.dataset_csv_B, header=None).to_numpy().squeeze()
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.root_A = self.root
        self.root_B = self.root
        if opt.dataroot_B:
            self.root_B = opt.dataroot_B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        with Image.open('/'.join([self.root_A, A_path])) as img:
            A_img = img.convert('L')
        with Image.open('/'.join([self.root_B, B_path])) as img:
            B_img = img.convert('L')
        # apply image normalization
        A = normalize_silhouette_image(A_img)
        B = normalize_silhouette_image(B_img)

        if not self.opt.no_flip:
            A = transforms.RandomHorizontalFlip()(A)
            B = transforms.RandomHorizontalFlip()(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the two
        """
        return max(self.A_size, self.B_size)
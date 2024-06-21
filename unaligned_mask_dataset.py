from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os
import numpy as np
import torch
from torchvision import transforms

class UnalignedMaskDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets and include binary masks.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively, as well as binary masks.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_masks_A = os.path.join(opt.dataroot, opt.phase + 'A_binary_mask')  # path to binary masks for domain A
        self.dir_masks_B = os.path.join(opt.dataroot, opt.phase + 'B_binary_mask')  # path to binary masks for domain B

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_masks_paths = sorted(make_dataset(self.dir_masks_A, opt.max_dataset_size))
        self.B_masks_paths = sorted(make_dataset(self.dir_masks_B, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.A_masks_size = len(self.A_masks_paths)
        self.B_masks_size = len(self.B_masks_paths)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        #print("A_path:", A_path)
        A_basename = os.path.basename(A_path)
        A_mask_filename = A_basename.split('.')[0] + '_binary_mask.jpg'
        A_mask_path = os.path.join(self.dir_masks_A, A_mask_filename)
        #print("A_mask_path:", A_mask_path)
        
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
            
        B_path = self.B_paths[index_B]
        B_basename = os.path.basename(B_path)
        B_mask_filename = B_basename.split('.')[0] + '_binary_mask.jpg'
        B_mask_path = os.path.join(self.dir_masks_B, B_mask_filename)
        
        A_img = Image.open(A_path).convert('RGB')
        #print("A_img dimensions:", A_img.size)
        A_mask = Image.open(A_mask_path).convert('RGB') # Load domain A mask 

         # Crop or pad the mask to match the dimensions of real_A (256x256)
        #A_mask = Image.fromarray(A_mask.astype('uint8'), mode='L')
        #A_mask = self.crop_or_pad(A_mask, (256, 256))
        #A_mask = np.transpose(A_mask, (1, 0))  # Transpose the mask to match image dimensions
        #print("A_mask dimensions:", A_mask.shape)
        #A_mask = np.tile(A_mask, (1, 1, 3))  # Expand to three channels to match the image 
       
        # Convert NumPy array to PIL Image
        #A_mask_pil = Image.fromarray(A_mask.astype('uint8'))

        B_img = Image.open(B_path).convert('RGB')
        B_mask = Image.open(B_mask_path).convert('RGB') # Load domain B mask 

        # Crop or pad the mask to match the dimensions of real_B (256x256)
        #B_mask = Image.fromarray(B_mask.astype('uint8'), mode='L')
        #B_mask = self.crop_or_pad(B_mask, (256, 256))
        #B_mask = np.transpose(B_mask, (1, 0))  # Transpose the mask to match image dimensions
        #B_mask = np.tile(B_mask, (1, 1, 3))  # Expand to three channels to match the image

        # Apply transformation to images and masks  
        #A = self.transform_A(A_img).unsqueeze(0)
        #B = self.transform_B(B_img).unsqueeze(0)
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        #A_mask = self.transform_A(A_mask)
        #A_mask = torch.from_numpy(np.array(A_mask)).unsqueeze(0).float()  # Convert PIL Image back to NumPy array
        #B_mask = self.transform_B(B_mask)

        # Normalize masks for domain A and B (specialized for grayscale images)
        A_mask = transforms.functional.resize(A_mask, (256, 256))
        A_mask = transforms.functional.to_tensor(A_mask).float()
        A_mask = (A_mask - 0.5) / 0.5  # Normalize between -1 and 1
        #A_mask = A_mask.repeat(1,3, 1, 1)  # Repeat along the channel dimension to match real_A

        
        B_mask = transforms.functional.resize(B_mask, (256, 256))
        B_mask = transforms.functional.to_tensor(B_mask).float()
        B_mask = (B_mask - 0.5) / 0.5  # Normalize between -1 and 1
        #B_mask = B_mask.repeat(1,3, 1, 1)  # Repeat along the channel dimension to match real_B
        # Convert binary masks to PyTorch tensors
        #A_mask_tensor = torch.from_numpy(A_mask).unsqueeze(0).float()
        #A_mask_tensor = transforms.functional.to_tensor(A_mask).unsqueeze(0).float()
        #B_mask_tensor = transforms.functional.to_tensor(B_mask).unsqueeze(0).float()
        #A_mask_tensor = torch.from_numpy(A_mask).unsqueeze(0).float()  # Convert PIL Image back to NumPy array
        #B_mask_tensor = torch.from_numpy(B_mask).unsqueeze(0).float()

        return {'A': A, 'B': B, 'A_mask': A_mask, 'B_mask': B_mask, 'A_paths': A_path, 'B_paths': B_path, 'A_mask_paths': A_mask_path, 'B_mask_paths': B_mask_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
    

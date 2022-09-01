#%%
import os
import os.path
import sys
import math

import numpy as np
from skimage import io, img_as_ubyte, exposure, color
from PIL import Image
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms

from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path


#%%

# directory/
#       └── train
#           └── Images
#             ├── class_x
#             │   ├── xxx.ext
#             │   ├── xxy.ext
#             │   └── ...
#             │   └── xxz.ext
#             └── class_y
#                 ├── 123.ext
#                 ├── nsdf3.ext
#                 └── ...
#                 └── asd932_.ext
#       └── val
#           └── Images
#             ├── class_x
#             │   ├── xxx.ext
#             │   ├── xxy.ext
#             │   └── ...
#             │   └── xxz.ext
#             └── class_y
#                 ├── 123.ext
#                 ├── nsdf3.ext
#                 └── ...
#                 └── asd932_.ext

# When running the script, navigate to the "Images" folder in both the train and validation folders separately
# As such, you will need to run the script twice

inputdir = None   # Path to where the images are located
workers = 1   # Num workers
patches_rows = 2   # Num rows of patches to make
patches_cols = 2   # Num columns of patches to make
outputfolder = None   # Path to where to save the patches to
noscale = False   # keep the original intensity of patch
convert2gray = False   # convert to grayscale if set to true

#%% folder

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_EXTENSIONS_TIF = [
    '.tif', '.tiff', '.TIF', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS+IMG_EXTENSIONS_TIF)

def is_tif_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS_TIF)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_tif_loader(path):
    with open(path, 'rb') as f:
        try:
            with Image.open(f) as img:
                if "I;16" in img.mode:
                    tmp = img.convert('I')
                    tmp = tmp.point(lambda i:i*(1./256))
                    return tmp.convert('L').convert('RGB')
                else:
                    return img.convert('RGB')
        except:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        if is_tif_image_file(path):
            return pil_tif_loader(path)
        else:
            return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#%%
def main(argsIn):
    print("Beginning Patching")

    # set input dir
    root = Tk()
    root.wm_attributes('-topmost', 1)
    targetDataDirectory = askdirectory ()
    targetDataPath = Path(targetDataDirectory)
    root.destroy()
    
    inputdir = targetDataDirectory
    outputfolder = f'{targetDataPath.parent}/patches'
    if (os.path.exists(outputfolder) == False): 
        os.mkdir(outputfolder)

    # parse input dir
    input_dataset = ImageFolder(inputdir, transform=transforms.ToTensor())

    # get the list
    imgs = input_dataset.imgs
    print(f'num of files = {len(imgs)}')


    # get the list of classes
    classes = input_dataset.classes
    print(f'classes: {classes}')

    # set the data loader
    image_loader = data.DataLoader(input_dataset, batch_size=1, shuffle=False, num_workers=workers)

    print("=> creating patches and writing to disk")
    makeWritePatches(image_loader, imgs, outputfolder, noscale, convert2gray)


def makeWritePatches(image_loader, imgs, outputfolder, noscale, convert2gray):

    if noscale=='true' or noscale=='True' :
        normalize=False
    else:
        normalize=True

    if convert2gray=='true' or convert2gray=='True':
        convert2gray = True
    else:
        convert2gray = False

    for batch_i, (tensorImage, target) in enumerate(image_loader):
        # batch size is set to 1, so only 1 image is in each iteration of image_loader
        # tensor to numpy:
        image = tensorImage.numpy()
        image = np.transpose(np.squeeze(image[0,:,:,:]),(1,2,0))
        image_patches = image2patches(img_as_ubyte(image), patches_rows, patches_cols, normalize, convert2gray)

        #write to disk
        image_filename, label = imgs[batch_i]
        filepath, imfilename = os.path.split(image_filename)
        rootpath, classname = os.path.split(filepath)
        noextfn, ext = os.path.splitext(imfilename)

        for i, patch in enumerate(image_patches):
            newname = noextfn + "_%02d.png" %(i)
            outPath = os.path.join(outputfolder, classname)
            if not os.path.exists(outPath):
                os.makedirs(outPath)
            io.imsave(os.path.join(outPath, newname), patch)

def image2patches(image, patches_rows, patches_cols, normalize = True, convert2gray = False):

    m,n,k = image.shape

    if convert2gray:
        image_gray = color.rgb2gray(image.copy())
        image = np.dstack((image_gray, image_gray, image_gray))

    size_rows = float(m)/float(patches_rows)
    size_cols = float(n)/float(patches_cols)
    patch_size = math.floor(min(size_rows,size_cols))

    nb_rows = int(patches_rows)
    nb_cols = int(patches_cols)
    cover_rows = nb_rows*patch_size
    cover_cols = nb_cols*patch_size
    rows_offset = int(float(m-cover_rows)/2.0)
    cols_offset = int(float(n-cover_cols)/2.0)

    patches =[]
    for i in range(0,nb_rows):
        for j in range(0,nb_cols):
            patch = image[rows_offset+i*patch_size:rows_offset+(i+1)*patch_size, cols_offset+ j*patch_size:cols_offset+(j+1)*patch_size,:]
            if normalize and np.max(patch)>0:
                patch = exposure.rescale_intensity(patch)
                # histvals, histbins = exposure.histogram(patch)
                # if histbins[-1]==255:
                #     histvals = histvals[:-1]
                #     histbins = histbins[:-1]
                # total = np.sum(histvals)
                # cumsum = np.cumsum(histvals).astype(float)
                # inds = np.where(cumsum/total> 0.99)
                # satIntensity_2 = histbins[inds[0][0]]
                # if satIntensity_2>10:
                #     normalized_patch = patch.astype(float)/satIntensity_2
                #     normalized_patch[normalized_patch>1] = 1
                #     patch = img_as_ubyte(normalized_patch)
            patches.append(img_as_ubyte(patch))
    return patches


if __name__ == '__main__':
    main(sys.argv[1:])
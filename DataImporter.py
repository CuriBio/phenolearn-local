#%%
import os
import numpy as np
from skimage import io, img_as_ubyte, exposure, color
import sys
import math

import torch.utils.data
import torchvision.transforms as transforms
import folder
import logging

#%%

# directory/
#             ├── class_x
#             │   ├── xxx.ext
#             │   ├── xxy.ext
#             │   └── ...
#             │       └── xxz.ext
#             └── class_y
#                 ├── 123.ext
#                 ├── nsdf3.ext
#                 └── ...
#                 └── asd932_.ext

# parser = argparse.ArgumentParser(description='Patching')
# arg = parser.add_argument
# arg('data', metavar='DIR', help='path to dataset')
# arg('-j', '--workers', default=4, type=int,  help='number of data loading workers (default: 4)')
# arg('-pr', '--patches-rows', default=1, type=int, help='number of patches along rows (default: 1)')
# arg('-pc', '--patches-cols', default=1, type=int, help='number of patches along columns (default: 1)')
# arg('--outputfolder', default='./patches', type=str, metavar='PATH', help='output folder')
# arg('--noscale', default='false', type=str, help='keep the original intensity of patch')
# arg('--convert2gray', default='false', type=str, help='convert to grayscale if set to true')
# arg('--log-file', default='patch.log', type=str, help='the log file')

def main(argsIn):
    global args
    print("Patching setup={}".format(args))
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info('entered patching')
    logging.info("Patching setup={}".format(args))

    # set input dir
    inputdir = args.data

    # parse input dir
    input_dataset = folder.ImageFolder(inputdir, transform=transforms.ToTensor())

    # get the list
    imgs = input_dataset.imgs
    logging.info('num of files = {}'.format(len(imgs)))


    # get the list of classes
    classes = input_dataset.classes
    print("classes: ",classes)
    logging.info('classes = {}'.format(classes))

    # make the output dir
    if not os.path.exists(args.outputfolder):
        os.makedirs(args.outputfolder)

    # set the data loader
    image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    print("=> creating patches and writing to disk")
    makeWritePatches(image_loader, imgs, args.outputfolder, args.noscale, args.convert2gray)


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
        image_patches = image2patches(img_as_ubyte(image), args.patches_rows, args.patches_cols, normalize, convert2gray)

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
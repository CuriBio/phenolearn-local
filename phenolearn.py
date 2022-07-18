#%%
import os
import sys
import shutil

import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, CenterCrop, Normalize
import torchvision.datasets as datasets
import torchvision.models as models

from loss import LossRegression

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#%% Args

augment = True   #Whether to use data augmentation when creating the training set
traindir = None   #OS directory where the training images are stored
valdir = None    #OS directory where the validation images are stored
outputfolder = None    #OS directory where to store the outputs of the network
precrop_size = None   #Pixel size that the user selected images are
batch_size = None    #Batch size for SGD
workers = None   #Num workers (kinda like threads)
cpu = True   #Whether to use CPU or GPU for processing
GPU = not cpu   # Not sure why we need this one but hey, that's what they chose
do_weighted_sampling = False   #Not sure what this does
arch = 'resnet50'   #The model network to use (a key of model_names)
transfer = True   #Whether to use transfer learning
start_epoch = 0   #Which epoch to start with
epochs = 5   #Total number of epochs to run
mode = 'categorical'   #Other option is 'regression'.  Not sure what regression does
lr = .1   #What the learning rate of the optimizer is
momentum = .1   #What the momentum of the optimizer is
weight_decay = .001   #What the weight decay of the optimizer is
reg_weight = .001   #The regression weight to use
max_iters_per_epoch = 1000000   #A hard limiter for how many iterations you can do per epoch.  I think this is bad
print_freq = 100   #How verbose to make your logging

#%%
def main(argsIn):

    # decide whether to use CPU or GPU
    cpu = True
    GPU = not cpu
    
    #WHY IS THIS RANDOM VARIABLE DECLARED HERE
    best_prec1 = 0

    # load/define data
    traindir = None   #os.path.join(args.data, 'Train')
    valdir = None   #os.path.join(args.data, 'Val')

    # set output folder path
    outputfolder = None
    # if not os.path.exists(args.outputfolder):
    #     os.makedirs(args.outputfolder)

    # Create the normalization vector
    # NOTE I do not know why the devs chose these values
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Set the input image suze for the network
    input_image_size = 224

    # Derive the size of the input images (in Phenolearn it is a passed in argument)
    precrop_size = None        

    # If images are smaller than or equal to the size of the inputs to the CNN
    if precrop_size <= input_image_size:
        # Upsample the images to make them the proper size to be put into the CNN 
        precrop_size = input_image_size
        if augment:
            # The only augmentation performed is a random horizontal flip of probability 50%
            train_transforms = Compose([Resize(input_image_size),RandomHorizontalFlip(),ToTensor(),normalize])
        else:
            train_transforms = Compose([Resize(input_image_size),ToTensor(),normalize])
    else:
        if augment:
            # I am not sure how precrop_size is determined, but I assume the Resize(precrop_size) call just standardizes the size of all the images
            # But I have some serious reservations about CenterCrop(input_image_size).  You are potentially cutting out a ton of data, especially if the image is large!
            train_transforms = Compose([Resize(precrop_size),CenterCrop(input_image_size),RandomHorizontalFlip(),ToTensor(),normalize])
        else:
            train_transforms = Compose([Resize(precrop_size),CenterCrop(input_image_size),ToTensor(),normalize])

    # train dataset
    train_dataset = datasets.ImageFolder(traindir, train_transforms)
    # Not quite sure how you get classes here...
    classes = train_dataset.classes
    num_classes = len(classes)

    # val dataset
    # No matter what, the transformations that are applied to the validation set are to resize all the images the same precrop size, center crop for network input size, and normalization
    val_dataset =  datasets.ImageFolder(valdir, Compose([Resize(precrop_size), CenterCrop(input_image_size), ToTensor(), normalize]))
    # val_imgs = val_dataset.imgs

    # if weighted sampling
    # Not quite sure how it works, gonna set it to false and leave it alone for now
    if do_weighted_sampling:
        weights, count = make_weights_for_balanced_classes(train_dataset.imgs, num_classes)
        print("counts per class in train: {}".format(count))
        weights_train = torch.DoubleTensor(weights)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))

        # do for val too?
        weights, count = make_weights_for_balanced_classes(val_dataset.imgs, num_classes)
        print("counts per class in val: {}".format(count))
        weights_val = torch.DoubleTensor(weights)
        sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))

        shuffle = False
    else:
        sampler_train = None
        sampler_val = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle= shuffle, num_workers=workers, pin_memory=not cpu, sampler=sampler_train)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val, num_workers=workers, pin_memory=not cpu)

    # create model
    if os.path.isfile(os.path.join(outputfolder, "model_best.pth.tar")):
        # I ASSUME this chunk of code referes to if we are running a network again (re-training, refining, etc), we just want to load the previous results
        # If the network has been run before, create the previous results OS path
        checkpoint_path = os.path.join(outputfolder, "model_best.pth.tar")
        print("=> loading the checkpoint '{}'".format(checkpoint_path))

        # Select from the torchvision library which architecture you want to use
        model = models.__dict__[arch]()
        if transfer:
            # If using transfer learning, turn off the modification of gradients in the architecture
            print("=> transfer learning")
            for param in model.parameters():
                param.requires_grad = False

        # Load the previous;y run network
        checkpoint = torch.load(checkpoint_path)
        start_epoch = 0
        # Not sure why this check it here, if it doesn't pass then it should crash yet there is no error handling
        if 'ckp_num_classes' in checkpoint:
            num_classes = checkpoint['ckp_num_classes']
            print("number of classes in loaded checkpoint is {}".format(num_classes))

            # If I understand this correctly, this function just adds the final layer to the model to allow for classification
            model = getModelForFineTune(model, arch, num_classes)

            # Not sure what this is TBH
            model.load_state_dict(checkpoint['state_dict'])

    else:
        # So it's the first time you are running this network!  That's okay, we are going to build it from scratch
        print("=> using pre-trained model '{}'".format(arch))

        # When loading the model, make sure it is PRETRAINED.  Otherwise, we need a HUGE dataset
        model = models.__dict__[arch](pretrained=True)
        if transfer:
            print("=> transfer learning")
            # So now this is interesting.  Transfer learning only fixes the first layer, but layer 2, 3, and 4 can still be learnable?
            # for name, child in model.named_children():
               # print(f'{name}, {child}')
            for name, child in model.named_children():
                if name in ['layer2','layer3','layer4']:   
                    for param in child.parameters():
                        param.requires_grad = True

                else:
                    for param in child.parameters():
                        param.requires_grad = False


            # It looks like at one point they had it so that all the weights were fixed?  What changed?
            # for param in model.parameters():
            #    param.requires_grad = False

        if mode == 'categorical':
            model = getModelForFineTune(model, arch, num_classes)
        else: # regression
            num_classes = 1
            model = getModelForFineTune(model, arch, 1)

    ##### Training setup ##################################
    # GPU
    if GPU:
        model.cuda()
        print("=> using GPU")


    # I hate hate hate the use of fixed learning rates and momentum in a vanilla SGD.  There are sooooooo many better ways to do this, this is VERY outdated
    if transfer:
        # optimizer = torch.optim.SGD(model.fc.parameters(), args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # Only add to the optimizer those gradients that are learnable if transfer learning is involved
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)


    else:
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)


    if mode == 'categorical':
        # If we are doing categorical learning (as opposed to regression based), we want to use a cross entropy loss to classify the images at the end of the network.  This is pretty standard
        criterion = nn.CrossEntropyLoss()
        if GPU:
            criterion = criterion.cuda()
            cudnn.benchmark = True

    else: # regression

        criterion = LossRegression(reg_weight=reg_weight,out_min=-1.0,out_max=len(classes))


    #Acquire a baseline
    print("=> testing on validation data")
    acc_0, loss_0 = validate(val_loader, model, criterion, regression = (mode == 'regression'))
    train_accuracy = 0

    print("=> training")
    progress = os.path.join(outputfolder, "progress.txt")
    print("epoch,training_accuracy,training_loss,val_accuracy,val_loss", file=open(progress, "w"))
    print("{0},{1:.4f},{2:.4f},{3:.4f},{4:.4f}".format(0,0.0,5.0,acc_0,loss_0), file=open(progress, "a"))

    #Ah so it looks like at least we weren't using a flat learning rate.  That makes me happier, but we should still move away from SGD
    if mode == 'categorical':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.90)
    else:
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 1.0)

    for epoch in range(start_epoch, epochs):

        # train for one epoch
        prec1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch, max_iters_per_epoch, mode, arch, print_freq)

        # evaluate on validation set
        prec1, loss_val = validate(val_loader, model, criterion, regression = (mode == 'regression'))

        scheduler.step()

        # for param_group in optimizer.param_groups:
        #     print("learning rate is: ", param_group['lr'])

        # remember best prec@1 and save checkpoint
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            # keep the current train accuracy to report
            train_accuracy = prec1_train
            print(f'New record!  Training accuracy for epoch {epoch} is {train_accuracy}')

        filename = 'checkpoint_last.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'ckp_num_classes': num_classes,
        }, is_best, filename)


        print("{0},{1:.4f},{2:.4f},{3:.4f},{4:.4f}".format(epoch+1,prec1_train,loss_train,prec1,loss_val), file=open(progress, "a"))
            
#%%
best_prec1 = 0
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight, count

#%% Modify the final layer to have the correct number of class outputs that you need
def getModelForFineTune(model, arch, num_classes):

    if ('resnet' in arch) or ('inception' in arch):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif 'vgg' in arch:
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, num_classes))
        model.classifier = nn.Sequential(*feature_model)

    elif 'densenet' in arch:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif 'squeezenet' in arch:
        in_ftrs = model.classifier[1].in_channels
        features = list(model.classifier.children())
        features[1] = nn.Conv2d(in_ftrs, num_classes, kernel_size=1)
        features[3] = nn.AvgPool2d(13, stride=1)
        model.classifier = nn.Sequential(*features)
        model.num_classes = num_classes

    return model

#%% Validation function
def validate(val_loader, model, criterion, regression = False):
    losses = AverageMeter()
    top1 = AverageMeter()
    # In case you want to save the top two instead of the top 1
    # top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):

            if regression:
                target = target.view(target.size(0),1).float()

            if GPU:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            if regression:
                prec1 = regression_accuracy(output.data, target)
                top1.update(prec1, input.size(0))
            else:
                prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
                top1.update(prec1[0], input.size(0))

        print('Test: Loss {losses.avg:.3f}\t Prec@1 {top1.avg:.3f}'.format(losses=losses, top1=top1))

    return top1.avg, losses.avg

#%% Validation helper function
def train(train_loader, model, criterion, optimizer, epoch, max_iters_per_epoch, mode, arch, print_freq):

    losses = AverageMeter()
    top1 = AverageMeter()
    # In case you want to save the top two instead of the top 1
    # top2 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # WHY ARE WE STOPPING TRAINING EARLY WAHHHHHH
        if i<max_iters_per_epoch:

            #Not sure what this flush is for
            sys.stdout.flush()

            if mode == 'regression':
                target = target.view(target.size(0),1).float()

            if GPU:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            if arch=='inception_v3':
                output, aux = model(input)
            else:
                output = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))


            # measure accuracy and record loss
            if mode =='regression':
                prec1 = regression_accuracy(output.data, target)
                top1.update(prec1, input.size(0))
            else:
                prec1, prec2 = accuracy(output, target, topk=(1, 2))
                top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss.val:.4f} \t' 'Prec@1 {top1.val:.3f}'.format(epoch, i, len(train_loader), loss=losses, top1=top1))
        else:
            break


    print('Train: Loss {losses.avg:.3f}\t Prec@1 {top1.avg:.3f}'.format(losses=losses, top1=top1))
    return top1.avg, losses.avg

#%% Accuracy calculater helper functions (faster ways to do this with built in pytorch libraries)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def regression_accuracy(output, target, pct_close=0.45):

    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.view(batch_size,1)
        n_correct = torch.sum((torch.abs(pred - target) <= pct_close))
        acc = (n_correct.item() * 100.0 / batch_size)
        return acc

#%% A custom class for advanced average metrics during training and validation runs
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#%% A helper function to save a checkpoint every epoch, that can be reloaded if the subsequent epoch is worse than thecurrent
def save_checkpoint(state, is_best, filename):
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    filename = os.path.join(outputfolder,filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outputfolder,'model_best.pth.tar'))
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import random
import torch.nn.functional as F
import os
import random
import io
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import scipy.misc as misc
import numpy as np


#https://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula

# class TripletLoss(torch.nn.Module):
#
#     def __init__(self, margin=2.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output_anchor, output_negative, output_positive, label):
#         alpha = 1
#         euclidean_distance_positive = F.pairwise_distance(output_anchor, output_positive)
#         euclidean_distance_negative = F.pairwise_distance(output_anchor, output_negative)
#         triplet_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
# Define DeepFace Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding = 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Conv2d(512, 4096, 3, padding = 0)
        self.fc2 = nn.Conv2d(4096, 4096, 3, padding = 0)
        self.fc3 = nn.Conv2d(4096, 2622, 3, padding = 0)


    def forward(self, inputdata):

        #we go into relu after convolution. is that oka??
        #we also need to add resnet

        hidden = F.relu(self.conv1(self.bn(self.drop(inputdata))))
        hidden = self.pool(F.relu(self.conv2(hidden))) #no BN in pooling layers?? we also dont have pooling here in vgg

        hidden = F.relu(self.conv3(self.bn2(self.drop(hidden))))#what is r here? and we have pooling here as well?
        hidden = self.pool(F.relu(self.conv4(hidden)))

        hidden = F.relu(self.conv5(self.bn3(self.drop(hidden))))
        hidden = F.relu(self.conv6(self.bn4(self.drop(hidden))))
        hidden = self.pool(F.relu(self.conv7(hidden)))

        hidden = F.relu(self.conv8(self.bn4(self.drop(hidden))))
        hidden = F.relu(self.conv9(self.bn5(self.drop(hidden))))
        hidden = self.pool(F.relu(self.conv10(hidden)))

        hidden = F.relu(self.conv11(self.bn5(self.drop(hidden))))
        hidden = F.relu(self.conv12(self.bn5(self.drop(hidden))))
        hidden = self.pool(F.relu(self.conv13(hidden)))

        hidden = F.relu(self.fc1(hidden))
        hidden = F.relu(self.fc2(hidden))

        out = F.softmax(self.fc3(hidden))

        return out



def load_dataToDict(data_path):
    mydict = {}
    mydict_multi = {}
    for i in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, i)):
            mydict[i] = []
            for j in os.listdir(os.path.join(data_path, i)):
                mydict[i].append(j)
            if len(mydict[i])>1:
                mydict_multi[i] = mydict[i]

    return mydict, mydict_multi

# Generates triplets such that first two elements are same and third is different (anchor, positive, negative)
def generateRandomTriplets(data_path, data_dict, data_multiInstances, size):

    dataset = []
    keys = data_multiInstances.keys()
    out = open('Dataset.txt', 'w')
    for i in range(0, size):
        num = random.randint(0, len(keys)-1 )
        im1 = random.randint(0, len(data_multiInstances[keys[num]])-1 )
        im2 = random.randint(0, len(data_multiInstances[keys[num]])-2 )
        if im2 >= im1:
            im2 = im2 + 1

        a = os.path.join(data_path, keys[num], data_multiInstances[keys[num]][im1])
        p = os.path.join(data_path, keys[num], data_multiInstances[keys[num]][im2])

        keys_all = data_dict.keys()
        num_negative = random.randint(0, len(keys_all)-2 )
        if num_negative >= num:
            num_negative = num_negative + 1

        im_neg = random.randint(0, len(data_dict[keys_all[num_negative]])-1 )
        n = os.path.join(data_path, keys_all[num_negative], data_dict[keys_all[num_negative]][im_neg])

        out.write(data_multiInstances[keys[num]][im1] + ', ' + data_multiInstances[keys[num]][im2] + ', ' + data_dict[keys_all[num_negative]][im_neg] + '\n')

        dataset.append([a, p, n])
    return dataset


loader = transforms.Compose([
    #transforms.Scale(448),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


class miniBatchHelper():
    def __init__(self, imagedataset, transform = None):
        self.transform = transform #tranform the input (image augmentation)
        self.image_names = imagedataset

    def __getitem__(self, index):
        """
        __getitem__ supports the indexing such that dataset[i] can be used to
        get the ith sample
        """
        images = self.image_names[index]

        ima = Image.open(images[0])
        imp = Image.open(images[1])
        imn = Image.open(images[2])

        image_anc = np.array(ima)
        image_pos = np.array(imp)
        image_neg = np.array(imn)
        #print("index: {2}, img_name: {0}, label: {1}, label_size: {3}".format(img_name, label, index+1, oh_label.size))
        #print("image shape: {0}, label shape: {1}".format(image.shape, oh_label.shape))

        ima.close()
        imp.close()
        imn.close()


        transformNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,0,0],[0.5,0.5,0.5])])

    	image_anc = transformNorm(image_anc)
    	image_pos = transformNorm(image_pos)
    	image_neg = transformNorm(image_neg)


        sample = {'anchor': image_anc, 'positive': image_pos, 'negative': image_neg}

        if self.transform:
            sample = self.transform(sample)

        return sample, images

    def __len__(self):
        """__len__ returns the size of the dataset. Use by calling len(dataset)"""
        return len(self.image_names)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_anc, image_pos, image_neg = sample['anchor'], sample['positive'], sample['negative']
        image_anc = misc.imresize(image_anc, (224, 224))
        image_pos = misc.imresize(image_pos, (224, 224))
        image_neg = misc.imresize(image_neg, (224, 224))
        #image = image.transpose((2, 0, 1))
        #print(image.shape)
        #print("input size: ({0}, {1}, {2})".format(INPUT_DEPTH, INPUT_SIZE[0], INPUT_SIZE[1]))
        image_anc = torch.from_numpy(image_anc).float()
        image_pos = torch.from_numpy(image_pos).float()
        image_neg = torch.from_numpy(image_neg).float()
        sample = {'anchor': image_anc, 'positive': image_pos, 'negative': image_neg}
        return sample

#learning_rate = 0.01
def test(dataset):
	

    f = open('testData.txt', 'w')
    neural_net = torch.load('model')
    triplet_loss = nn.TripletMarginLoss(margin = 1.0)
    neural_net.cuda()

    correct = 0
    total = 0
    similar_total = 0
    different_total = 0
    similar_correct = 0
    different_correct = 0
    similar_wrong = []
    different_wrong = []
    for data, paths in dataset:
        #print data[0], data[1], data[2]

        randomInteger = random.randint(1, 2)
	
	#print paths
        image1 = Variable(data['anchor']).cuda()
	if randomInteger == 1:
        	image2 = Variable(data['positive']).cuda()
		f.write("" + paths[0][0] + ', ' +  paths[1][0]+ '\n')
	elif randomInteger == 2:
		image2 = Variable(data['negative']).cuda()
		f.write("" + paths[0][0] + ', ' +  paths[2][0] + '\n')
        #n = Variable(data['negative']).cuda()

        #print a.size(), a.size()[1]
        image1 = image1.transpose(1,3)
        image2 = image2.transpose(1,3)
        #n = n.transpose(1,3)

        #image1 = Variable(loader(Image.open(data[0]))).cuda().unsqueeze(0)
        #image2 = Variable(loader(Image.open(data[randomInteger]))).cuda().unsqueeze(0)

        output1 = neural_net(image1)
        output2 = neural_net(image2)

        output1 = output1.squeeze(2)
        output1 = output1.squeeze(2)
        output2 = output2.squeeze(2)
        output2 = output2.squeeze(2)

        #print total

        pdist = nn.PairwiseDistance(p=2)
        distance = pdist(output1, output2)

        #if randomInteger == 1:
        #    print 'similar', distance
        #elif randomInteger == 2:
        #    print 'different', distance

	total += 1
	
	#print 'DISTANCE', distance.data[0], 'DIST'
	if distance.data.cpu().numpy()[0]> 0.00052:
		if randomInteger == 1:
			similar_total+=1
			similar_wrong.append([paths[0],paths[1],distance.data.cpu().numpy()[0]])
		else:
			different_total+=1
			different_correct+=1
			correct+=1
	else:
		if randomInteger == 1:
			similar_total+=1
			similar_correct+=1
			correct+=1
		else:
			different_total+=1
			different_wrong.append([paths[0], paths[2], distance.data.cpu().numpy()[0]])

    print 'Accuracy', correct*100.0/total
    print 'Similar Accuracy', similar_correct*100.0/similar_total
    #print 'Similar wrong distance: '
    #for sim in similar_wrong:
	#print sim
    print 'Different Accuracy', different_correct*100.0/different_total
    #print 'Different wrong distance: '
    #for diff in different_wrong:
	#print diff
data, data_simiar = load_dataToDict('lfw')
dataset = generateRandomTriplets('lfw', data, data_simiar, 1000)
minibatches = miniBatchHelper(imagedataset = dataset, transform = ToTensor())
trainData = DataLoader(minibatches, batch_size=1, shuffle = True, num_workers = 2)
test(trainData)


#print data.keys()

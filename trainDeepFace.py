#add dropout
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
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import scipy.misc as misc


#https://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula

# class TripletLoss(torch.nn.Module):#
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


#Contrastive loss function
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = reduce(lambda x, y: x + y, x0) / len(x0) - reduce(lambda x, y: x + y, x1) / len(x1)
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

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

        self.fc1 = nn.Conv2d(512, 1024, 3, padding = 0)
        self.fc2 = nn.Conv2d(1024, 1024, 3, padding = 0)
        self.fc3 = nn.Conv2d(1024, 2622, 3, padding = 0)


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


# #loader = transforms.Compose([
#     #transforms.Scale(448),  # scale imported image
#     transforms.ToTensor()])  # transform it into a torch tensor


learning_rate = 0.000001
def train(training_batches):

    neural_net = Net()
    neural_net#.cuda()
    contrastive_loss=ContrastiveLoss(margin = 0.0001)#.cuda()
    triplet_loss = nn.TripletMarginLoss(margin = 0.0001)#.cuda()
    learning_rate = 0.000001
    optimizer = torch.optim.Adam(neural_net.parameters(), learning_rate)
    count = 0

    neural_net.train()
    print "came here"
    for batch, data in enumerate(training_batches):
        #print data[0], data[1], data[2]
        print "start iterations"
        count+=1

        a = Variable(data['anchor'])#.cuda()
        p = Variable(data['positive'])#.cuda()
        n = Variable(data['negative'])#.cuda()
	
	#data['anchor'].close()
	#data['positive'].close()
	#data['negative'].close()

        #print a.size(), a.size()[1]
        a = a.transpose(1,3)
        p = p.transpose(1,3)
        n = n.transpose(1,3)

        #print '1'
        anc = neural_net(a)
        #print '2'
        pos = neural_net(p)
        #print '3'
        neg = neural_net(n)
        #print '4'


        anc = anc.squeeze(2)
        anc = anc.squeeze(2)
        pos = pos.squeeze(2)
        pos = pos.squeeze(2)
        neg = neg.squeeze(2)
        neg = neg.squeeze(2)
	
	#print anc
	#print pos
	#print neg
        if count<0.7*(maxData/16):
            print "Triplet"
            loss = triplet_loss(anc, pos, neg)

        else:   #do contrastive loss
            print "Contrastive"
            rand_img=random.sample([pos,neg],1)
            if rand_img is pos:
                my_y=0
            else:
                my_y=0.0001
            loss=contrastive_loss(anc,rand_img,my_y)

        
        #if count%100==0:
            #learning_rate=0.8*learning_rate
            
        #optimizer = torch.optim.Adam(neural_net.parameters(), learning_rate)
        print "Iteration: "+str(count), "Loss: " ,loss
        print "learning_rate: "+str(learning_rate)


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        #print '6'

	if count%32 ==0:
    		torch.save(neural_net, 'final_raj_trained_model10')
    torch.save(neural_net, 'final_raj_trained_model10')

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

	ima.close()
	imp.close()
	imn.close()

	#image_anc = nn.Normalize(image_anc)
	#image_pos = nn.Normalize(image_pos)
	#image_neg = nn.Normalize(image_neg)

	transformNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,0,0],[0.5,0.5,0.5])])

	image_anc = transformNorm(image_anc)
	image_pos = transformNorm(image_pos)
	image_neg = transformNorm(image_neg)

        #print("index: {2}, img_name: {0}, label: {1}, label_size: {3}".format(img_name, label, index+1, oh_label.size))
        #print("image shape: {0}, label shape: {1}".format(image.shape, oh_label.shape))
        sample = {'anchor': image_anc, 'positive': image_pos, 'negative': image_neg}

	#transformNorm = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
	#self.transform = transforms.Compose([self.transform, transforms.ToPILImage(), transformNorm])	
		

        if self.transform:
            sample = self.transform(sample)

        return sample

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


maxData=10000
data, data_simiar = load_dataToDict('lfw')
print "data loaded"
dataset = generateRandomTriplets('lfw', data, data_simiar, maxData)
print "generated triplets"
minibatches = miniBatchHelper(imagedataset = dataset, transform = ToTensor())
print "minibatches done"
trainData = DataLoader(minibatches, batch_size=16, shuffle = True, num_workers = 2)
print "about to start training"
learning_rate = 0.0001
train(trainData)


#print data.keys()

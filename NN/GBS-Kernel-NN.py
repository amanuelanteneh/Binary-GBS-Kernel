#import standard ML & graph libraries
import networkx as nx
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
torch.manual_seed(1)

#import regualr python libraries
from tabulate import tabulate
import sys

#import GraKel library
from grakel.datasets import fetch_dataset
from grakel import graph_from_networkx
from grakel import GraphKernel, Graph

#import quantum optics library
from strawberryfields.apps import sample

import warnings  #to ignore complex cast warning
warnings.filterwarnings('ignore')


class GraphNNClassifier(nn.Module):  # inheriting from nn.Module

    def __init__(self, numClasses, maxGraphSize):

        super(GraphNNClassifier, self).__init__()

        # input dimesnion of network is max graph size + 1 bc thats the dimension of the feature vector
        # output dimension of network is the number of classes
        self.linear = nn.Linear((maxGraphSize)+1, numClasses)


    def forward(self, featureVec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        return F.log_softmax(self.linear(featureVec), dim=1)

def processDatasets(datasetNames, removeIsolates=False, keepLargestCC=False):
  cleanGraphs = {} 
  for name in datasetNames:
      graphData = fetch_dataset(name, verbose=False, as_graphs=True) #get data sets 
      graphs, y = graphData.data, graphData.target # split into x and y set
      maxNodes = max([g.get_adjacency_matrix().shape[0] for g in graphs])
      filteredGraphs = []
      filteredLabels = []
      for i in range(0, len(graphs)):
        
        G = nx.from_numpy_array( graphs[i].get_adjacency_matrix() ) 
        if (removeIsolates==True): #remove isolated nodees
          G.remove_nodes_from(list(nx.isolates(G)))  
        if (keepLargestCC == True): #extract largest connected component
          largestCC = max(nx.connected_components(G), key=len)
          G = G.subgraph(largestCC).copy()
        graphs[i] = G #store the nx version of the graph for gbs application
        
        # filter datasets for have 6 <= x <= 25 nodes
        if ( nx.adjacency_matrix( graphs[i] ).shape[0] >= 6 and  nx.adjacency_matrix( graphs[i] ).shape[0] <= 25):
          filteredGraphs.append(graphs[i]) 
          filteredLabels.append(y[i])
      
      lb = LabelEncoder() 
      # pytorch requires labels be 1 or 0
      filteredLabelsEncoded = lb.fit_transform(filteredLabels)
      cleanGraphs[name] = (filteredGraphs, filteredLabelsEncoded) #format of dict values - tuple: (non-padded graphs, class labels)
  return(cleanGraphs, maxNodes)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


class GraphDataset(Dataset):
    def __init__(self, datasetIndex, datasets, transform=None):
        self.datasetName = datasets[datasetIndex]
        self.processedGraphs, self.maxNodes = processDatasets([self.datasetName], False, False) 
        self.featureVectors = []

        for key in self.processedGraphs:
          
          for g in self.processedGraphs[key][0]:
            adjMat = nx.to_numpy_array(g) 

            v = generateFeatVector(adjMat, numSamples, numPhotons, displacement, maxPhotons, self.maxNodes) 
    
            self.featureVectors.append(v)
        
    def __len__(self):
        return len(self.processedGraphs[0])
    
    def __getitem__(self, index):
        # format: (feature vector, label) as pytorch tensors
        return torch.tensor(self.featureVectors[index]), torch.tensor(self.processedGraphs[self.datasetName][1][index])

"""
Function to generate feature vectors, to avoid OUT_OF_MEMORY error we generate half the
desired number of samples first the delete
"""
def generateFeatVector(adjMatrix, numSamples, meanN, displacement, maxPhotons, maxModes):
  
  M = maxModes #feature vectors are of length maxNodes
  N = numSamples//10 # 1/10 total number of samples to avoid OUT_OF_MEMORY error
  v = [] #feature vector

  for i in range(0, M+1): #create vector of 0's of length M+1
    v.append(0)
    
  for i in range(10):
    # sample using SF sample module
    samples = sample.sample(adjMatrix, meanN, N, threshold=True)
   
    for L in samples: #for each sample add up all photons seen in each mode
      numClicks = sum(L) 
      v[numClicks] += 1
  
    samples.clear() #clear array to free up memory
    
  for i in range(len(v)): #divide by total number of samples to get probability
      v[i] /= numSamples

  return(v)


"""
In the next lines we calcaute the GBS-generated feature vector for each graph in each data set
"""

#read in command line args
numSamples = 2000
numPhotons = 5
displacement = 0.0
maxPhotons = 6
datasetIndex = 9

#Fetch and preprocess graph datasets

datasetNames = ["AIDS", "BZR_MD", "COX2_MD", "ENZYMES", "ER_MD", "IMDB-BINARY", "MUTAG", "NCI1", "PROTEINS", "PTC_FM", "DD", "COLLAB", "FINGERPRINT"]

dataset = GraphDataset(datasetIndex=datasetIndex, datasets=datasetNames)

batch_size = 5
validation_split = .3
shuffle_dataset = True
random_seed= 42

dataset_size = len(dataset.processedGraphs[datasetNames[datasetIndex]][0])

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
  np.random.seed(random_seed)
  np.random.shuffle(indices)
  
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

table = [ [ "Data set",  datasetNames[datasetIndex] ] ]
table.append( [ "Number of\n Graphs" ] ) #row 1
table.append( [ "Avg. Number\n of Edges" ] )
table.append( [ "Avg. Number\n of Vertices" ] )
table.append( [ "Number of\n Classes" ] )

for key in dataset.processedGraphs:
  avgNodes = sum([ g.number_of_nodes() for g in dataset.processedGraphs[key][0] ]) / len(dataset.processedGraphs[key][0]) 
  avgEdges = sum([ g.number_of_edges() for g in dataset.processedGraphs[key][0] ]) / len(dataset.processedGraphs[key][0])
  numClasses = len(set(dataset.processedGraphs[key][1]))
  table[1].append( len(dataset.processedGraphs[key][0]) )  
  table[2].append(avgEdges)
  table[3].append(avgNodes)
  table[4].append(numClasses)

table = np.array(table).T.tolist()

print(tabulate(table, headers='firstrow', tablefmt='simple'), flush=True)
    

print("\nNumber of samples per graph:", numSamples,"\nMean number of photons:", \
numPhotons, "\nDisplacement on each mode:", displacement, "\nMax photon number:", maxPhotons, "\n", flush=True)

print("Max nodes:", dataset.maxNodes)

model = GraphNNClassifier(numClasses, dataset.maxNodes)
    
print(model)
    
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


train_acc = []
train_loss = []
total_step = len(train_loader)

for epoch in range(20):
  correct_train = 0
  total_train = 0
  running_loss = 0.0
  for batch_index, (instance, label) in enumerate(train_loader):
   # PyTorch accumulates gradients.
   # We need to clear them out before each instance
    model.zero_grad()
    
    featureVec = instance
    target = label

    # Run our forward pass.
    outputs = model(featureVec)

    # Compute the loss, gradients, and update the parameters by
    # calling optimizer.step()
    loss = loss_function(outputs, target)
    loss.backward()
    running_loss += loss.item()
    optimizer.step()
    _,pred = torch.max(outputs, dim=1)
    correct_train += torch.sum(pred==label).item()
    total_train += label.size(0)
  train_acc.append(100 * correct_train / total_train)
  train_loss.append(running_loss/total_step)
  print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct_train / total_train):.4f}')

  val_acc = []    
  val_loss = []
  correct_val = 0
  total_val = 0
  batch_loss = 0
  with torch.no_grad():
      model.eval()
      for (instance, label) in (validation_loader):
        outputs_t = model(instance)
        loss_t = loss_function(outputs_t, label)
        batch_loss += loss_t.item()
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_val += torch.sum(pred_t==label).item()
        total_val += label.size(0)
    
      val_acc.append(100 * correct_val / total_val)
      val_loss.append(batch_loss/len(validation_loader))      
      print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_val / total_val):.4f}\n')


  
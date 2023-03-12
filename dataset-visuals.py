#import standard ML libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#import graph libraries library
import networkx as nx
from grakel import graph_from_networkx
from grakel import GraphKernel, Graph
from grakel.datasets import fetch_dataset

#import quantum optics librarys
from thewalrus.samples import torontonian_sample_state
from thewalrus.quantum import gen_Qmat_from_graph, Covmat

#import standard py libraries
from tabulate import tabulate
import sys
import itertools

import warnings  #to ignore complex cast warning
warnings.filterwarnings('ignore')
def load_fingerprint(path_to_folder):
    node2graph = {}
    Gs = []
    
    with open(path_to_folder+"/Fingerprint_graph_indicator.txt", "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1
    
    with open(path_to_folder+"/Fingerprint_A.txt", "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))
    
    with open(path_to_folder+"/Fingerprint_node_attributes.txt", "r") as f:
        c = 1
        for line in f:
            Gs[node2graph[c]-1].nodes[c]['attributes'] = np.array(line.split(','), dtype=float)
            c += 1
        
    labels = []
    with open(path_to_folder+"/Fingerprint_graph_labels.txt", "r") as f:
        for line in f:
            labels.append(int(line[:-1]))
    
    labels  = np.array(labels, dtype=float)
    return Gs, labels


def preprocessDataset(dataset, removeIsolates=False, keepLargestCC=False):
  if (dataset == "FINGERPRINT"):
     graphsNX, y = load_fingerprint('graphs/Fingerprint')
     graphs =  list(graph_from_networkx(graphsNX, as_Graph=True))
  else:
       graphData = fetch_dataset(dataset, verbose=False, as_graphs=True) # get dataset 
       graphs, y = graphData.data, graphData.target # split into X and y set
  
  filteredGraphs = []
  filteredLabels = []
  for i in range(0, len(graphs)): 
    
    if dataset == "FINGERPRINT":
       # as done by Schuld et al. for the fingerprint dataset filter 
       # out graphs that aren't of the 3 dominant classes 0, 4 and 5 
       if (y[i] != 0.0 and y[i] != 4.0 and y[i] != 5.0):
          continue
    
    G = nx.from_numpy_array( graphs[i].get_adjacency_matrix() ) 
        
    if (removeIsolates==True): # remove isolated nodes if opted for
      G.remove_nodes_from(list(nx.isolates(G)))  
        
    if (keepLargestCC == True): # extract largest connected component if opted for
      largestCC = max(nx.connected_components(G), key=len)
      G = G.subgraph(largestCC).copy()
        
    # filter datasets to have 6 <= |V| <= 25 nodes
    # if user specefices an M > 6 this would need to be modified so that graphs of 
    # size < M are also filtered out
    if (graphs[i].get_adjacency_matrix().shape[0] >= 6 and  graphs[i].get_adjacency_matrix().shape[0] <= 25):
      # for GBS kernel return adj matrix not GraKel graph object  
      filteredGraphs.append(G) 
      filteredLabels.append(y[i])

  # return a tuple: index 0 is the preprocessed graphs, index 1 is their labels
  return(filteredGraphs, filteredLabels)


def makeVisual(datasetNames, mode):
    for i in range(len(datasetNames)):

      colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
      labels = [ "Class " + str(r) for r in range(0, 6) ] #list(range(0, 6))
      graphs, y = preprocessDataset(datasetNames[i]) #graphData.data, graphData.target # split into X and y set
      for j in range(len(y)):
          if y[j] == -1:
             y[j] = 0
          if datasetNames[i] == 'ENZYMES':
             y[j] -= 1
          if datasetNames[i] == 'FINGERPRINT':
            if y[j] == 0.0:
                y[j] = 0
            if y[j] == 4.0:
                y[j] = 1
            if y[j] == 5.0:
                y[j] = 2

          if datasetNames[i] == 'PROTEINS':
            if y[i] == 1: 
               y[j] = 0
            else:
               y[j] = 1

      #print(datasetNames[i], set(y))
      
      xmax = 28
      if mode == 3:
         xmax = xmax**2 // 2
        
      x = np.arange(xmax)
      
        
      classes = len(set(y))

      counts = [ list(np.zeros(xmax, dtype=int)) for c in range(classes) ]

      ax = fig.add_subplot(6, 2, i+1)
      for j in range(len(graphs)):
          if mode == 0:
              n = graphs[j].number_of_nodes()
              ax.set_xlabel('Graph size')
              name = 'size-histogram.png'
          elif mode == 1:
              n = graphs[j].degree
              n = max( [ c[1] for c in n ] )
              ax.set_xlabel('Maximum degree')
              name = 'degree-histogram.png'
          elif mode == 2:
              n = nx.node_connectivity(graphs[j])
              ax.set_xlabel('Connectivity')
              name = 'connectivity-histogram.png'
          elif mode == 3:
              #xmax = 28**2
              #x = np.arange(xmax)
              n = graphs[j].number_of_edges()
              ax.set_xlabel('Number of edges')
              name = 'edge-histogram.png' 
          
          
          counts[y[j]][n] += 1

      width = 0.8/classes
      if datasetNames[i] == "ENZYMES":
         width = 1.8/classes
      
      ax.axis(xmin=-1.0, xmax=xmax)
      for c in range(classes):
          ax.bar(x+width*c, counts[c], width=width, color=colors[c], label=labels[c], alpha=0.7, align='center')

      
      ax.set_ylabel('Count')
      ax.set_xticks(range(0, xmax, 5))
      ax.set_title(datasetNames[i])
      #ax.autoscale(tight=True)
      if datasetNames[i] in ["BZR_MD", "COX2_MD", "ER_MD"] and mode == 2:
          ax.legend(prop={'size': 15}, loc='upper left')
      elif mode !=3:
          ax.legend(prop={'size': 15}, loc='upper left')
      else:
          ax.legend(prop={'size': 15}, loc='upper right')
          ax.set_xticks(range(0, xmax, 25))

    fig.tight_layout()
    plt.savefig(name, dpi=240)

    plt.clf()

    
datasetNames = ["AIDS", "BZR_MD", "COX2_MD", "ENZYMES", "ER_MD", \
                "IMDB-BINARY", "MUTAG", "NCI1", "PROTEINS", "PTC_FM", "FINGERPRINT"]


plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(22, 19), dpi=180 )

makeVisual(datasetNames, 0)
makeVisual(datasetNames, 1)
makeVisual(datasetNames, 2)
makeVisual(datasetNames, 3)

plt.close()


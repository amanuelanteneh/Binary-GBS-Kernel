#import standard ML libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#import graph libraries library
import networkx as nx
from grakel.datasets import fetch_dataset
from grakel import graph_from_networkx
from grakel import GraphKernel, Graph, SubgraphMatching

#import standard py libraries
from tabulate import tabulate
import sys

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
     graphsNX, y = load_fingerprint('../graphs/Fingerprint')
     graphs =  list(graph_from_networkx(graphsNX, as_Graph=False))
  elif (dataset == "IMDB-BINARY"):
       graphData = fetch_dataset(dataset, verbose=False, as_graphs=True) # get dataset 
       graphs, y = graphData.data, graphData.target # split into X and y set
  else:
       graphData = fetch_dataset(dataset, verbose=False, as_graphs=False) # get dataset 
       graphs, y = graphData.data, graphData.target # split into X and y set
  
  filteredGraphs = []
  filteredLabels = []
  for i in range(0, len(graphs)): 
    
    if dataset == "FINGERPRINT":
    # as done by Schuld et al. for the fingerprint dataset filter out graphs that aren't of the 3 dominant classes 0, 4 and 5
       # also for some reason some of the adj matriceis aren't square so filter those out as well...
       if (y[i] != 0.0 and y[i] != 4.0 and y[i] != 5.0):
          continue
       
       if (len(graphs[i][0]) >= 6 and len(graphs[i][0]) <= 25):
          # for GBS kernel return adj matrix not GraKel graph object  
          filteredGraphs.append(graphs[i]) 
          filteredLabels.append(y[i])
    elif dataset == "IMDB-BINARY":
       if (graphs[i].get_adjacency_matrix().shape[0] >= 6 and  graphs[i].get_adjacency_matrix().shape[0] <= 25):
          # for GBS kernel return adj matrix not GraKel graph object  
          filteredGraphs.append(graphs[i]) 
          filteredLabels.append(y[i])
    else:      
        # filter datasets to have 6 <= |V| <= 25 nodes
        if (len(graphs[i][1]) >= 6 and len(graphs[i][1]) <= 25):
          # for GBS kernel return adj matrix not GraKel graph object  
          filteredGraphs.append(graphs[i]) 
          filteredLabels.append(y[i])

  # return a tuple: index 0 is the prerocessed graphs, index 1 is their labels
  print("Number of graphs:", len(filteredGraphs))
  return(filteredGraphs, filteredLabels)


#read in command line args
datasetIndex = int(sys.argv[1])

datasetNames = ["AIDS", "BZR_MD", "COX2_MD", "ENZYMES", "ER_MD", \
                "IMDB-BINARY", "MUTAG", "NCI1", "PROTEINS", "PTC_FM", "FINGERPRINT", "IMDB-MULTI"]
dataset = datasetNames[datasetIndex]

X, y = preprocessDataset(dataset) 

print("\nKernel: Subgraph Matching", "\n", flush=True)

table = [ [ "Data set", dataset ] ] #reset table
table.append( [ "SVM" ] ) #SVM standard error column

# subgraph matching kernel
sm = {"name": 'subgraph_matching', 'kv': None, 'ke': None, "k":5}
#sm = SubgraphMatching(verbose=True, normalize=True, kv=None, ke=None)

paramGrid = [{'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0 ,1e2, 1e3]}]

accuracies = []

for i in range(10): # 10 repeats of double cross validation
  seed = i
  kfold1 = KFold(n_splits=10, shuffle=True, random_state=None)
  classifier = make_pipeline(
       GraphKernel(kernel=sm, normalize=True),
       #sm,
       GridSearchCV(SVC(kernel='precomputed', class_weight='balanced'), param_grid=paramGrid,
                      scoring="accuracy", cv=kfold1))

  kfold2 = KFold(n_splits=10, shuffle=True, random_state=None)

  scores = cross_validate(classifier, X=X, y=y, cv=kfold2, n_jobs=-1, return_train_score=False)

  # Get best output of this fold
  accuracies.append(np.mean(scores['test_score']))

  # Get stats of this fold and print
  testMean = np.mean(scores['test_score'])
  testStd = np.std(scores['test_score'])
    

results = np.array(accuracies)

table = [ [ "Data set", dataset ] ] #reset table
table.append( ["SVM"] ) #SVM standard deviation column
table[1].append( str( '%2.2f'% (100*np.mean(results) ) ) + " \u00B1 " + str( '%2.2f'% (100*np.std(results) ) ) )
print(tabulate(table, headers='firstrow', tablefmt='simple'))
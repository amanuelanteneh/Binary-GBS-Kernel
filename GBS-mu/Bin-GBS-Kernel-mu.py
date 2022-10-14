#import standard ML libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
#from load_data import load_dortmund_graphs, get_grakel_graphs


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

"""
Given graphs and labels returns a tuple of form (X, y) of graphs and labels filtered by size
"""
def preprocessDataset(dataset, removeIsolates=False, keepLargestCC=False):
  if (dataset == "FINGERPRINT"):
     graphsNX, y = load_fingerprint('../../graphs/Fingerprint')
     graphs =  list(graph_from_networkx(graphsNX, as_Graph=True))
  else:
       graphData = fetch_dataset(dataset, verbose=False, as_graphs=True) # get dataset 
       graphs, y = graphData.data, graphData.target # split into X and y set
  
  filteredGraphs = []
  filteredLabels = []
  for i in range(0, len(graphs)): 
    
    if dataset == "Fingerprint" or dataset == "FINGERPRINT":
    # as done by Schuld et al. for the fingerprint dataset filter out graphs that aren't of the 3 dominant classes 0, 4 and 5
       # also for some reason some of the adj matriceis aren't square so filter those out as well...
       if (y[i] != 0.0 and y[i] != 4.0 and y[i] != 5.0):
          continue
    
    G = nx.from_numpy_array( graphs[i].get_adjacency_matrix() ) 
        
    if (removeIsolates==True): # remove isolated nodes if opted for
      G.remove_nodes_from(list(nx.isolates(G)))  
        
    if (keepLargestCC == True): # extract largest connected component if opted for
      largestCC = max(nx.connected_components(G), key=len)
      G = G.subgraph(largestCC).copy()
        
    # filter datasets to have 6 <= |V| <= 25 nodes
    if (graphs[i].get_adjacency_matrix().shape[0] >= 6 and  graphs[i].get_adjacency_matrix().shape[0] <= 25):
      # for GBS kernel return adj matrix not GraKel graph object  
      filteredGraphs.append(graphs[i]) 
      filteredLabels.append(y[i])
    
  maxNodes = max([g.get_adjacency_matrix().shape[0] for g in filteredGraphs]) # get maximum graph size of dataset
  # return a tuple: index 0 is the preprocessed graphs, index 1 is their labels
  return(filteredGraphs, filteredLabels, maxNodes)

"""
Function to generate feature vectors, to avoid OUT_OF_MEMORY error on computing cluster we generate 1/10 the
desired number of samples in a for loop and run it 10 times each times freeing the memory by deleting the original samples
"""
def calcFeatureVector(adjMatrix, numSamples, meanN, displacement, maxPhotons, maxNodes):
  
  Q = gen_Qmat_from_graph(adjMatrix, meanN)
  V = Covmat(Q, hbar=2)
  modes = V.shape[0]
  d = np.full( shape=(modes), fill_value=displacement, dtype=np.float64 )
  M = maxNodes 
  N = numSamples//20 # 1/20 total number of samples to avoid OUT_OF_MEMORY error
  v = [] # feature vector

  for i in range(maxPhotons+1): #create vector of 0's of length M+1
    v.append(0)
    
  for i in range(20):
    # use the walrus module to generate GBS samples b/c SF does not support adding displacement
    samples = torontonian_sample_state(cov=V, mu=d, samples=N, max_photons=maxPhotons, parallel=True) 
   
    for L in samples: #for each sample add up all photons seen in each mode
      numClicks = sum(L)
      v[numClicks] += 1
    
  for i in range(len(v)): #divide each entry by total number of samples to get probability
      v[i] /= numSamples

  return(v)


#read in command line args
numSamples = int(sys.argv[1])
avgPhotons = float(sys.argv[2])
displacement = float(sys.argv[3])
maxPhotons = int(sys.argv[4])
datasetIndex = int(sys.argv[5])


datasetNames = ["AIDS", "BZR_MD", "COX2_MD", "ENZYMES", "ER_MD", \
                "IMDB-BINARY", "MUTAG", "NCI1", "PROTEINS", "PTC_FM", "FINGERPRINT", "IMDB-MULTI"]

dataset = datasetNames[datasetIndex]

X, y, maxNodes = preprocessDataset(dataset)

table = [ [ "Data set", dataset ] ]
table.append( [ "Number of\n Graphs" ] ) # row 1 ot table
table.append( [ "Mean Number\n of Edges" ] ) # row 2
table.append( [ "Mean Number\n of Vertices" ] )
table.append( [ "Number of\n Classes" ] )


avgNodes = sum([ g.get_adjacency_matrix().shape[0] for g in X ]) / len(X) 
avgEdges = sum([ len(g.get_edges())/2 for g in X ]) / len(X)
numClasses = len(set(y))
table[1].append( len(X) )  
table[2].append(avgEdges)
table[3].append(avgNodes)
table[4].append(numClasses)

table = np.array(table).T.tolist()

print(tabulate(table, headers='firstrow', tablefmt='simple'), flush=True)


paramGrid = [{'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0 ,1e2, 1e3]}]
accuracies = []
featureVectors = []

print("\nNumber of samples per graph:", numSamples,"\nMean number of photons:", \
avgPhotons, "\nDisplacement on each mode:", displacement, "\nMax photon number:", maxPhotons, "\nSize |V| of largest graph:", maxNodes, flush=True)   
    
# loop to generate feature vector for each graph
for g in X:

  v = calcFeatureVector(g.get_adjacency_matrix(), numSamples, avgPhotons, displacement, maxPhotons, maxNodes) 
    
  featureVectors.append(v)
  
  X = np.asarray(featureVectors) # X is now a np array of feature vectors instead of a list of graph objects 
  y = np.asarray(y) 

for i in range(10): # 10 repeats of double cross validation
  seed = i
  kfold1 = KFold(n_splits=10, shuffle=True, random_state=None)
  classifier = make_pipeline(
       GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=paramGrid,
                      scoring="accuracy", cv=kfold1))

  kfold2 = KFold(n_splits=10, shuffle=True, random_state=None)

  scores = cross_validate(classifier, X=X, y=y, cv=kfold2, n_jobs=-1, return_train_score=False)

  # Get best output of this fold
  accuracies.append(np.mean(scores['test_score']))

  # Mean and std of this fold
  testMean = np.mean(scores['test_score'])
  testStd = np.std(scores['test_score'])


results = np.array(accuracies)

table = [ [ "Data set", datasetNames[datasetIndex] ] ] #reset table
table.append( ["SVM"] ) #RF standard deviation column
table[1].append( str( '%2.2f'% (100*np.mean(results) ) ) + " \u00B1 " + str( '%2.2f'% (100*np.std(results) ) ) )
print(tabulate(table, headers='firstrow', tablefmt='simple'))

import networkx as nx
import csv
import numpy as np
from random import randint
from sklearn.linear_model import LogisticRegression

# Create a graph
G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# Read the abstract of each paper
abstracts = dict()
with open('abstracts.txt', 'r') as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract

# Map text to set of terms
for node in abstracts:
    abstracts[node] = set(abstracts[node].split())

# Create the training matrix. Each row corresponds to a pair of nodes and
# its class label is 1 if it corresponds to an edge and 0, otherwise.
# Use the following 3 features for each pair of nodes:
# (1) sum of number of unique terms of the two nodes' abstracts
# (2) absolute value of difference of number of unique terms of the two nodes' abstracts
# (3) number of common terms between the abstracts of the two nodes
X_train = np.zeros((2*m, 3))
y_train = np.zeros(2*m)
n = G.number_of_nodes()
for i,edge in enumerate(G.edges()):
    # an edge
    X_train[2*i,0] = len(abstracts[edge[0]]) + len(abstracts[edge[1]])
    X_train[2*i,1] = abs(len(abstracts[edge[0]]) - len(abstracts[edge[1]]))
    X_train[2*i,2] = len(abstracts[edge[0]].intersection(abstracts[edge[1]]))
    y_train[2*i] = 1

    # a randomly generated pair of nodes
    n1 = randint(0, n-1)
    n2 = randint(0, n-1)
    X_train[2*i+1,0] = len(abstracts[n1]) + len(abstracts[n2])
    X_train[2*i+1,1] = abs(len(abstracts[n1]) - len(abstracts[n2]))
    X_train[2*i+1,2] = len(abstracts[n1].intersection(abstracts[n2]))
    y_train[2*i+1] = 0

print('Size of training matrix:', X_train.shape)

# Read test data. Each sample is a pair of nodes
node_pairs = list()
with open('test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))

# Create the test matrix. Use the same 4 features as above
X_test = np.zeros((len(node_pairs), 3))
for i,node_pair in enumerate(node_pairs):
    X_test[i,0] = len(abstracts[node_pair[0]]) + len(abstracts[node_pair[1]])
    X_test[i,1] = abs(len(abstracts[node_pair[0]]) - len(abstracts[node_pair[1]]))
    X_test[i,2] = len(abstracts[node_pair[0]].intersection(abstracts[node_pair[1]]))

print('Size of training matrix:', X_test.shape)

# Use logistic regression to predict if two nodes are linked by an edge
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_pred = y_pred[:,1]

# Write predictions to a file
predictions = zip(range(len(y_pred)), y_pred)
with open("submission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row)
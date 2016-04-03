#!/usr/bin/python
# -*- coding: utf8 -*-
"""
LoOP: Local Outlier Probabilities
Gloss: Global - Local Outliers in SubSpaces
~~~~~~~~~~~~

This module implements the Local Outlier Probabilities algorithm as well as an extra module for the implementation of Gloss.

:copyright: (c) 2016 Bas van Stein
:license: MIT, see LICENSE for more details.

"""
from __future__ import division
import math
import numpy as np
import scipy
from scipy.spatial.distance import cityblock, euclidean, braycurtis, chebyshev, jaccard
from sklearn.neighbors import NearestNeighbors

def l1(instance1,instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    dist = cityblock(instance1.reshape(1, -1),instance2.reshape(1, -1))
    return dist

def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be tuples of equal length.
    Returns: Euclidean distance
    Signature: ((attr_1_1, attr_1_2, ...), (attr_2_1, attr_2_2, ...)) -> float"""
    def detect_value_type(attribute):
        """Detects the value type (number or non-number).
        Returns: (value type, value casted as detected type)
        Signature: value -> (str or float type, str or float value)"""
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute
    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x**2, differences)) / len(differences))**0.5
    return rmse

class LoOP:
    """Helper class for performing LoOP computations and instances normalization."""
    def __init__(self, instances, normalize=True, distance_function=l1):
        self.instances = tuple(map(tuple, instances))
        self.normalize = normalize
        self.distance_function = distance_function
        if normalize:
            self.normalize_instances()

    def compute_instance_attribute_bounds(self):
        min_values = [float("inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * n.inf 
        max_values = [float("-inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * -1 * n.inf
        for instance in self.instances:
            min_values = tuple(map(lambda x,y: min(x,y), min_values,instance)) #n.minimum(min_values, instance)
            max_values = tuple(map(lambda x,y: max(x,y), max_values,instance)) #n.maximum(max_values, instance)
        self.max_attribute_values = max_values
        self.min_attribute_values = min_values
            
    def normalize_instances(self):
        """Normalizes the instances and stores the information for rescaling new instances."""
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()
        new_instances = []
        for instance in self.instances:
            new_instances.append(self.normalize_instance(instance)) # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances
        
    def normalize_instance(self, instance):
        return tuple(map(lambda value,max,min: (value-min)/(max-min) if max-min > 0 else 0, 
                         instance, self.max_attribute_values, self.min_attribute_values))
    def reconstruct_instance(self, instance):
        return tuple(map(lambda value,max,min: (value)*(max-min)+min if max-min > 0 else 0, 
                         instance, self.max_attribute_values, self.min_attribute_values))
        
    def get_instances(self):
        return self.instances
    def get_neighbours(self):
        return self.neighbours
    def get_pdist(self):
        return self.pdists

    #for this function to work first call local_outlier_probabilities with all features
    def local_outlier_search(self, L=3, k=20, verbose=False,feature_start=0, feature_end=-1 ):
        pdists = {}
        pdistarray = []
        if (feature_end == -1):
            feature_end = len(self.instances[0]) #take full length
        self.instances_temp = np.array(self.instances)[:,feature_start:feature_end] #split on features
        #check if self.neighbours is defined
        if (len(self.neighbours) < 1):
            print "Error: first run local_outlier_probabilities on the full feature set"
            return []
        
        for i in range(len(self.instances)):
            o = self.instances[i]
            if (verbose):
                print "o:",o
            S = self.neighbours[o]
            S_local = np.array(S)[:,feature_start:feature_end]
            o_local = self.instances_temp[i]
            if (verbose):
                print "S_local",S_local
            pdists[o] = pdist(L,o_local,S_local,self.distance_function)
            pdistarray.append(pdists[o])
            if (verbose):
                print "pdists[o]",pdists[o]
        self.pdists = pdistarray
        #now we have the pdist of every instance
        PLOFS = []
        for i in range(len(self.instances)):
            o = self.instances[i]

            pdistS = 0
            for s in self.neighbours[o]:
                pdistS += pdists[s]
            pdistS = pdistS / float(len(self.neighbours[o]))
            pdistS = max(pdistS,0.00001)
            PLOFS.append(pdists[o] / pdistS - 1)
            if (verbose):
                print "PLOFS[o]",PLOFS[-1]
        PLOFS = np.array(PLOFS)
        nPLOF = L * PLOFS.std()
        if (verbose):
            print "nPLOF",nPLOF


        LOOPS = []
        for i in range(len(self.instances)):
            o = self.instances[i]
            LOOPS.append(max(0,scipy.special.erf( PLOFS[i]/ (nPLOF*math.sqrt(2))  )))
        return LOOPS


    
    def local_outlier_probabilities(self, L=3, k=20, verbose=False, feature_end=-1):
        if (feature_end == -1):
            feature_end = len(self.instances[0]) #take full length
        pdists = {}
        pdistarray = []
        self.neighbours = {}

        #kNN
        distances, indices = knn(self.instances, k)

        for i in range(len(self.instances)):
            o = self.instances[i]
            S = []
            for oi in indices[i]:
                S.append(self.instances[oi]) #add k Neirest neighbours
            #print len(S)
            #if (verbose):
            #    print "o:",o
            #distances = {}
            #for instance2 in self.instances:
            #    if (o != instance2):
            #        distance_value = self.distance_function(o[:feature_end], instance2[:feature_end])
            #        if distance_value in distances:
            #            distances[distance_value].append(instance2)
            #        else:
            #            distances[distance_value] = [instance2]
            #distances = sorted(distances.items())
            #S = []
            #[S.extend(n[1]) for n in distances[:k]]
            if (verbose):
                print "S",S
            self.neighbours[o] = S
            pdists[o] = pdist(L,o,S,self.distance_function,feature_end)
            pdistarray.append(pdists[o])
            if (verbose):
                print "pdists[o]",pdists[o]
        #now we have the pdist of every instance
        self.pdists = pdistarray
        PLOFS = []
        for i in range(len(self.instances)):
            o = self.instances[i]

            pdistS = 0
            for s in self.neighbours[o]:
                pdistS += pdists[s]
            pdistS = pdistS / float(len(self.neighbours[o]))
            pdistS = max(pdistS,0.00001)
            PLOFS.append(pdists[o] / pdistS - 1)
            if (verbose):
                print "PLOFS[o]",PLOFS[-1]
        PLOFS = np.array(PLOFS)
        nPLOF = L * PLOFS.std()
        if (verbose):
            print "nPLOF",nPLOF


        LOOPS = []
        for i in range(len(self.instances)):
            o = self.instances[i]
            LOOPS.append(max(0,scipy.special.erf( PLOFS[i]/ (nPLOF*math.sqrt(2))  )))
        return LOOPS

            

def knn(df,k):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(df)
    distances, indices = nbrs.kneighbors(df)
    return distances, indices



# lambda times the standard distance of the objects in S to o
def pdist(L, o, S, distance_function=distance_euclidean, feature_end=-1):
    if (feature_end == -1):
        feature_end = len(o) #take full length
    o = np.array(o)
    #pdist = \lambda * \sigma(p,S) = \sqrt{ \frac{\sum_{s \in S} d(p,s)^2 }{\left|S\right|}  }
    totaldist = 0
    for s in S:
        s = np.array(s)
        d = distance_function(o[:feature_end], s[:feature_end])
        totaldist += d**2
    totaldist = totaldist / float(len(S))
    return L * math.sqrt(totaldist)



# Michael Tripp
# 5/1/2023 | CS614
# Lab 3: Unsupervised Learning
#
# Program to implement 2 types of unsupervised learning
# algorithms, agglomerative clustering and DBSCAN. See 
# readme.txt for more information on how these are implemented
# outside of the code comments.

import pprint
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

'''
Function that calculates the euclidian distance
between two points p1 and p2
'''
def euclidian_distance(p1, p2):
    if len(p1) != len(p2):
        print("Error: p1 and p2 are not the same dimension")
    
    summation = 0
    n = len(p1)
    
    for i in range(0, n):
        summation += (float(p1[i]) - float(p2[i]))**2
    
    return math.sqrt(summation)

'''
Method used to calculate the distance between all points in data
using a distance function d and return these distances in a list.
'''
def calc_distances(data, d):
    distances = []
    n = len(data)
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                dist = 0
            else:
                dist = d(data[i][0], data[j][0])
            row.append(dist)
        distances.append(row)
    return distances

'''
Method used to plot 4D data in a 2D plane given x_index and y_index
as the indices of which features to plot. For higher dimensional data lists.
'''
def plot_data(data, x_index, y_index, show_plot):
    x = []
    y = []
            
    for i in range(len(data)):
        x_row = []
        y_row = []
        for j in range(len(data[i])):
            x_row.append(data[i][j][x_index])
            y_row.append(data[i][j][y_index])
        x.append(x_row)
        y.append(y_row)
    
    
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    for i in range(len(data)):
        plt.scatter(x[i], y[i])

    if show_plot:
        plt.show()

'''
Method used to plot 4D data in a 2D plane given x_index and y_index
as the indices of which features to plot. For 1D data lists.
'''
def plot_data_1D(data, x_index, y_index):
    x = []
    y = []
            
    for i in range(len(data)):
        x.append(data[i][x_index])
        y.append(data[i][y_index])
    
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    plt.scatter(x, y, color='grey')
    plt.show()

'''
Single Link Agglomerative Clustering
    - cluster_map : data to be clustered
    - d : distance function
    - growth_rate : rate at which radius around each data point increases
    - x_axis/y_axis : indicate which features to plot as the axes

'''
def agglomerative_clustering(cluster_map, d, growth_rate, x_axis, y_axis):
    radius = 0.0                                
    num_clusters = len(cluster_map)             # Used to partition later at 3 clusters
    distances = calc_distances(cluster_map, d)  # Store distance data between all points once at start for use later
    n = len(cluster_map)                    
    
    partition_at_3_clusters = []
    partition_with_longest_time = []
    greatest_time = 0                   # Greatest amount of time spent at a given number of clusters
    time = 0                            # Time spent at current number of clusters
    
    while num_clusters > 1:
        radius += growth_rate           # Increase radius at each time step
        time += 1                       # Increment time to track current time step
        
        # For each point cluster_map[i], cluster points that are within radius
        for i in range(n):
            for j in range(i+1, n):
                if distances[i][j] <= radius:
                    q = i
                    r = j
                    
                    # Find what cluster each point is actually in by following the map
                    # (Since points already clustered are replaced by the cluster index they were merged into)
                    while type(cluster_map[q]) == int:
                        q = cluster_map[q]
                    while type(cluster_map[r]) == int:
                        r = cluster_map[r] 
                    if q != r:
                        '''
                        Combine two clusters into each other
                        '''
                        # Start by saving partitions according to cluster number/time spent
                        if num_clusters == 3 or time > greatest_time:
                            clusters = []
                            # Save clusters in dictionary in a list to be stored as partitions
                            for k in range(len(cluster_map)):
                                if type(cluster_map[k]) != int:
                                    clusters.append(cluster_map[k])
                            # Save partitions accordingly
                            if num_clusters == 3:
                                partition_at_3_clusters = copy.deepcopy(clusters)
                            if time > greatest_time:
                                greatest_time = time
                                partition_with_longest_time = copy.deepcopy(clusters)                            
                        time = 0
                        
                        # Combine clusters
                        for k in range(len(cluster_map[r])):
                            cluster_map[q].append(cluster_map[r][k])
                        cluster_map[r] = q
                        
                        # Finally, decrement number of clusters
                        num_clusters -= 1
    
    plt.figure(num_clusters)
    #plot_data(partition_at_3_clusters, x_axis, y_axis)
    plot_data(partition_with_longest_time, x_axis, y_axis, True)
    
'''
Method used to convert a list of indices to a list containing the actual
data entries corresponding to those indices. For higher dimensional lists.
'''
def indicies_to_data(data, data_indices):
    l = []
    for i in range(len(data_indices)):
        row = []
        for j in range(len(data_indices[i])):
            row.append(data[data_indices[i][j]])
        l.append(row)
    return l

'''
Method used to convert a list of indices to a list containing the actual
data entries corresponding to those indices. For 1D lists.
'''
def indicies_to_data2(data, data_indices):
    l = []
    for i in data_indices:
        l.append(data[i])
    return l
    
'''
Function to implement DBSCAN
    - data : data to be clustered
    - d : distance function
    - eps : search distance epsilon
    - minPts : minimum features per cluster
    - x_axis/y_axis : indicate which features to plot as the axes
'''
def DBSCAN(data, d, eps, minPts, x_axis, y_axis):
    n = len(data)
    all_Nx_indices = []
    core_pt_indices = []
    noise_indices = []
    clusters = []
    
    ''' 1. Find all core points '''
    # For all points x, find and store its corresponding Nx list
    for i in range(n):
        x = data[i]
        Nx_indices = []
        for j in range(i+1, n):
            y = data[j]
            if d(x, y) <= eps:
                Nx_indices.append(j)            # If distance x and y are within the search distance, add y to Nx
        if len(Nx_indices) > minPts:
            core_pt_indices.append(i)           # Mark x as a core point if there are enough points in Nx
        all_Nx_indices.append(Nx_indices)
        
    ''' 2. Find all core points reachable from other core points 
           and merge accordingly '''
    m = len(core_pt_indices)
    # For each core point core_pt1, find which core points are reachable and merge as necessary
    for i in range(m):
        core_pt1_index = core_pt_indices[i]
        core_pt1 = data[core_pt1_index]
        for j in range(i+1, m):
            core_pt2_index = core_pt_indices[j]
            core_pt2 = data[core_pt2_index]
            # Join both core points into the same cluster if within distance
            if d(core_pt1, core_pt2) <= eps:
                cluster_pt1 = -1    # Stores which cluster pt1 is in (-1 if not in a cluster)
                cluster_pt2 = -1    # Same for pt2
                # Find if either are already in a cluster and store which cluster they are in if so
                for k in range(len(clusters)):
                    if core_pt1_index in clusters[k]:
                        cluster_pt1 = k
                    if core_pt2_index in clusters[k]:
                        cluster_pt2 = k
                    if cluster_pt1 != -1 and cluster_pt2 != -1:
                        break
                
                if cluster_pt1 != -1 and cluster_pt1 == cluster_pt2:
                    # Both core points are already in the same cluster
                    break
                elif cluster_pt1 == -1 and cluster_pt2 == -1:
                    # Neither core point is already in a cluster, so add them both to a new cluster
                    clusters.append([core_pt1_index, core_pt2_index])
                elif cluster_pt1 != -1 and cluster_pt2 != -1:
                    # Both core points are in clusters already, so merge the smaller cluster into the bigger one
                    if len(clusters[cluster_pt1]) < len(clusters[cluster_pt2]):
                        clusters[cluster_pt2].extend(clusters[cluster_pt1])
                        clusters.pop(cluster_pt1)
                    else:
                        clusters[cluster_pt1].extend(clusters[cluster_pt2])
                        clusters.pop(cluster_pt2)
                elif cluster_pt1 != -1:
                    # Only core point 1 is in a cluster, so merge core point 2 into core point 1's cluster
                    clusters[cluster_pt1].append(core_pt2_index)
                else:
                    # Only core point 2 is in a cluster, so merge core point 1 into core point 2's cluster
                    clusters[cluster_pt2].append(core_pt1_index)
    
    # Add in remaining core points as clusters of their own
    for i in range(m):
        clustered = False
        for j in range(len(clusters)):
            # Search through all clusters and check if core point is in a cluster already
            if core_pt_indices[i] in clusters[j]:
                clustered = True
                break
        if not clustered:
            # Add core point as its own individual cluster if not in any cluster already
            clusters.append([core_pt_indices[i]])
    
    ''' 3. For each non-core point z, join into cluster or assign as noise '''
    for z in range(n):
        if z in core_pt_indices:
            continue
        is_noise = True
        for j in range(len(core_pt_indices)):
            # Check if z is in neighborhood Nx of any core point x
            x_index = core_pt_indices[j]
            Nx = all_Nx_indices[x_index]
            if z in Nx:
                # Add z to x's cluster (loop through to find what cluster is x's)
                for k in range(len(clusters)):
                    if x_index in clusters[k]:
                        clusters[k].append(z)
                        break
                is_noise = False
        if is_noise:
            noise_indices.append(z)
    
    # Plot each cluster in a different color and noise in grey
    clusters_data = indicies_to_data(data, clusters)
    plot_data(clusters_data, x_axis, y_axis, False)
    noise_data = indicies_to_data2(data, noise_indices)
    plot_data_1D(noise_data, x_axis, y_axis)
    
def plot_labeled_data(labeled_data):
    x = []
    y = []
            
    for i in range(len(labeled_data)):
        x_row = []
        y_row = []
        for j in range(len(labeled_data[i])):
            x_row.append(labeled_data[i][j][x_axis])
            y_row.append(labeled_data[i][j][y_axis])
        x.append(x_row)
        y.append(y_row)
    
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    fig, ax = plt.subplots()

    plt.scatter(x[0], y[0], label = 'Iris-setosa')
    plt.scatter(x[1], y[1], label = 'Iris-versicolor')
    plt.scatter(x[2], y[2], label = 'Iris-virginica')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    
    plt.legend(handles=handles, loc = 'lower right')
    plt.show()

if __name__ == '__main__':
    
    '''
    FILE NAME: Set file name path for data here
    '''
    fname = "C:/Users/Mctri/Workspaces/github-repos/clustering-iris-flower-data/data/iris.data"
    
    fin = open(fname)
    data_map = {}
    data_list = []
    labeled_data = [[], [], []]
    clusternum = 0
    AGGLOM_ALG = 0
    DBSCAN_ALG = 1
    PLOT_LABELS = 2
    
    # Store data from file
    for line in fin:
        if line != "" and line != "\n":
            line = ((line.strip()).split(","))
            label = line[4]
            line = line[:4]
            line = list(map(float, line))
            
            # Collect labeled data for later comparison
            if label == "Iris-setosa":
                labeled_data[0].append(line)
            elif label == "Iris-versicolor":
                labeled_data[1].append(line)
            else:
                labeled_data[2].append(line)
                
            # Store data in a list for DBSCAN and in a dictionary for agglomerative clustering
            data_list.append(line)
            data_map[clusternum] = [line]
            clusternum += 1
    
    '''
    PARAMETERS: Set specific parameters here!
    '''
    # Agglomerative clustering
    growth_rate = 0.01
    
    # DBSCAN
    search_distance = 0.6
    minPts = 10
    
    # indicate which features to plot as the axes
    x_axis = 0
    y_axis = 1
    
    ALG = DBSCAN_ALG # <-- Choose what algorithm to run
    
    if ALG == DBSCAN_ALG:
        DBSCAN(data_list, euclidian_distance, search_distance, minPts, x_axis, y_axis)
    elif ALG == AGGLOM_ALG:
        agglomerative_clustering(data_map, euclidian_distance, growth_rate, x_axis, y_axis)
    elif ALG == PLOT_LABELS:
        plot_labeled_data(labeled_data)
            
            

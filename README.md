# clustering-iris-flower-data
Michael Tripp,
5/1/2023

Utilizes various unsupervised learning clustering algorithms (Agglomerative Clustering and DBSCAN) to analyze and build models for the iris flower dataset.

Contained in the Code folder is clustering_alg.py, the program containing
my code for both agglomerative clustering and DBSCAN. There is also
a document results_discussion.pdf providing discussion and analysis of 
the algorithms implemented in this program. 

How to use: 

Simply scroll down to my main program and enter the file
name path for the data in the space provided. In the Data folder, 
you can find the data files I used for this project: iris.names 
(file containing info on how data is set up) and iris.data 
(actual data to be clustered). Further down, there is a
PARAMETERS section, where you can change the parameters for both
algorithms. Note that since the data is 4D, we specify two of the four
features to plot in order to visualize the data, which can be adjusted
in the PARAMETERS section as well.

Implementation: 

I'll note that I took two different approaches when coding these 
two algorithms in how I stored the data, so I'll explain them here 
as I think it would be difficult to understand just by my code comments. 

For agglomerative clustering, I used a dictionary to store the data 
points so that I could use the key values to figure out what cluster 
a given point is in when a merge happens. Then, when a point is merged, 
I replace the position where it was in the dictionary with the index of
the cluster where it was just merged to. This way, I can go to any 
given point's index in the dictionary and find which cluster that point
is in by following the dictionary key values. Then at the end, the
dictionary ends up with a few key values with a list of data points
as the values, while the rest are simply set to int indexes. As such,
I can then extract the clusters from this dictionary.

In DBSCAN, I wanted to try a list index based method because I wasn't
sure I was liking the dictionary based method much. Rather, in my DBSCAN
algorithm, I stored a regular 1D list of all the data points, and simply
had my algorithm keep track of indices into that list throughout the
method. This way, I could easily check if a given data point was in a cluster
by simply storing clusters as a list of the data points' indices and checking
to see if the given data point's index was in that cluster's list. This concept
is used throughout the algorithm. Then, I could simply use a method to retrieve
the data corresponding to those indices anytime by indexing into the original
data list.

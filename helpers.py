import tqdm
import urllib
import zipfile
import requests
from pathlib import Path
import skimage
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table
from scipy.ndimage import binary_fill_holes

import numpy as np
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev
from cv2 import findNonZero
###############################################################################

def download(url, fname = None):
  """
  Download the file at the given url

  Parameters
  ----------
  url : string
    The url of the file to download
  fname : pathlike, default=None
    The on-disc name of the downloaded file (if None, derived from the url)

  Returns
  -------
  The disc path to the downloaded artifact
  """

  # prepare the file name
  url = urllib.parse.urlparse(url) # parse the url to extract the file name
  p_url = Path(url.path)
  if fname is None: fname = Path(p_url.name)
  else: fname = Path(fname).with_suffix(p_url.suffix)

  # destroy the existing file prior to download if any
  fname.unlink(missing_ok = True)

  # download the file
  with open(fname, 'wb') as f: # open on-disc destination file for writing
    with requests.get(url.geturl(), stream=True) as r: # issue GET request
      r.raise_for_status()
      total = int(r.headers.get('content-length', 0)) # read total byte size
      params = { 'desc': url.geturl()
               , 'total': total
               , 'miniters': 1
               , 'unit': 'B'
               , 'unit_scale': True
               , 'unit_divisor': 1024 }
      with tqdm.tqdm(**params) as pb: # progress bar setup
        for chunk in r.iter_content(chunk_size=8192): # go through result chunks
          pb.update(len(chunk)) # update progress bar
          f.write(chunk) # write file content
  return fname

def download_and_extract(url, dest_dir = './data', fname = None):
  """
  Download and extract the file at the given url.

  Parameters
  ----------
  url : string
    The url of the file to download
  dest_dir : pathlike, default='./data'
    The path to the destination folder for download and extraction

  Returns
  -------
  None
  """
  # create destination directory if necessary, and prepare destination file name
  dest_dir = Path(dest_dir)
  dest_dir.mkdir(parents = True, exist_ok = True)
  fpath = Path(urllib.parse.urlparse(url).path)
  if fname is None: fname = fpath.name
  # download dataset
  dl = download(url, dest_dir / fname)
  suffixes = Path(dl).suffixes
  # extract dataset
  with zipfile.ZipFile(dl, 'r') as zf:
    zf.extractall(path = dest_dir)

# BBBC010_v1_foreground_eachworm dataset
#
# Sources:
# - here is where the dataset comes from:
#   https://bbbc.broadinstitute.org/BBBC010
# - here is the download link:
#   https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
#
# You can get a copy of the dataset running the following:
# $ wget https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
# $ unzip BBBC010_v1_foreground_eachworm.zip

_dataset_url = 'https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip'

################################################################################

def get_dataset(url = _dataset_url, dest_dir = './data'):
  """
  Download and extract the BBBC010_v1_foreground_eachworm dataset.

  Parameters
  ----------
  url : string
    The url for the BBBC010_v1_foreground_eachworm dataset download
  dest_dir : pathlike, default='./data'
    The path to the destination folder for download and extraction

  Returns
  -------
  None
  """
  download_and_extract(url, dest_dir)

###############################################################################

def rgb2grey(rgb, cr = 0.2989, cg = 0.5870, cb = 0.1140):
  """
  Turn an rgb array into a greyscale array using the following reduction:
    grey = cr * r + cg * g + cb * b

  Parameters
  ----------
  rgb : 3-d numpy array
    A 2-d image with 3 colour channels, red, green and blue
  cr : float, default=0.2989
    The red coefficient
  cg : float, default=0.5870
    The green coefficient
  cb : float, default=0.1140
    The blue coefficient

  Returns
  -------
  2-d numpy array
    The greyscale image
  """
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  return cr * r + cg * g + cb * b

def find_longest_contour(mask, normalise_coord=False):
  """
  Find the longest of all object contours present in the input `mask`

  Parameters
  ----------
  img : 2-d numpy array
    The image with masked objects (if it has a 3rd dimension, it is assumed to
    contain the r, g and b channels, and will be first converted to a greyscale
    image)
  normalise_coord: bool, default=False
    optionally normalise coordinates

  Returns
  -------
  2-d numpy array
    the longest contour as sequence of x, y coordinates in a column stacked
    array
  """
  # force the image to grayscale
  if len(mask.shape) == 3: # (lines, columns, number of channels)
    mask = rgb2grey(mask)
  # extract the contours from the now grayscale image
  contours = skimage.measure.find_contours(mask, 0.8)
  # sort the contours by length
  contours = sorted(contours, key=lambda x: len(x), reverse=True)
  # isolate the longest contour (first in the sorted list)
  x, y = contours[0][:, 0], contours[0][:, 1]
  # optionally normalise the coordinates in the countour
  if normalise_coord:
    x = x - np.min(x)
    x = x / np.max(x)
    y = y - np.min(y)
    y = y / np.max(y)
  # return the contour as a pair of lists of x and y coordinates
  return np.column_stack([x, y])

def contour_spline_resample(contour, n_samples, sparsity=1, per=True):
  """
  Return a resampled spline interpolation of a provided contour

  Parameters
  ----------
  contour : 2-d numpy array
    A sequence of x, y coordinates defining contour points
  n_samples : int
    The number of points to sample on the spline
  sparsity : int, default=1
    The distance (in number of gaps) to the next point to consider in the
    original contour (i.e. whether to consider every point, every other point,
    every 3 points... One would consider non-1 sparsity to avoid artifacts due
    to high point count contours over low pixel resolution images, with contours
    effectively curving around individual pixel edges)

  Returns
  -------
  2-d numpy array
    The spline-resampled contour with n_samples points as a sequence of x, y
    coordinates
  """
  # Force sparsity to be at least one
  sparsity = max(1, sparsity)
  # prepare the spline interpolation of the given contour
  sparse_contour = contour[::sparsity]
  tck, u = splprep( sparse_contour.T # reshaped contour
                  , s = 0 # XXX
                  , k = min(3, len(sparse_contour) - 1)
                  , per = per # closed contour (periodic spline)
                  )
  # how many times to sample the spline
  # last parameter is how dense is our spline, how many points.
  new_u = np.linspace(u.min(), u.max(), n_samples)
  # evaluate and return the sampled spline
  return np.column_stack(splev(new_u, tck))

#######################

# The functions euclidean_distance, compute_edges, build_graph, calculate_total_distance, DFS, PathFinder, and SingleCellLister are copied from the MEDUSSA repository https://github.com/OReyesMatte/MEDUSSA

def euclidean_distance(p1:tuple,p2:tuple)->float:
    """Function to calculate the Euclidean distance between two points in a two-dimensional space

    Args:
        p1(tuple): coordinates (x,y) of the first point
        p2(tuple): coordinates (x,y) of the second point

    Returns: 
        float: distance value between the two points
    """

    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def compute_edges(points:list,threshold:float)->list:
    
    """Convert a list of spatial coordinates to an edge list, where each point is considered to be the edge of a specific point if it's below a certain distance.
    If it is only one point, then it won't go through, as it wouldn't be a rod-shaped cell

    Args:
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        edges: list of candidate edges for each point
    """
    
    edges = []

    if len(points) == 1:
        return print("Only one point, cannot be converted to graph")

    else:

        for i in range(len(points)):
            for j in range(i+1, len(points)):

                p1 = points[i]
                p2 = points[j]
                distance = euclidean_distance(p1,p2)
                if distance <= threshold:
                    edges.append((p1,p2))

        return edges

## Build graph as list
def build_graph(points:list,edges:list)->dict:

    """Convert a list of points and edges into a graph
    
    Args:
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        graph(dict): graph representation of the skeleton, where lists the edges of each point in the skeleton
    """

    graph = {point: [] for point in points}
    
    for p1,p2 in edges:
        graph[p1].append(p2)
        graph[p2].append(p1)

    return graph

def calculate_total_distance(path:list)->float:

    """Calculate the total distance as the seum of euclidean distances between two consecutive points
    
    Args:
        path(list): list of points in (x,y) coordinate

    Return:
        total_distance(float): total calculated distance as a single float number
    """

    total_distance = 0

    for i in range(len(path) - 1):
        total_distance += euclidean_distance(path[i],path[i+1])
    return total_distance

### Depth First Search
def DFS(graph, current, visited, path, longest_path, longest_distance):

    """Run a Depth First Search algorithm 
    """

    visited.add(current)
    path.append(current)

    ## Update path
    if len(path) > len(longest_path):
        longest_path[:] = path[:]
        longest_distance[0] = calculate_total_distance(path)

    for neighbor in graph[current]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited, path, longest_path, longest_distance)

    visited.remove(current)
    path.pop

### Check for multible possible paths
def PathFinder(path:list,points:list,threshold:float)->list:

    """ It can happen that DFS computes multiple possible paths, leaving to wrong measurements. 
    For a candidate longest path, find if, indeed, is it just one path or multiple, as the latter can happen with branched skeletons
    If two successive points in the path are more than one threshold away (typically, the euclidean distance between two diagonal points), the path is assumed to split there into multiple

    Args:
        path(list): 
        points(list): list of points coordinates (x,y)
        threshold(float): minimum distance for two points to be considered edges

    Returns: 
        paths(list): computed individual paths
    """

    paths = []

    j = 0
    
    for i in range(len(path)-1):

        if euclidean_distance(path[i],path[i+1]) > threshold:
            
            subpath = path[j:(i+1)]
            paths.append(subpath)
            j = i+1
            
    return paths

def SingleCellLister(maskList:list) -> list:
    
    """From a list that contains instance segmentation images, obtain a list of individual masks

    Args:
        maskList(list): list of masks. If only one image is called, make sure to pass it as [image] for the function to run properly
    
    Returns:
        AllCells(list): list that contains a binary image of each single cell

    """

    AllCells = []

    for mask in maskList:
        reg = regionprops_table(mask,properties=['image'])
        Cells = [np.pad(binary_fill_holes(image),4) for image in reg['image']]
        AllCells += Cells

    return AllCells
##############################

def count_edges(points:np.array,threshold:float=np.sqrt(2))->int:

    n_edges = 0

    for i in range(len(points)):
    
        neighbors = 0
    
        for j in range(len(points)):
    
            distance = euclidean_distance(points[i],points[j])
    
            if distance > 0 and distance <= threshold: neighbors +=1 
    
            if neighbors > 2: break
    
        if neighbors == 1: n_edges += 1

    return n_edges
    

def mask2sampledSkel(mask:np.array, n_samples:int=64, resample_sparsity:int=1, closed:bool=False,threshold=np.sqrt(2),skeleton_percentage_threshold=0.75):
    """
    Return a resampled spline interpolation of the skeleton of a mask
    
    Parameters
    ----------
    mask : 2-d numpy array
    A binary mask of a single cell
    n_samples : int
    The number of points to sample on the spline
    sparsity : int, default=1
    The distance (in number of gaps) to the next point to consider in the
    original contour (i.e. whether to consider every point, every other point,
    every 3 points... One would consider non-1 sparsity to avoid artifacts due
    to high point count contours over low pixel resolution images, with contours
    effectively curving around individual pixel edges)
    closed : bool, default = False
    A necessary flag to return a resampled skeleton, else it will close it in undesired ways
    
    Returns
    -------
    2-d numpy array
    The spline-resampled skeleton with n_samples points as a sequence of x, y
    coordinates
    """
    
    skel = skeletonize(mask)
    points = (findNonZero(np.uint8(skel)))
    if points.shape[0] == 1:
        
        return []

    else:
        points = points.squeeze()
        points = [tuple(xy) for xy in points]
        
        ### First, check that the skeleton is not branched or enclosed at one point
        if count_edges(points,threshold) == 2:
        
            resampled = contour_spline_resample(np.array(points),n_samples,per=closed)
            
            y,x = resampled[:,0],resampled[:,1]
        
            return np.stack([x,y],axis=1)
    
        ### IF closed (edges = 0), simply remove the first point to "open" the skeleton and 
        
        ### If branched, compute candidate skeletons from the branched one, keep only the ones that cover at least 75% of the total skeleton
        else: 
            #### Compute edges and build graph
            edges = compute_edges(points=points,threshold=threshold)
            graph = build_graph(points=points,edges=edges)
    
            #### Calculate the longest path of the skeleton
            longest_path = []
            longest_distance = [0]
    
            for point in points:
                visited = set()
                DFS(graph,point,visited,[],longest_path,longest_distance)            
    
            paths = PathFinder(path=longest_path,points=points,threshold=threshold)
            
            newpaths = []
    
            for p in paths:
    
                skel_perc = len(p)/len(points)
    
                if skel_perc >= skeleton_percentage_threshold:
                    newpaths.append(p)
            
            for p in paths:
    
                skel_perc = len(p)/len(points)
            
                if skel_perc >= 0.75:
                    newpaths.append(p)
            
            resampled_ = []
            
            for p in newpaths:
            
                try:
                    resample = contour_spline_resample(np.array(p),n_samples,per=closed)
                    y,x = resample[:,0],resample[:,1]
                    resampled_.append(np.stack([x,y],axis=1))
            
                except:
                    pass   
                    
            return resampled_
    

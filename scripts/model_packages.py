import warnings
warnings.filterwarnings("ignore")

# Core
import os
import glob
import random
import pandas as pd
import numpy as np
import sys
print(sys.executable)
from pickle import dump

from  functools import reduce
import warnings
warnings.filterwarnings('ignore')

import folium

from folium.plugins import HeatMap

from scipy import spatial
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import json

from statsmodels.tsa.seasonal import seasonal_decompose


#import joblib
import matplotlib.colors as colors
import pickle
import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
from matplotlib import lines as lines
import statsmodels.formula.api as smf
import graphviz as gr
from linearmodels.datasets import wage_panel

from sklearn.cluster import KMeans

from scipy.stats import linregress

import plotly.graph_objects as go

from linearmodels.panel import PanelOLS
import lxml 
import calendar
import collections

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
from pysal.viz import splot
from splot.esda import plot_moran
import contextily
import shapely
import plotly.express as px

# Analysis and ML model building
import geopandas as gpd
import fiona 
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed
from numpy.random import randn
from numpy import percentile


from tsmoothie.smoother import *
from tsmoothie.utils_func import create_windows




from tqdm import tqdm
from scipy.interpolate import interp1d


from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data


from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error,median_absolute_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer,TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, RepeatedKFold, cross_val_score


from sklearn import metrics


from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.compose import TransformedTargetRegressor
import scipy as sp



import shap



from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats




import pingouin as pg

import osmnx as ox
from statsmodels.stats.outliers_influence import variance_inflation_factor   
#import interpret.glassbox


#import pyreadr

from factor_analyzer import FactorAnalyzer








from datetime import datetime
from meteostat import Point, Daily, Monthly, Stations




from geographiclib.geodesic import Geodesic
    
from shapely.geometry import Polygon




import folium

from folium import plugins
from folium.plugins import HeatMap



from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn import cluster

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


import time
import branca
import branca.colormap as cm

from pycaret.regression import *
#from pycaret.classification import *
from pycaret.regression import load_model 


# Confirm Pycaret version is 2.1
from pycaret.utils import version
print('Confirm Pycaret version is ?')
print('Pycaret Version: ', version())

import pathlib

from itertools import cycle

import dataframe_image as dfi
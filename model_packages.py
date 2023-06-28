import warnings
warnings.filterwarnings("ignore")


import os
import glob
import random
import pandas as pd

import numpy as np
from numpy.random import seed
from numpy.random import randn
from numpy import percentile

import sys
print(sys.executable)
import pickle
from pickle import dump

from  functools import reduce
import warnings
warnings.filterwarnings('ignore')



from matplotlib import cm
from matplotlib import lines as lines
import matplotlib.colors as colors
from matplotlib import style
from matplotlib import pyplot as plt



import folium
from folium.plugins import HeatMap
from folium import plugins

import scipy as sp
from scipy import spatial
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go



from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor   

import graphviz as gr

from linearmodels.datasets import wage_panel
from linearmodels.panel import PanelOLS

from sklearn.cluster import KMeans
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
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import lxml 
import calendar
import collections

# Graphics
import seaborn as sns

from pysal.viz import splot
from splot.esda import plot_moran
from pysal.explore import esda
from pysal.lib import weights

import contextily

import shapely
from shapely.geometry import Polygon

# Analysis and ML model building
import geopandas as gpd

import fiona 

from tsmoothie.smoother import *
from tsmoothie.utils_func import create_windows
from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data

from tqdm import tqdm

import shap

import pingouin as pg

import osmnx as ox

from factor_analyzer import FactorAnalyzer

from datetime import datetime

from meteostat import Point, Daily, Monthly, Stations

from geographiclib.geodesic import Geodesic

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
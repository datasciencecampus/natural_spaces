"""Land Type Classification.

Gather area statistics for rural-urban classification around counter sites 
and assign clusters using density based clustering.
"""



from model_config import *
from model_packages import *
from model_utils import *

# constants

# load rural/urban features from raw census data
dataset = pd.read_pickle(data_folder+'raw_census_features_with_counter_info.pkl')
dataset = dataset.groupby(['counter', 'urban_rural'])['area_sq_km'].sum().reset_index()

# load counter locations _data
locations_buffer = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_deg)

# This mapping has to be determined manually and will need reviewing as more counter dat is input
urbn_url_clstr_map = dict(zip([2,1,0,-1], ['rural_settings','major_urban_settings','minor_urban_settings','mixed_settings']))



def load_data():
  locations_buffer = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_deg)
  sites_profile = locations_buffer[['counter','geometry']].merge(dataset, on=['counter'], how='inner')
  sites_profile['geometry'] = sites_profile['geometry'].centroid
  sites_profile = sites_profile[['counter','geometry','urban_rural','area_sq_km']]
  
  return sites_profile

def preprocess(sites_profile):
  """Preprocesses data for clustering."""

  sites_profile_pv = sites_profile.pivot_table('area_sq_km', ['counter'], 'urban_rural')
  sites_profile_pv.reset_index(drop=False, inplace=True)
  
  colm_nams = sites_profile_pv.columns
  sites_profile_pv = sites_profile_pv.reindex(colm_nams, axis=1).fillna(0)
  sites_profile_pv = sites_profile_pv.rename_axis(None, axis=1)

  sites_profile = sites_profile_pv.copy()
  sites_profile = locations_buffer[['counter','geometry']].merge(sites_profile, on=['counter'], how='inner')
  sites_profile['geometry'] = sites_profile['geometry'].centroid

  coordinates = sites_profile.select_dtypes(include=np.number).values

  return coordinates, sites_profile

def cluster(coordinates):
  """Performs clustering to assign labels."""

  np.random.seed(8)
  labels = HDBSCAN(min_cluster_size=7).fit(coordinates).labels_

  return labels

def assign_labels(sites_profile, labels):
  """Assigns interpretable labels based on clustering."""

  sites_profile['land_type_labels'] = labels

  sites_profile['land_type_labels'] = sites_profile['land_type_labels'].map(urbn_url_clstr_map)
  
  return sites_profile

def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter')
    static.to_pickle(data_folder+'static_data.pkl')



def main():
  
  # load rural/urban features from raw census data
  dataset = pd.read_pickle(data_folder+'raw_census_features_with_counter_info.pkl')
  dataset = dataset.groupby(['counter', 'urban_rural'])['area_sq_km'].sum().reset_index()

  # load counter locations _data
  locations_buffer = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_deg)


  # merge counter locations with rural/urban features from raw census data
  sites_profile = load_data()

  # create coordinates for clustering and format df containing rural/urban features 
  coordinates, sites_profile = preprocess(sites_profile)

  # perform clustering to assign labels
  labels = cluster(coordinates)

  # use label mapping to give meaning to clustering assigned lables
  sites_profile = assign_labels(sites_profile, labels)

  # save all data 
  sites_profile.to_pickle(data_folder+'/rural_urban_clusters.pkl')

  # add needed columns to static data set
  add_to_static_data(sites_profile[['counter','land_type_labels']])


if __name__ == "__main__":
  main()

# ftrs_names_ur_rural=[ x for x in sites_profile.select_dtypes(include=np.number).columns if x not in ['land_type_labels']]

# # Compute the mean area in each of the rural-urban class 
# # falling under each cluster
# df=sites_profile.groupby(['land_type_labels'])[ftrs_names_ur_rural].mean().unstack().\
# reset_index().sort_values(by='land_type_labels')

# df.rename(columns={'level_0':'urban_rural',0:'area_sq_km'},inplace=True)

# df2 = df.groupby(["urban_rural","land_type_labels"]).sum().unstack("urban_rural").fillna(0)



# df2['area_sq_km'].plot.barh(stacked=True,colormap='Paired',figsize=(15,10))
# plt.ylabel('Rural-Urban Classification', fontsize=16)
# plt.xlabel('Area in Km$^2$', fontsize=16)
# plt.title('Land Type Clusters', fontsize = 16)
# plt.legend(fontsize=18, title= 'Land Type', title_fontsize='large', frameon=1)

# plt.savefig(f"./outputs/land_type_class_.png", format= 'png', dpi=300, bbox_inches='tight')


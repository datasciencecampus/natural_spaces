

"""Refactored land habitat classification module."""

from model_config import *
from model_packages import *
from model_utils import *

# load counter locations _data
locations_buffer = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_mtr)

world = gpd.read_file(world_boundaries)
uk = world[world.name == 'U.K. of Great Britain and Northern Ireland']

# constants

# This mapping has to be determined manually and will need reviewing as more counter data is input
habt_clstr_map=dict(zip([-1,0,1],['Grassland_woodland_wetland','Grassland_woodland',
                                  'Grassland_woodland_bareground']))

def read_shapefiles(data_loc):
    """Read shapefiles from given location."""
    file_loc_shp = [x for x in os.listdir(data_loc+'data') if x.split('.')[1]=='shp']
    return [gpd.read_file(data_loc+'data/'+x) for x in file_loc_shp]

def process_habitat_data():
    """Process habitat classification data."""
    #   data_loc = data_folder+'NE_LivingEnglandHabitatMapPhase4_SHP_Full/'

    #   habitat_df = pd.concat(read_shapefiles(data_loc)).reset_index(drop=True)
    #   habitat_df.to_pickle('data/ne_living_habitat.pkl')

    habitat_df = pd.read_pickle(data_loc_ne_habitat)

    high_prob_df = habitat_df[habitat_df['A_prob']>=habitat_df['A_prob'].mean()][['A_pred','A_prob','geometry']].reset_index(drop=True)

    return high_prob_df

def get_site_buffers():
    """Get buffer zones for site points."""
    sites_df = locations_buffer.copy()

    sites_df = sites_df[[x for x in sites_df.columns if x not in ['area']]].overlay(uk.to_crs(crs_mtr), how='intersection')

    return sites_df

def intersect_with_habitat(sites_df, habitat_df):
    """Intersect sites with habitat data."""
    sites_habitat_df = sites_df.overlay(habitat_df.to_crs(crs_mtr), how='intersection')

    sites_habitat_df['area_habitat_sq_km'] = sites_habitat_df.geometry.area/10**6

    # map to reduced number of habitat types  
    sites_habitat_df['A_pred'] = sites_habitat_df['A_pred'].map(habitat_dict)

    return sites_habitat_df

def calculate_habitat_areas(sites_habitat_df):
    """Calculate habitat areas for each site."""
    # sum of areas by counter location and habitat type
    sites_habitat_cover = sites_habitat_df.groupby(['counter','A_pred'])['area_habitat_sq_km'].sum().reset_index()
    # assign a primary habitat type to each people counter location 
    sites_habitat_cover.rename(columns={'A_pred':'primary_habitat'}, inplace=True)

    # create a pivot table to show the habitat make up, in terms of area,  of each buffer zone 
    sites_pivot_df = sites_habitat_cover.pivot_table('area_habitat_sq_km', ['counter'], 'primary_habitat')

    sites_pivot_df = sites_pivot_df.fillna(0)
    sites_pivot_df.rename_axis(None, axis=1, inplace=True)

    sites_pivot_df.columns = ['primary_habitat_'+x.replace(",","").strip().replace(" ","_") for x in sites_pivot_df.columns]

    sites_pivot_df = sites_pivot_df.reset_index()

    sites_df_habitat_cover_pv = locations_buffer[['counter','geometry']].merge(sites_pivot_df, on=['counter'])

    return sites_df_habitat_cover_pv

def get_cluster_labels(coordinates):
    """Cluster sites based on habitat areas."""
    np.random.seed(8)
    labels = HDBSCAN(min_cluster_size=9).fit(coordinates).labels_

    return labels

def assign_labels(sites_profile, labels):
    """Assigns interpretable labels based on clustering."""

    sites_profile['land_habitat_labels'] = labels

    sites_profile['land_habitat_labels'] = sites_profile['land_habitat_labels'].map(habt_clstr_map)

    return sites_profile

def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter')
    static.to_pickle(data_folder+'static_data.pkl')

def main():
    


    habitat_df = process_habitat_data()

    sites_df = get_site_buffers()

    sites_habitat_df = intersect_with_habitat(sites_df, habitat_df)

    sites_df_habitat_cover_pv = calculate_habitat_areas(sites_habitat_df)

    coordinates = sites_df_habitat_cover_pv.select_dtypes(include=np.number).values

    labels = get_cluster_labels(coordinates)

    sites_df_habitat_cover_pv= assign_labels(sites_df_habitat_cover_pv, labels)

    sites_df_habitat_cover_pv.to_pickle(data_folder+'land_habitat_clusters.pkl')

    add_to_static_data(sites_df_habitat_cover_pv[['counter','land_habitat_labels']])

    return sites_df_habitat_cover_pv 



if __name__ == '__main__':
  sites_df_habitat_cover_pv= main()

# ftrs_names_habitat=[x for x in sites_df_habitat_cover_pv.select_dtypes(include=np.number).columns \
#                     if x not in ['labels']]

# # Visualisation

# # Looking at area make-up of each cluster: 
# # and now assigning a'physical label' for each cluster
# df=sites_df_habitat_cover_pv.groupby(['land_habitat_labels'])[ftrs_names_habitat].mean().unstack().\
# reset_index().sort_values(by='land_habitat_labels')

# df.rename(columns={'level_0':'habitat',0:'area_sq_km'},inplace=True)

# df2 = df.groupby(["habitat","land_habitat_labels"]).mean().unstack("habitat").fillna(0)

# df2['area_sq_km'].plot.barh(stacked=True,colormap='Paired',figsize=(10, 5))


# plt.ylabel('Land Habitat Classification', fontsize=16)
# plt.yticks(fontsize=14, )
# plt.xlabel('Area in Km$^2$', fontsize=16)

# plt.legend(fontsize=14, title= 'Habitat Type', title_fontsize='large', loc='upper right')

# plt.title('Land Type Clusters', fontsize = 16)
# plt.legend(fontsize=12, title= 'Land Type', title_fontsize='large', frameon=1)

# plt.savefig(f"./outputs/land_habitat_class_.png", format= 'png', dpi=300, bbox_inches='tight')


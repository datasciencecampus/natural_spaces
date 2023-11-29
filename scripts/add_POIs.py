

from model_config import *
from model_packages import *
from model_utils import *

# constants
scng_dist_pois = 5000  # Example scanning distance
#load counter locations _data
df_loc = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg')


# # POIs to use in count_pois function
# pois=["amenity","tourism","highway"]

sel_colms = ["element_type", "osmid", "amenity", "geometry", "tourism", "highway"]
desired_columns = ["element_type", "osmid", "amenity", "geometry", "tourism", "highway"]

def coord_lister(geom):
    """Extracts coordinates from a GeoDataFrame geometry.
    
    Parameters:
    geom (GeoDataFrame.geometry): Geometry column of a GeoDataFrame.

    Returns:
    list: Coordinates extracted from the geometry.
    """
    return [geom.centroid.x, geom.centroid.y]


def process_single_provider(df_loc, provider):
    """Processes POI data for a single provider and saves the consolidated data."""
    provider_df = df_loc[df_loc['provider'] == provider]
    return get_pois(provider_df, provider)

def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter', how='inner')
    static.to_pickle(data_folder+'static_data.pkl')

def count_pois(df_poi,poi_col):

    """

        
    ----------
        df_poi : Pandas Dataframe
            
        poi_col : 
    Returns
    -------
        df_poi_cnt: Pandas Dataframe
            
            
    """
    counts = df_poi.groupby(poi_col)[poi_col].count()
    df_poi_cnt_dict = {
        poi_col: counts.index.tolist(),
        poi_col + '_count': counts.tolist()
    }
    df_poi_cnt = pd.DataFrame(df_poi_cnt_dict).set_index(poi_col).T.reset_index(drop=True)

    # Create a renaming dictionary for the transposed DataFrame
    renm_dict = {col: f'{poi_col}_{col}' for col in df_poi_cnt.columns if col != 'index'}

    # Rename the columns in df_poi_cnt using the renaming dictionary
    df_poi_cnt.rename(columns=renm_dict, inplace=True)

    # renm_dict=dict(zip(list(df_poi_cnt.columns),[poi_col+'_' +x for x in list(df_poi_cnt.columns)]))
    
    # df_poi_cnt.columns.name = None
    
    # df_poi_cnt.rename(columns=renm_dict,inplace=True)
    
    
    return df_poi_cnt



def get_pois(df_loc,fl_nam):

    """
    Creates a picle file containing all POIs in buffer zones surrounding people counter locations.

    Parameters
    ----------
    df_loc : GeoDatafram 
        The locations of people counters
        
    fl_nam : str
         Prefix to add to file name when saving file 

    Returns
    ----------
    all_sites_pois_df : 
        Pickle file of POIs in buffer zones surrounding all people counter locations
    """ 
    
    ox.config(log_console=True, use_cache=True)
    
    amnty_lstt=['bar','beer_garden','bus_station','cafe','coach_parking','food_court','holiday_park','parking',\
    'restaurant','taxi_station','toilets']
    
    tags = {'amenity': amnty_lstt, 'tourism': ['camp_site', 'guest_house','hotel', 'picnic_site'],\
            'highway':['bus_stop']}
    
    coordnts = list(df_loc.geometry.apply(coord_lister))
    
    str_pois_df=[]
    
    for posn in range(len(df_loc)):
        
        gdf = ox.geometries.geometries_from_point(reversed(coordnts[posn]),dist=scng_dist_pois,\
                                                  tags=tags).reset_index()
        
        if gdf.empty:
            
            # Dictionary with zeros for each POI category
            zero_data = {f'{poi}_count': 0 for poi in ['amenity', 'tourism', 'highway']}
            zero_data['counter'] = [df_loc['counter'].iloc[posn]]
            tmp_pois_df = pd.DataFrame(zero_data)
            
        else:
            # Desired columns with string formatting
            desired_columns = ["element_type", "osmid", "amenity", "geometry", "tourism", "highway"]

            # Filter to include only those columns that are present in gdf
            available_columns = [col for col in desired_columns if col in gdf.columns]

            # Use only the available columns for further processing
            gdf = gdf[available_columns].drop_duplicates()
            

            tmp_pois_df = []
            for feature in ['amenity', 'tourism', 'highway']:
                # Skip the feature if it's not present or has only null values in gdf
                if feature in gdf.columns and not gdf[feature].isnull().all():
                    tmp_df = count_pois(gdf, feature)
                    tmp_pois_df.append(tmp_df)

            # Concatenate the results if tmp_pois_df is not empty
            if tmp_pois_df:
                tmp_pois_df = pd.concat(tmp_pois_df, axis=1)
            else:
                tmp_pois_df = pd.DataFrame()  # or appropriate default DataFrame
            
            
            tmp_pois_df['counter'] = df_loc['counter'].iloc[posn]
            

        str_pois_df.append(tmp_pois_df)
    
    all_sites_pois_df=pd.concat(str_pois_df,axis=0).fillna(0)
    
    all_sites_pois_df.to_pickle(data_folder+'{}_pois_df.pkl'.format(fl_nam))

    return all_sites_pois_df


def main():

    unique_providers = df_loc['provider'].unique()
    all_provider_pois = []

    for provider in unique_providers:
        provider_pois = process_single_provider(df_loc, provider)
        all_provider_pois.append(provider_pois)

    # Combine all provider POIs into a single DataFrame
    combined_pois = pd.concat(all_provider_pois).reset_index(drop=True)
    # combined_pois= combined_pois.drop(['amenity_count', 'tourism_count', 'highway_count'], axis=1)
    combined_pois= combined_pois.merge(df_loc[['area', 'counter']],on=['counter'],how='inner')
    
    num_cols=[x for x in combined_pois.columns if x not in ['area_sq_km','counter']]

    # # Density of pois
    combined_pois[num_cols]=combined_pois[num_cols].div(combined_pois['area'],axis=0)

    del combined_pois['area']

    combined_pois.to_pickle(data_folder+'new_pois_data_all_sites.pkl')

    add_to_static_data(combined_pois)
    

if __name__ == "__main__":
    main()


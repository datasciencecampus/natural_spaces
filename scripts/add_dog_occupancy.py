

"""Process People and Nature survey data."""

from model_config import *
from model_packages import *
from model_utils import *

def load_survey_data():
  """Load and preprocess survey data."""

  df = pd.ExcelFile(survey+'PANS_Y2_Q1_Q4.xlsx') 
  df = pd.read_excel(df, 'Data')

  # Add date column
  df['Visit_Week'] = pd.to_datetime(df['Visit_Week'].astype(str),errors='coerce') 
  df = df[~df.Visit_Week.isnull()].reset_index(drop=True)
  df['Visit_Week'] = df['Visit_Week'].dt.to_period('M')
  df.rename(columns={'Visit_Week':'Date'}, inplace=True)
  
  df = df[ftrs_selection]
  df = df.dropna(subset=['No_Of_Visits','Date', 'Visit_Latitude','Visit_Longitude']).reset_index(drop=True)

  return df

def intersect_with_buffers(df):
  """Intersect survey data with buffer zones around counter sites."""
  world = gpd.read_file(world_boundaries)
  uk = world[world.name == 'U.K. of Great Britain and Northern Ireland']

  
  # load counter locations _data
  sites_df = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_mtr)
  sites = sites_df.overlay(uk.to_crs(crs_mtr), how='intersection')

  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Visit_Longitude, df.Visit_Latitude)).set_crs(crs_deg)
  gdf = gdf.to_crs(crs_mtr)

  intersect_df = gdf.sjoin(sites, how="left", op='intersects').dropna(subset=['counter'])
  intersect_df = intersect_df.to_crs(crs_deg)

  return intersect_df

def get_dog_occupancy(df):
  """Get mean dog occupancy for each site."""
  sites_df = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_mtr)
  
  df['Dog'] = df['Dog'].eq('Yes').mul(1)

  dog_occupancy = df.groupby('counter')[['No_Of_Visits', 'Dog']].mean().reset_index()
  dog_occupancy = sites_df[['counter','geometry']].merge(dog_occupancy, on=['counter'])
  dog_occupancy['geometry'] = dog_occupancy['geometry'].centroid
  
  return dog_occupancy

def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter', how='inner')
    static.to_pickle(data_folder+'static_data.pkl')




def main():
  # load counter locations _data
  sites_df = gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg').to_crs(crs_mtr)

  world = gpd.read_file(world_boundaries)
  uk = world[world.name == 'U.K. of Great Britain and Northern Ireland']

  # load PNAS curvey data
  df = load_survey_data()

  # overlap PNAS data with buffers
  intersect_df = intersect_with_buffers(df)

  # get mean dog occupancy for each site
  dog_occupancy = get_dog_occupancy(intersect_df)

  # save data
  dog_occupancy.to_pickle(data_folder+'dog_occupancy_sites.pkl')
  add_to_static_data(dog_occupancy[['counter', 'Dog']])

if __name__ == "__main__":
  main()







from model_config import *
from model_packages import *
from model_utils import *

# import json


# constants
# config JSON fil with all the input data sets and providers. Data files are saved as gpkg with CRS already identified
config_file = './scripts/config.json' 
# input datasets log file to document if and when processing has occurred
input_datasets_log= f'{data_folder}input_datasets_log.csv'
input_datasets = pd.read_csv(input_datasets_log)
# absolute value to alter coordinates by in anonymise function
rand_shift = 5000 


def get_config_file_paths(config_file):
  """Loads file dictionary from configuration file."""

  with open(config_file) as f:
    config = json.load(f)

  file_dict = config

  return file_dict

def check_input_log(file_dict, input_datasets):
  """
  Takes an input dictionary containing provider names and filepath to counter location files.
  The files are checked for previous processing and how long ago processing occured. Files
  that have not been processed before are then added to a new dictionary to continue processing.

  Args:
    file_dict: dictionary containing provider(keys) and filepath to counter data locations (values)
    input_datasets_log: csv file containing provider, filepath and timsetamp last processed 
                        for each input dataset

  Returns:
    files_to_ptocess: a dictionary possessing the same structure as the input dictionary however
                      the contents of the dictionary have been edited to remove previously processed files
  """
  files_to_process = {}

  for provider, info in file_dict.items():
    if (provider) not in input_datasets.provider:
      # File not processed yet
      files_to_process[provider] = []
      files_to_process[provider].append(info['x_y_path'])
      
    else:
      # Check timestamp
      last_processed = input_datasets.loc[input_datasets['provider']==provider].timestamp 
      if datetime.now() - pd.to_datetime(last_processed) > pd.Timedelta(hours=0):
        files_to_process[provider] = []
        files_to_process[provider].append(info['x_y_path'])
    
  return files_to_process
  



def load_counter_data(file_dictionary):
  """Loads counter data from a dictionary.

  Args:
    file_dictionary: List of file paths of counter data.

  Returns:
    Pandas DataFrame containing all counter data.
  """
  data = {}
  for name, file_path in file_dictionary.items():
      df = gpd.read_file(file_path[0])
      
      df['provider']= name
      df['counter']= df['counter'].apply(lambda x: x.replace("  "," ").replace(" ","_"))
      data[name] = df
  
  
  return data

def assign_locations(df):
  """Adds latitude, longitude, buffer, and area to counter data.

  Args:
    df: Pandas DataFrame containing counter data.

  Returns:
    df: DataFrame with location data added.
  """
  df = df.to_crs(crs_mtr)
  df['geometry'] = df.buffer(bufr_zones_mrts)
  df['area'] = df['geometry'].area / 1e6
  df['geom_type'] = '5km buffer'

    

  return df

def anonymise_data(df, rand_shift):
  """Anonymises counter lat/lon coordinates.

  Args:
    df: Pandas DataFrame containing counter location data.
    rand_shift: Amount to randomly shift coordinates.

  Returns:
    df: DataFrame with anonymized coordinates.
  """

  anonymised_df=  df.copy()
  random_angles = np.random.randint(0, 360, df.shape[0])
  rand_shifts = np.random.randint(-rand_shift, rand_shift, df.shape[0])
  
  anonymised_df['lat'] += np.sin(np.radians(random_angles)) * rand_shifts 
  anonymised_df['lon'] += np.cos(np.radians(random_angles)) * rand_shifts

  

  return anonymised_df

def update_log(provider):
    file_dict=get_config_file_paths(config_file)
    file= file_dict[provider]
    input_datasets.loc[input_datasets['provider']==provider].timestamp = datetime.now()
    # input_datasets.loc[(provider, file), 'timestamp'] = datetime.now()
    input_datasets.to_csv(data_folder+'input_datasets_log.csv')

# def process_data(config_file, input_datasets_log):
#   """Runs full data processing pipeline.

# Args:
#   config_file: JSON file containig inpout data sets used in the project
#   input_datasets_log: csv file containing provider, filepath and timsetamp last processed 
#                       for each input dataset. Index of file must be multi_index of provider, filepath

# Returns:
#   processed_df: file containing latitude, longitude, 5km buffer geometry for each counter location
#   anonymised_df: same contents as processed_df only locations have been anonymised
#   """
#   # create input dictionary from config file
#   # data = load_counter_data(files)
#   file_dict=get_config_file_paths(config_file)
  

#   # check for previous processing
#   files_to_process = check_input_log(file_dict, input_datasets)
  
#   data= load_counter_data(files_to_process)
  
#   combined_df = []
#   combined_anonymised_df=[]

#   for provider, df in data.items():
#     # Process each source data
    
#     df = assign_locations(df)  
#     combined_df.append(df)
#     anonymised_df= anonymise_data(df, rand_shift)
#     combined_anonymised_df.append(anonymised_df)
    
  
#   processed_df = pd.concat(combined_df, ignore_index=True)
#   # keep only relevant columns
#   processed_df= processed_df[['counter', 'lat', 'lon', 'geometry', 'provider', 'area','geom_type']]
#   processed_df=processed_df.to_crs('EPSG:4326')
#   processed_df.to_file(data_folder+'counter_locations_processed.gpkg', crs= 'EPSG:4326')
  
#   anonymised_df = pd.concat(combined_anonymised_df, ignore_index=True)
#   # keep only relevant columns
#   anonymised_df= anonymised_df[['counter', 'lat', 'lon', 'geometry', 'provider', 'area','geom_type']]
#   anonymised_df=anonymised_df.to_crs('EPSG:4326')
#   anonymised_df.to_file(data_folder+'counter_locations_anonymised_processed.gpkg',crs= 'EPSG:4326' )


#   # update log for data that has been processed
#   for provider in processed_df.provider:
#     update_log(provider)


#   return  processed_df, anonymised_df

def main():

    # processed_df, anonymised_df = process_data(config_file, input_datasets)
    

    # create input dictionary from config file
    # data = load_counter_data(files)
    file_dict=get_config_file_paths(config_file)
    

    # check for previous processing
    files_to_process = check_input_log(file_dict, input_datasets)
    
    data= load_counter_data(files_to_process)
    
    combined_df = []
    combined_anonymised_df=[]

    for provider, df in data.items():
      # Process each source data
      
      df = assign_locations(df)  
      combined_df.append(df)
      anonymised_df= anonymise_data(df, rand_shift)
      combined_anonymised_df.append(anonymised_df)
      
    
    processed_df = pd.concat(combined_df, ignore_index=True)
    # keep only relevant columns
    processed_df= processed_df[['counter', 'lat', 'lon', 'geometry', 'provider', 'area','geom_type']]
    processed_df=processed_df.to_crs('EPSG:4326')
    processed_df.to_file(data_folder+'counter_locations/counter_locations_processed.gpkg', crs= 'EPSG:4326')
    
    anonymised_df = pd.concat(combined_anonymised_df, ignore_index=True)
    # keep only relevant columns
    anonymised_df= anonymised_df[['counter', 'lat', 'lon', 'geometry', 'provider', 'area','geom_type']]
    anonymised_df=anonymised_df.to_crs('EPSG:4326')
    anonymised_df.to_file(data_folder+'counter_locations/counter_locations_anonymised_processed.gpkg',crs= 'EPSG:4326' )


    # update log for data that has been processed
    for provider in processed_df.provider:
      update_log(provider)


      # visualize_data(processed_df)

    return processed_df, anonymised_df

if __name__ == "__main__":
    processed_df, anonymised_df= main()


# COLORS = {
#   'ne': '#206095',
#   'ndw': '#27A0CC',
#   'crt': '#003C57',
#   "thames": "#118C7B",
#   "dorset":"#A8BD3A",
#   "notts":"#871A5B",
#   "kent":"#F66068",
#   "bow": '#746CB1'

# }

# LABEL_MAP = {
#   'ne': 'Natural England',
#   'ndw': 'North Downs Way',
#   'crt': 'Canal & River Trust',
#   "thames": "Thames Basin",
#   "dorset":"Dorset Heaths",
#   "notts":"Nottingham",
#   "kent":"Kent County Council",
#   "bow": 'Bowlands'
# }

# def visualize_data(processed_df):

#   # Create GeoDataFrame
#   gdf = gpd.GeoDataFrame(processed_df, geometry='geometry')
#   gdf.geometry=gdf.geometry.centroid
#   # Plot each provider separately
#   fig, ax = plt.subplots()
#   for provider, data in gdf.groupby('provider'):
#     label = LABEL_MAP.get(provider, provider)
#     data.plot(ax=ax, color=COLORS[provider], markersize=10, label=label)

#   # Add basemap  
#   contextily.add_basemap(ax,crs= 4326,source=contextily.providers.Esri.WorldGrayCanvas)

#   ax.set_title('People Counter Locations by Provider')
#   ax.legend(frameon=1, facecolor='white', loc='upper left')
#   ax.axis('off')
#   # fig.savefig(f"./outputs/sites_x_source_all_providers.png", format= 'png', dpi=300, bbox_inches='tight')




# visualize_data(processed_df.loc[processed_df.counter=='Banks_lane'])


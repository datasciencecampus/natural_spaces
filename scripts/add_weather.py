

from model_packages import *
from model_config import *
from model_utils import *

# Constants
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2020, 12, 31)

"""Processes weather data for people monitoring sites."""


def get_weather_data(site_locations):
  """Fetches weather data for each site from nearest weather station."""

  weather_data = []

  for index, row in site_locations.iterrows():
    # Get nearby weather stations
    stations = Stations() 
    nearby_stations = stations.nearby(row['geometry'].centroid.x, row['geometry'].centroid.y)
    closest_station = nearby_stations.fetch(1)

    point=Point(closest_station['latitude'][0],closest_station['longitude'][0])
    # Get weather data for date range
    weather_site = Monthly(point,
                           START_DATE, END_DATE)
    monthly_weather = weather_site.fetch()
    
    # Add site name and append to list
    monthly_weather['site'] = row['counter']
    weather_data.append(monthly_weather)
  weather_data= pd.concat(weather_data).reset_index(names= 'Date')
  return weather_data

def clean_weather_data(weather_df):
  """Cleans weather DataFrame by dealing with NaNs and aggregating data."""
  # Aggregate to monthly 
  weather_df['Date'] = weather_df.Date.dt.to_period('M')
  weather_df = weather_df.groupby(['site', 'Date'])['tavg'].mean().reset_index()

  # Split into list of DataFrames per site
  weather_dfs = [df for _, df in weather_df.groupby('site')]

  # Fill NaNs 
  for df in weather_dfs:
    df = df.fillna(method='ffill').fillna(method='bfill')

  # Concatenate back into single DataFrame
  weather_df = pd.concat(weather_dfs)

  return weather_df


def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter')
    static.to_pickle(data_folder+'static_data.pkl')

def main():
  """Main process to load, process, and save weather data."""

  site_locations=gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg')
  
  # Get weather data
  weather_df = get_weather_data(site_locations)  
  # Clean data
  weather_df = clean_weather_data(weather_df)

  # Save cleaned data
  weather_df.to_pickle(data_folder+'weather_data.pkl')

  

if __name__ == '__main__':
  main()


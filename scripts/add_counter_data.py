
"""Processes people counter data.

Removes outliers, imputes missing data, handles null values, and aggregates to 
monthly frequency.
"""
from model_config import *
from model_packages import *
from model_utils import *

# Constants
NULL_PCT_THRES = 50  # Percentage threshold to drop null columns.
OUTLIER_CUTOFF = 0.1  # Cutoff for removing top/bottom outliers.
NULL_PCT_MONTHLY_THRES = 10  # Threshold to drop null columns after aggregation.

def assign_lat_lon(df_loc, col_name):
  """Adds latitude/longitude from geometry and replaces whitespace in name."""
  df = df_loc.to_crs(crs=crs_deg)

  df['latitude'] = df.geometry.y  
  df['longitude'] = df.geometry.x
  
  # Replace whitespace in counter name
  df[col_name] = df[col_name].str.replace(r'\s+', '_')  

  return df
def process_columns(df):
  """Removes outliers, aggregates to monthly."""
  
  df_monthly = pd.DataFrame()
  

  for col in df:
    
    series = df[col]
    
    # Ensure series has datetime index 
    series.index = df.index
    series.index= pd.to_datetime(series.index)
    
    # Outlier removal 
    series = series.clip(*series.quantile([0.1, 0.9]))
    
    # Resample and aggregate
    monthly = series.resample('M').mean().sort_index()
    
    df_monthly[col] = monthly
    
         
  return df_monthly
def impute_missing(df):

  df_smooth = df.copy()
  up = []
  low = []
  
  for col in df:
  
    smoother = KalmanSmoother(component='level_season', n_seasons=12,\
                                  component_noise={'level':0.01,'season':0.01})
    

    smoother.smooth(df[col]) 
    
    
    df_smooth[col] = smoother.smooth_data.T
    
    # Get intervals
    lower, upper = smoother.get_intervals('kalman_interval')
    
    up.append(pd.DataFrame(lower.T).set_index(df_smooth.index))
    low.append(pd.DataFrame(upper.T).set_index(df_smooth.index))

  return df_smooth, up, low


def prepare_counter_data(data_file, cutoff_year):
  """Cleans people counter data.
  
  Args:
    data_file: file path to raw counter data file.
    cutoff_year: Furthest year back there is data for.
    
  Returns:
    DataFrame with cleaned monthly counter data.
  """

  # Load and filter by cutoff year
  df = pd.read_csv(data_file)
  df['Date'] = pd.to_datetime(df['Time']).dt.date
  df = df.sort_values('Date').drop('Time', axis=1)
  df = df[pd.to_datetime(df['Date']).dt.year >= cutoff_year]

  # Drop columns with high percentage of nulls
  null_pct = df.isnull().mean() * 100 
  df = df.drop(null_pct[null_pct > NULL_PCT_THRES].index, axis=1)
  # drop remaining NAN
  df= df.dropna()
  df= df.set_index('Date', drop=True)
  
  # Remove outliers and aggregate to monthly
  df_monthly = process_columns(df)
  
  # Drop any columns with remaining nulls
  df_monthly = df_monthly.dropna(axis=1, thresh=NULL_PCT_MONTHLY_THRES)

  # impute missing values
  df_smooth, up,low = impute_missing(df_monthly)
  
  # Remove zeros  
  df_smooth = df_smooth.loc[:, (df_smooth != 0).any(axis=0)]  

  return df, df_monthly,df_smooth, up, low
def plot_example(df_monthly, df_smooth, up, low):
  """Plots example time series before and after cleaning."""
  
  # Pick random column
  col= df_monthly.sample(1, axis= 1).columns[0]
  
  
  fig, ax = plt.subplots()

  # Plot raw
  df_monthly[col].plot(ax=ax, style='-o', label='Monthly Average People Count', legend= True) 
  
  # Plot cleaned
  df_smooth[col].plot(ax=ax, style='-*', label='Smoothed Monthly Average People Count',legend=True)
  # ax.legend(loc='upper right', frameon=1, facecolor='white')
  up[df_monthly.columns.get_loc(col)].plot(ax=ax,color='black',style=['--'], label='_up')
  low[df_monthly.columns.get_loc(col)].plot(ax=ax,color='black',style=['--'],label='_low')

  plt.title(f"Cleaned Monthly Average People Count Data for {col}")
  hand, labl = ax.get_legend_handles_labels()
  labs= ['Monthly Average People Count', 'Smoothed Monthly Average People Count', 'Range']  
  handout=[]
  lablout=[]
  for h,l in zip(hand,labl):
      if l not in lablout:
          lablout.append(l)
          handout.append(h)
  ax.legend(handout, labs, loc='upper right', frameon=1, facecolor='white')
  # ax.grid(visible=False)
  # ax.set_xlim(0,1200)
  # ax.set_xlabel('Date')
  # ax.set_ylabel('Count')
  # ax.legend()

  # plt.tight_layout()
  # plt.savefig('cleaning_example.png')

def update_python_config_with_sites_list(df, config_file_path, variable_name):
    """
    Updates a Python configuration file with the column names of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame whose column names are to be saved.
        config_file_path (str): Path to the Python configuration file.
        variable_name (str): The variable name under which the column names will be saved.
    """
    # Convert DataFrame columns to a list
    df.columns=[x.replace("  "," ").replace(" ","_") for x in df.columns]
    columns_list = df.columns.tolist()
    
    # Read the file
    with open(config_file_path, 'r') as file:
        lines = file.readlines()

    # Check if the variable already exists
    var_exists = any(line.strip().startswith(variable_name) for line in lines)

    # Update or add the variable
    new_line = f"{variable_name} = {columns_list}\n"
    if var_exists:
        for i, line in enumerate(lines):
            if line.strip().startswith(variable_name):
                lines[i] = new_line
                break
    else:
        # Ensure new line starts on a new line
        if lines and not lines[-1].endswith('\n'):
            lines.append('\n')
        lines.append(new_line)

    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        file.writelines(lines)


def main(provider, data_file, cutoff_year):
  
  # Process data
  df_raw, df_monthly,df_smooth, up, low  = prepare_counter_data(data_file, cutoff_year)
  
  # Visualize
  plot_example(df_monthly, df_smooth, up, low)

  # Save data
  df_smooth.to_pickle(f'./data/people_counter_data_processed_{provider}.pkl')

  # save site names that have data of high quality
  update_python_config_with_sites_list(df_smooth, './scripts/model_config.py', f'{provider}_high_quality_sites ')
  
  return df_raw, df_monthly, df_smooth, up, low

if __name__ == '__main__':
    main()


"""Processes stava data.

Removes outliers, imputes missing data, handles null values, and aggregates to 
monthly frequency.
"""
import model_config 
from model_packages import *
from model_utils import *

remove_suffix = '_1_edge_daily_2020-01-01-2023-09-30_ped'


def clean_folder_names(data_loc, remove_suffix):
    """
    Cleans up the names of folders obtained from Strava data by removing a specified suffix 
    and replacing spaces with underscores.

    Args:
        data_loc (str): The directory where the Strava data is located.
        remove_suffix (str): The suffix to be removed from the folder names.
    """

    file_names = [x[0] for x in os.walk(data_loc) if '_ped' in x[0]]
    for x in file_names:
        # Remove the specified suffix
        new_name = x.replace(remove_suffix, "")
        # Replace spaces with underscores
        new_name = new_name.replace(" ", "_")
        # Rename the folder
        os.rename(x, new_name)


def prepare_strava(loc_names, loc_fil_nam):
    """
    Prepares Strava data by combining individual data files into a single DataFrame.

    Args:
        loc_names (List): List containing names of people counter locations.
        loc_fil_nam (str): Path to folder containing Strava data.

    Returns:
        Pandas DataFrame: A dataframe containing combined Strava Metro data.
    """
    strava_count = []
    for site in loc_names:
        
        site_path = pathlib.Path(os.getcwd(), loc_fil_nam, site)
        
        if site_path.exists():
            print('site exists')
            shp_path = glob.glob(loc_fil_nam + site + '/*.shp')[0]
            
            csv_path = glob.glob(loc_fil_nam + site + '/*.csv')[0]
            
            shp=gpd.read_file(shp_path)

            shp=shp.drop_duplicates("edgeUID").reset_index(drop=True)

            csv=pd.read_csv(csv_path)

            csv=csv.drop_duplicates().reset_index(drop=True)

            csv_path=csv[csv['edge_uid'].isin(shp["edgeUID"])].reset_index(drop=True)
    
            csv_path['date']=pd.to_datetime(csv_path['date'])
            csv_path['date']=csv_path['date'].dt.to_period('M')
    
            csv_path_edge_month=csv_path.groupby(['date']).mean().reset_index()
    
            csv_path_edge_month['site']=site
    
            strava_count.append(csv_path_edge_month)
            
    strava_count=pd.concat(strava_count).reset_index(drop=True)  
    strava_count.drop(columns=['osm_reference_id','edge_uid'],inplace=True)
    
    strava_count.rename(columns={'date':'Date'},inplace=True)
    
    strava_count['Date']=strava_count['Date'].astype(str)
    return strava_count


def merge_strava_counter(provider, df):
    counter= pd.read_pickle(f'./data/people_counter_data_processed_{provider}.pkl')
    counter.columns=[x.replace("  "," ").replace(" ","_") for x in counter.columns]
    counter=counter.stack().reset_index()
    counter.columns = ['Date', 'site', 'people_counter_data']
    counter['Date']=counter['Date'].dt.to_period('M').astype("string")
    merged= counter.merge(df, how='inner',left_on=['Date','site'],\
                                            right_on=['Date','site'])
    
    return merged



def main(provider, data_loc):
    # print(provider, data_loc)
    data_loc= data_loc.strip("'")
    
    # create variable to access list of high quality sites in model config
    sites_var= f'{provider}_high_quality_sites'
    
    # retrieve attribute from model_config
    if hasattr(model_config, sites_var):
        
        loc_names = getattr(model_config, sites_var)
        clean_folder_names(data_loc, remove_suffix)
        # process strava data
        strava_count = prepare_strava(loc_names, data_loc)
        # merge with counter data 
        merged= merge_strava_counter(provider, strava_count)
        
        merged.to_pickle(data_folder+f'pc_and_strava_{provider}.pkl')

        return merged
    
        

    else:
        print("Invalid provider or configuration not found for the given provider.")


if __name__ == "__main__":
    merged= main()
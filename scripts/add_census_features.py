

from model_config import *
from model_packages import *
from model_utils import *

# Constants
census_locn_file_data
data_folder
counter_info= gpd.read_file(data_folder+'counter_locations/counter_locations_processed.gpkg')

household_occupancy_ftrs=['1-2 people in household','3+ people in household']
age_ftrs=['Age group 0-25','Age group 25-65','Age group 65+']
deprivation_ftrs=['Household is not deprived in any dimension','Household is deprived in at least 1 dimension']
people_density_ftrs=['Density (number of persons per sq_km)']
econominc_activity_ftrs=['Economically active', 'Economically Inactive','Unemployed_population']
health_ftrs=['Population in Good Health','Population in Bad Health']
ethnic_ftrs=['White','Asian/Asian British','Mixed/Black/others']
vehicle_ftrs=['No cars or vans in household', '1 car or van in household',\
                  '2 or more cars or vans in household']

ftrs_sbset=[household_occupancy_ftrs,age_ftrs,deprivation_ftrs,econominc_activity_ftrs,health_ftrs,ethnic_ftrs,\
                vehicle_ftrs]

grouped_features= ['3+ people in household', '1-2 people in household', 'Age group 0-25', 'Age group 25-65', 'Age group 65+', 'Household is deprived in at least 1 dimension', 
    'Household is not deprived in any dimension','Economically active', 'Economically Inactive','Unemployed_population', 'Population in Good Health', 'Population in Bad Health', 
    'White','Asian/Asian British','Mixed/Black/others','No cars or vans in household', '1 car or van in household', '2 or more cars or vans in household', 'urban_rural']

baseline_pop = ['1-2 people in household', 'Age group 65+', 'Household is not deprived in any dimension', 
                'Economically Inactive' ,'Population in Good Health', 'White', '1 car or van in household']

# process raw census data downloads
def load_shapefiles():
    """Loads Output Area shapefiles."""
    df = gpd.read_file(census_locn_file_data + '/infuse_oa_lyr_2011')
    df = df[df['geo_code'].str.lower().str.startswith('e')]
    df = df[['geo_code', 'geometry']]
    df.rename(columns={'geo_code': '2011 output area'}, inplace=True)
    return df

def load_urban_rural_data():
    """Loads urban/rural classification for Output Areas."""
    df = pd.read_csv(census_locn_file_data + 'RUC11_OA11_EW.csv', skiprows=0)
    df = df[df['OA11CD'].str.lower().str.startswith('e')]
    df = df[['OA11CD', 'RUC11']].reset_index(drop=True)
    df.rename(columns={'OA11CD': '2011 output area', 'RUC11': 'urban_rural'}, inplace=True)
    return df
    
def merge_urban_rural_shapefiles(df_shapefiles, df_urban_rural):
    """Merges urban/rural data with shapefiles."""
    return df_shapefiles.merge(df_urban_rural, on='2011 output area', how='inner').dropna().reset_index(drop=True)

def load_census_data():
    """Loads all census data."""
    dfs = []
    for filename in ['household_occupancy', 'age_groups', 'deprivation_dimension', 
                     'population_density', 'working_population', 'population_health',
                     'ethnicity', 'cars']:
        df = pd.read_csv(census_locn_file_data + filename + '.csv')
        df = df.set_index('2011 output area')
        dfs.append(df)
    return pd.concat(dfs, axis=1).reset_index()

def merge_census_ur(df_census, df_shapefiles):
    """Merges census data with shapefiles."""
    return df_census.merge(df_shapefiles, on='2011 output area', how='inner').dropna().reset_index(drop=True)

def save_data(df, filename):
    """Saves dataframe to pickle file."""
    df.to_pickle(data_folder + filename)






# functions to add census data to counter loctions.
def load_site_data():
    """Loads people monitoring site data."""
    df = gpd.read_file(data_folder + 'counter_locations/counter_locations_processed.gpkg')
    return df[['counter', 'geometry', 'provider', 'geom_type']].reset_index(drop=True)

def intersect_sites_output_areas(df_sites, df_census):
    """Intersects sites with Output Areas to get census data."""
    return df_sites.overlay(gpd.GeoDataFrame(df_census).to_crs(crs_deg), how='intersection')

def add_area_column(df):
    """Adds area column to dataframe."""
    return df.area / 10**6  








# census data processing from 8.data_augmentation

def get_area_sites(df):
    """Calculates area of buffer region around each site."""
    # Buffer zone is 5km radius around each site: pi*r^2 = 78.5 sq km
    return df.groupby('counter')['area'].sum().reset_index()

def engineer_features(df):
    """Engineers new features from existing columns."""

    # Define collective features
    df['3+ people in household'] = df[['3 people in household','4 people in household',\
                                           '5 people in household','6 people in household',\
                                           '7 people in household','8 or more people in household']].sum(axis=1) 

    df['1-2 people in household'] = df[['1 person in household','2 people in household']].sum(axis=1)
    
    # Age groups
    df['Age group 0-25'] = df[['Age 0 to 4','Age 5 to 7','Age 8 to 9', 'Age 10 to 14',\
                                   'Age 15', 'Age 16 to 17','Age 18 to 19', 'Age 20 to 24']].sum(axis=1)
    df['Age group 25-65'] = df[['Age 25 to 29', 'Age 30 to 44','Age 45 to 59', 'Age 60 to 64']].sum(axis=1) 
    df['Age group 65+'] = df[['Age 65 to 74', 'Age 75 to 84','Age 85 to 89', 'Age 90 and over']].sum(axis=1)
    
    # Deprivation
    df['Household is deprived in at least 1 dimension'] = df[['Household is deprived in 1 dimension',\
                                                                  'Household is deprived in 2 dimensions',\
                                                                  'Household is deprived in 3 dimensions',\
                                                                  'Household is deprived in 4 dimensions']].sum(axis=1)
    
    # Economic activity
    df['Unemployed_population'] = df[['Unemployed: Age 16 to 24','Unemployed: Age 50 to 74',\
                                          'Unemployed: Never worked','Long-term unemployed']].sum(axis=1)
    
    # Health 
    df['Population in Good Health'] = df[['Very good health','Good health','Fair health']].sum(axis=1)
    df['Population in Bad Health'] = df[['Bad health', 'Very bad health']].sum(axis=1)

    # Ethnicity
    df['Mixed/Black/others'] = df[['Mixed/multiple ethnic groups',\
                                       'Black/African/Caribbean/Black British',\
                                       'Other ethnic group']].sum(axis=1)

    # Vehicles
    df['2 or more cars or vans in household'] = df[['2 cars or vans in household',\
                                                        '3 cars or vans in household',\
                                                        '4 or more cars or vans in household']].sum(axis=1)

    return df

def get_prop(df, columns):
    """Gets proportion of features."""
    total = df[columns].sum(axis=1)
    print(total)
    df[columns]= df[columns].div(total, axis=0)
    print('done')
    return df[columns]

def save_static_data(df):
    """Saves dataframe to pickle file."""
    df.to_pickle(data_folder + 'static_data.pkl')

def merge_with_static(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter', how='inner')
    static.to_pickle(data_folder+'static_data.pkl')


def main():

    # Load shapefiles
    df_shapefiles = load_shapefiles()  

    # Load urban/rural data
    df_urban_rural = load_urban_rural_data()

    # Merge urban/rural with shapefiles
    df_ur = merge_urban_rural_shapefiles(df_shapefiles, df_urban_rural)
    
    # Save urban/rural shapefiles
    save_data(df_shapefiles, 'urban_rural_oa.pkl')

    # Load all census data
    df_census = load_census_data()

    # Merge census data with urban/rural
    df_census = merge_census_ur(df_census, df_ur)

    # Save census shapefiles and data
    save_data(df_census, 'census_oa_shapefiles.pkl')


    # merge census data with sites info
    # Load counter location data 
    df_sites = load_site_data()

    # Intersect sites with Output Areas
    df_sites_oa = intersect_sites_output_areas(df_sites, df_census)

    # Add area column
    df_sites_oa['area_sq_km'] = add_area_column(df_sites_oa)

    # Save site info with raw census data
    df_sites_oa.to_pickle(data_folder+'raw_census_features_with_counter_info.pkl')




    # process census data to features needed for model development
    df = df_sites_oa
    df['Density (number of persons per sq_km)']=100*df['Density (number of persons per hectare)']
    
    # groups census features into larger and more relevant caetgories e.g. 3+ people in household
    grouped_census = engineer_features(df)
    
    # Get proportion of features in each subset (e.g. proportion of 'Age group 0-25' in ['Age group 0-25','Age group 25-65'])
    for cols in ftrs_sbset:   
        grouped_census[cols]=get_proportion(grouped_census, cols)


    # Get baseline features for regression 
    baseline_features= [f for f in grouped_features if f in baseline_pop]
    baseline_features.append('counter')
    baseline__features_df= grouped_census.groupby('counter')[baseline_pop].mean(numeric_only=True).reset_index()
    # merge with counter info to get provider column for subsetting
    baseline__features_df= baseline__features_df.merge(df_sites[['counter', 'provider', 'geometry', 'geom_type']], on='counter')
    # save baseline features
    baseline__features_df.to_pickle(data_folder + 'baseline_features.pkl')

    # get features not included in baseline
    non_baseline_features = [f for f in grouped_features if f not in baseline_pop]
    non_baseline_features.append('counter')
    non_baseline_features_df= grouped_census.groupby('counter')[non_baseline_features].mean(numeric_only=True).reset_index()

    # merge with counter info to get provider column for subsetting
    non_baseline_features_df = non_baseline_features_df.merge(df_sites[['counter', 'provider', 'geometry', 'geom_type']], on='counter')
    # save features as static training data 
    non_baseline_features_df.to_pickle(data_folder + 'static_data.pkl')

if __name__ == "__main__":
    main()
  
  


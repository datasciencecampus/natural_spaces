# import packages required to run this analysis
from model_packages import *


# references to data folder structure
data_folder = "./data/"
# add strava data folder
data_folder_updated = "./data/strava_and_counter_data_updated/"

# Natural England counter locations
ne_countr_locn_file = data_folder + "counter_location_bng.gpkg"

# north downs way counter locations
ndw_countr_locn_file = data_folder + "north_downs_way_counter_location_bng.gpkg"

# River and Canals Trust counter locations
crt_countr_locn_file = (
    data_folder + "canals_and_rivers_trust_counter_locations_bng.gpkg"
)

# census data folder
census_locn_file_data = data_folder + "census/"

# census sociodemographic geactures
census_locn_file = data_folder + "census_oa_socio_economic_ftrs.pkl"

# rural-urbal cluster data
land_cluster_file = data_folder + "rural_urban_clusters.pkl"

# land habitat data
data_loc_ne_habitat = data_folder + "ne_living_habitat.pkl"

# land habitat type clusters 
habitat_cluster_file = data_folder + "land_habitat_clusters.pkl"

# people and nature survey data
nature_srvy_visits = data_folder + "nature_survey_time_series_data.pkl"

# mean dog occupancy data 
dog_ownership = data_folder + "dog_occupancy_sites.pkl"


# green infrastructure data
green_infstrcr = data_folder + "green_infrastructure.pkl"

# average monthly temperature data
weathr_data = data_folder + "weather_df.pkl"

# degrees crs
crs_deg = "EPSG:4326"

# metres crs
crs_mtr = "EPSG:27700"



# Natural England people counter location strava data
ne_strava_data_file = (
    data_folder_updated
    + "data_01-2020_to_11-2022/"
    + "natural_england_national_trails_people_counters_daily.csv"
)


# North Downs way people counter location strava data
ndw_strava_data_file = (
    data_folder_updated
    + "data_01-2020_to_11-2022/"
    + "north_downs_way_people_counters_daily.csv"
)

# cut-off year for reading counter data
cnt_ct_off_yr = 2020

cnt_ct_off_yr_ne = 2020

cnt_ct_off_yr_nd = 2021

# cut-off % to keep columns with missing values below null_prcntg(%) in the counter data set
null_prcntg = 50


null_prcntg_mnthly = 25

# whether to apply smoothening flag
apply_smtheng_flg = True

# window length to pad the sequence while applying smoothing
# The following has been chosen with monthly aggregation in mind (freq ~M)
# For different sampling frequencies, different window_len might be optimal
window_len = 4

# List of allowed smoothening functions
# lst_smthers=['exp','conv','sptrl','poly','splne','gaussn','binr','lwes','klmn']
whch_smther = "exp"

# create subdirectories for storing STRAVA metro data for Natural England counter sites
strava_data_loc = (
    data_folder_updated
    + "data_01-2020_to_11-2022/"
    + "natural_england_national_trails_strava_single_edge/"
)

# create subdirectories for storing STRAVA metro data for North Downs Way counter sites
val_strava_data_loc = (
    data_folder_updated
    + "data_01-2020_to_11-2022/"
    + "north_downs_way_strava_single_edge/"
)

# create subdirectories for storing STRAVA metro data for Canals and Rivers Trust counter sites
canal_trust_strava_data_loc = (
    data_folder_updated
    + "data_01-2020_to_11-2022/"
    + "canals_and_rivers_trust_strava_single_edge/"
)

# Which STRAVA feature to be used in modelling while building a regression
# model for Natural england counter dataset as the target variable.
chsen_ftr_strava = "total_trip_count"

# target feature for model
target = "people_counter_data"


# maximum distance to look for pois around a point
scng_dist_pois = 5000  # (in metres)

# buffer zones to count the number of recreational sites
buffr_zone_dist_rcrtnl = 0.1
# buffer zones in metres 1000m ~ 1km
bufr_zones_mrts = 5000

bufr_zones_mrts_cnsus = 5000


# buffer zones to get the socio-economic and demographic features around people counter locations
buffr_zone_dist_demogrphc = 0.1

# Threshold value for VIF to reduce colinearity of variables.
vif_threshld = 10  # ideally <10


# start and end dates for retrieving weather data
start = datetime(2020, 1, 1)
end = datetime(2022, 12, 31)


# data file with people counter locations
accesb_area_file = "accessibility.shp"

# location for Green Infrastructure data
GI_Access_maps = data_folder + "GI_Access_maps.gpkg"

# location for Blue Infrastructure data
Blue_Infrastructure_Waterside = data_folder + "Blue_Infrastructure_Waterside.gpkg"

# world boundaries shapefile
world_boundaries = data_folder + "world-administrative-boundaries"

# data folder for people and nautre survey
survey = data_folder + "survey/"

datasets = ['ne_countr_locn_file', 
            'ndw_countr_locn_file', 
            'crt_countr_locn_file',
            'census_locn_file_data',
            f'{census_locn_file_data}infuse_oa_lyr_2011',
            f'{census_locn_file_data}RUC11_OA11_EW.csv',
            f'{census_locn_file_data}household_occupancy.csv',
            f'{census_locn_file_data}age_groups.csv',
            f'{census_locn_file_data}deprivation_dimension.csv',
            f'{census_locn_file_data}population_density.csv',
            f'{census_locn_file_data}working_population.csv',
            f'{census_locn_file_data}population_health.csv',
            f'{census_locn_file_data}ethnicity.csv',
            f'{census_locn_file_data}cars.csv',
            f'{data_folder}accessibility.shp',
            'census_locn_file',
            'land_cluster_file',
            'data_loc_ne_habitat',
            'habitat_cluster_file',
            'nature_srvy_visits',
            'dog_ownership',
            'green_infstrcr',
            'weathr_data',
            'ne_strava_data_file',
            'ndw_strava_data_file',
            'strava_data_loc',
            'val_strava_data_loc',
            'canal_trust_strava_data_loc'
            'GI_Access_maps',
            'Blue_Infrastructure_Waterside',
            'world_boundaries',
            'survey',
            f'{survey}PANS_Y2_Q1_Q4.xlsx'
]

for dataset in datasets:
    if os.path.exists(dataset)== False:
        print(f"The {dataset} dataset does not exist. Please check the data set has been downloaded and saved to the correct directory.")



# CRT sites similar to locations in the training data
sim_sites=['Foxton_Locks', 's163_Leighton_trs003', 'Eshton_Road',
       'Hempsted_Lane', 'Diglis_Dock_Road', 'Stafford', 'Selby',
       'Bath_Road', 'Nottingham_Trent_Lock', 'S145_The_Locks_Car_Park',
       'The_Locks', 'Northgate_Street', 'Rugeley_Towpath',
       'Burton_on_Trent', 'Rathwell_Close', 'Litherland',
       'St_Bernards_Drive', 'Anderton_Boat_Lift', 'Banbury_Towpath',
       'Rochdale', 'Saul_Junction_-_Junction_Bridge',
       'Gloucester_&_Sharpness_Canal', 'Intl_White_Water_Cntr',
       'Riverside', 'Tunnel_Street', 'Canal_Side', 'UofW_Science_Park',
       'S143_science_park_path_to_towpath_crt']

#Reduced number of habitat types
# from https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::living-england-habitat-map-phase-4/about
habitat_dict=dict(zip(['Acid, Calcareous, Neutral Grassland','Arable and Horticultural','Bare Ground','Bare Sand',
'Bog','Bracken','Broadleaved, Mixed and Yew Woodland','Built-up Areas and Gardens','Coastal Saltmarsh',
 'Coastal Sand Dunes','Coniferous Woodland','Dwarf Shrub Heath','Fen, Marsh and Swamp','Improved Grassland',
 'Scrub','Water','Unclassified'],['Grassland','Cropland','Bare Ground','Bare Ground','Wetland','Grassland',
'Woodland','Urban','Coastal','Coastal','Woodland','Heath', 'Wetland','Grassland','Woodland','Freshwater',
                                  'Unclassified']))

# columns to be explored from the People and nature survey data
ftrs_selection=['Respondent_ID','Wave',
               'Region','Age_Band','Gender',
               'Qualification','Interview_Date','Date',
               'No_Of_Visits','Any_Visits_7',
               'Any_Visits_14',
               'Marital_Status','No_Of_Children',
               'Work_Status','Student_Work_Status',
               'Income','Ethnicity','No_Of_Vehicles',
               'Dog','Home_Rural_Urban_Asked',
               'Home_IMD_Decile','Visit_Latitude',
               'Visit_Longitude','Visit_Easting',
               'Visit_Northing','Visit_IMD_Decile']
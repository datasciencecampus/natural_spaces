

from model_config import *
from model_packages import *
from model_utils import *

# functions to extract relevant data layers from downloaded data
def extract_green_infrastructure(GI_Access_maps):
    """Extract green infrastructure data and save to folder."""
    geopkg = gpd.read_file(GI_Access_maps, layer='Map1_OGL')
    green_infra = geopkg[geopkg['Accessible']=='Yes'][['TypologyTitle','geometry']]
    green_infra.to_file(data_folder+'infrastructure_green.gpkg', driver= 'GPKG')
    return green_infra

def extract_woodlands(GI_Access_maps):
    """Extract woodlands data and save to folder."""
    geopkg = gpd.read_file(GI_Access_maps, layer='Map4') 
    woodlands = geopkg[geopkg['AccessLevel']!='Non Accessible']
    woodlands.to_file(data_folder+'infrastructure_woodland.gpkg', driver= 'GPKG')
    return woodlands

def extract_prow(GI_Access_maps):
    """Extract public rights of way data and save to folder."""
    geopkg = gpd.read_file(GI_Access_maps, layer='England_ProwDensity_1kmGrid')
    prow = geopkg[['PROW_Total_length_m','geometry']]
    prow.to_file(data_folder+'infrastructure_prow.gpkg', driver= 'GPKG')
    return prow

def extract_waterside(Blue_Infrastructure_Waterside):
    """Extract inland waterside data and save to folder."""
    geopkg = gpd.read_file(Blue_Infrastructure_Waterside, layer='Inland_Waterside_PROW_ANG')
    waterside = geopkg
    waterside.to_file(data_folder+'infrastructure_waterside.gpkg', driver= 'GPKG')
    return waterside

# functions to merge green and blue infrastructure data with buffer zones around counter locations
def load_site_data():
    """Loads people monitoring site data."""
    df = gpd.read_file(data_folder + 'counter_locations/counter_locations_processed.gpkg')
    return df[['counter', 'geometry', 'provider', 'geom_type']].reset_index(drop=True)

def overlay_with_buffers(data, sites_df, length):
    """Overlay data with sites and calculate areas/lengths."""
    overlay = data.to_crs(crs_mtr).overlay(sites_df,how='intersection')
    
    if length == True:
        overlay['length'] = overlay.length
        overlay= overlay.groupby('counter')['length'].sum().reset_index()
        overlay= overlay[['counter', 'length']]
        
    else:
        overlay['area'] = overlay.area
        overlay=overlay.groupby('counter')['area'].sum().reset_index()
        overlay= overlay[['counter', 'area']]
        
    return overlay


def add_to_static_data(df):
    static= pd.read_pickle(data_folder + 'static_data.pkl')
    static= pd.merge(df, static, on='counter', how='inner')
    static.to_pickle(data_folder+'static_data.pkl')

    

def main():

    # load sites_df
    sites_df= load_site_data()
    sites_df= sites_df.to_crs(crs_mtr)

    # Run extract data functions
    green_infra = extract_green_infrastructure(GI_Access_maps)
    woodlands = extract_woodlands(GI_Access_maps)
    prow = extract_prow(GI_Access_maps)
    waterside = extract_waterside(Blue_Infrastructure_Waterside)

    # Overlay data onto counter locations

    # green area
    green_overlay = overlay_with_buffers(gpd.read_file(data_folder+'infrastructure_green.gpkg'), sites_df, False)
    # rename variable and adjust units
    green_overlay['accessible_green_space_area']=green_overlay['area']/(10**6)
    

    # woodlands
    woodlands_overlay = overlay_with_buffers(gpd.read_file(data_folder+'infrastructure_woodland.gpkg'), sites_df, False) 
    # rename variable and adjust units
    woodlands_overlay['accessible_woodland_space_area']=woodlands_overlay['area']/(10**6)
    
    # PRoW- this has to be processed differently to the other variables
    prow_lengths= gpd.read_file(data_folder+'infrastructure_prow.gpkg') 
    prow_overlay = prow_lengths.to_crs(crs_mtr).overlay(sites_df,how='intersection')
    prow_overlay= prow_overlay.groupby('counter')['PROW_Total_length_m'].sum().reset_index()
    prow_overlay= prow_overlay[['counter', 'PROW_Total_length_m']]
    # rename variable and adjust units
    prow_overlay['PROW_Total_length_km']=prow_overlay['PROW_Total_length_m']/(10**3)

    # waterside
    waterside_overlay = overlay_with_buffers(gpd.read_file(data_folder+'infrastructure_waterside.gpkg'), sites_df, True)
    # rename variable and adjust units
    waterside_overlay['waterside_length_km']=waterside_overlay['length']/(10**3)
    
    # Combine all features to a single data frame
    
    features = reduce(lambda df1,df2: pd.merge(df1,df2, on='counter'), [green_overlay, woodlands_overlay, prow_overlay, waterside_overlay])
    features= features[['counter','accessible_green_space_area', 'accessible_woodland_space_area', 'PROW_Total_length_km', 'waterside_length_km']]
    

    # Save combined data
    features.to_pickle(data_folder+'infrastructure_features.pkl')

    # # add new variables to static data set
    add_to_static_data(features)

if __name__ == "__main__":
    main()



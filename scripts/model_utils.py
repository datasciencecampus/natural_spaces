
from model_packages import *
from model_config import *

def assgn_lat_lon(df_loc,colm_nam):

    """
    Assigns latitude and longitude values from geometry column of GeoDataFrame.
    Also replaces whitespace in counter name column with an underscore

    Parameters
    ----------
        df_loc : Pandas Dataframe
            Data passed into the method
        colm_nam : column containing people counter location names
    Returns
    -------
        df: GeoDataFrame
            Data frame containing geometry, latitude, longitude and counter columns
    """
    
    df=gpd.read_file(df_loc).to_crs(crs=crs_deg)
    
    #Extract lat/lon and create a tuple of lat/lon
    df['latitude'] = df.geometry.apply(lambda p: p.y)
    df['longitude'] = df.geometry.apply(lambda p: p.x)
    df[colm_nam]= df[colm_nam].apply(lambda x: x.replace("  "," ").replace(" ","_"))
    
    return df
    
    
# *** NOT USED ***
# def remove_outlier(dat_frm):

#     """

#     ----------

#     Returns
#     -------

#     """
    
#     # generate univariate observations
#     data = dat_frm
#     # calculate interquartile range
#     q25, q75 = percentile(data, 25), percentile(data, 75)
#     iqr = q75 - q25
#     #print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
#     # calculate the outlier cutoff
#     cut_off = iqr * 1.5
#     lower, upper = q25 - cut_off, q75 + cut_off
#     # identify outliers
#     outliers = [x for x,y in enumerate(data) if y < lower or y > upper]
#     #print('Identified outliers: %d' % len(outliers))
#     # remove outliers
#     #outliers_removed = [x for x in data if x >= lower and x <= upper]
#     #print('Non-outlier observations: %d' % len(outliers_removed))
#     return outliers


# *** NOT USED ***
# class Smoother:
    
#     def __init__(self,df):
#         self.df = df
        
        
#     def apply_smoother(self,smoothng_func):
        
#         if smoothng_func=='exp':
#             smoother = ExponentialSmoother(window_len=window_len, alpha=0.15)
            
#         elif smoothng_func=='conv':
#             smoother = ConvolutionSmoother(window_len=window_len, window_type='ones')
        
#         elif smoothng_func=='sptrl':
#             smoother = SpectralSmoother(smooth_fraction=0.1, pad_len=120)

#         elif smoothng_func=='poly':
#             smoother = PolynomialSmoother(degree=10)

#         elif smoothng_func=='splne':
#             smoother = SplineSmoother(n_knots=6, spline_type='natural_cubic_spline')

#         elif smoothng_func=='gaussn':
#             smoother = GaussianSmoother(n_knots=6, sigma=0.1)

#         elif smoothng_func=='binr':
#             smoother = BinnerSmoother(n_knots=6)

#         elif smoothng_func=='lwes':
#             smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)

#         elif smoothng_func=='klmn':
#             smoother = KalmanSmoother(component='level_trend_season',
#                                       component_noise={'level':0.05, 'trend':0.,'season':0.},n_seasons=2)
       
    
#         else:
#             print('undefined smoother')
#         if smoothng_func=='exp':#or smoothng_func=='conv':
#             indx=self.df.index[window_len:]
#         else:
#             indx=self.df.index
            

#         smoother.smooth(self.df)
    
#         orig_data=smoother.data[0]
#         smth_data=smoother.smooth_data[0]
#         # generate intervals
#         low, up = smoother.get_intervals('sigma_interval')
        
#         return indx,orig_data, smth_data, low[0], up[0]


#     def plot_smooth(self,nam_of_site,datm,actl_dat,trnfmd_dat,bound_a,bound_b):
        
#         # plot the first smoothed timeseries with intervals
#         plt.figure(figsize=(10,5))
#         plt.plot(pd.DataFrame(actl_dat).set_index(datm), '.k')
#         plt.plot(pd.DataFrame(trnfmd_dat).set_index(datm), linewidth=3, color='blue')
#         plt.xlabel('Date')
#         plt.fill_between(pd.date_range(datm.min(), datm.max(), freq="M"), bound_a[0],bound_b[0], alpha=0.3)
#         plt.title(nam_of_site)
#         plt.show()



def prepare_counter_data(data_file_cntr,cut_off_yr):

    """
    Cleaning of people counter data files. Steps include:

        Selects people counter data for selected time period by date.
        Drops columns with % of null data greater than null percentage defined in model_congi.py
        Drops Na values from remaining columns.
        Removes data in the top or bottom 10% of people count values and aggregates to monthly data.
        Imputes missing data using KalmanSmoother
        Plots a random site with upper and lower ranges.

    Parameters    
    ----------
        data_file_cntr : Pandas Dataframe
            Data passed into the method
        cut_off_yr : most recent year of data available
    Returns
    -------
        df_count: Pandas Dataframe
            Contains cleaned People counter data for each location.
            
    """
    
    df_count=pd.read_csv(data_file_cntr)
    
    df_count['Date']=pd.to_datetime(df_count['Time']).dt.date
    
    del df_count['Time']
    
    df_count=df_count.sort_values(by='Date',ascending=True).reset_index(drop=True)
    df_count['Date']=pd.to_datetime(df_count['Date'])
    
    df_count=df_count[df_count['Date'].dt.year>=cut_off_yr]#cnt_ct_off_yr]
    
    # Below code gives percentage of null in every column
    null_percentage = df_count.isnull().sum()/df_count.shape[0]*100

    #print(null_percentage)

    # Below code gives list of columns having more than null_prcntg% 

    col_to_drop = null_percentage[null_percentage>null_prcntg].keys()

    df_count = df_count.drop(col_to_drop, axis=1)


    df_count=df_count.set_index('Date')


    non_empty=[]

    non_empty_no_outlr=[]

    non_empty_no_outlr_mnthly=[]

    for indx in range(df_count.shape[1]):
        
        data=df_count.iloc[:,indx].dropna()
        non_empty.append(data)
    
        # remove outliers: outside 1st and 3rd quantiles
        
    
       # Two average ?
        #quant =data.quantile([0.25, 0.75])
        
        # remove outliers: lowest and top 10%
        
        quant =data.quantile([0.1, 0.9])
    
        no_outlr=data[~data.clip(*quant).isin(quant)]
    
        non_empty_no_outlr.append(no_outlr)
        
        no_outlr_mnthly=no_outlr.resample('M').mean().sort_index()
        
        non_empty_no_outlr_mnthly.append(no_outlr_mnthly)
    
    
    df_count=pd.concat(non_empty_no_outlr_mnthly,axis=1)

    # Below code gives percentage of null in every column
    null_percentage = df_count.isnull().sum()/df_count.shape[0]*100

    #print(null_percentage)

    print(df_count.shape)

    # Below code gives list of columns having more than null_prcntg% 

    col_to_drop = null_percentage[null_percentage>null_prcntg_mnthly].keys()

    df_count = df_count.drop(col_to_drop, axis=1)

    print(df_count.shape)
    
    df_count_smooth=df_count.copy()
    
    
    
    str_low=[]

    str_up=[]

    #To fill-in missing values
    for colmns in range(len(df_count.columns)):
        
        smoother = KalmanSmoother(component='level_season', n_seasons=12,\
                                  component_noise={'level':0.01,'season':0.01})
        
        smoother.smooth(df_count.iloc[:,colmns])
        
        df_count_smooth.iloc[:,colmns]=smoother.smooth_data.T
        
        #  Generate range interval 
        low, up = smoother.get_intervals('kalman_interval')
        
        str_low.append(pd.DataFrame(low.T).set_index(df_count_smooth.index))
        
        str_up.append(pd.DataFrame(up.T).set_index(df_count_smooth.index))
    
    

    #remove all negative counts
    num = df_count_smooth._get_numeric_data()

    num[num < 0] = 0



    empty_cols=[x for x,y in enumerate(df_count_smooth.mean(axis=0)) if y==0]


    #print(empty_cols)

    df_count_smooth=df_count_smooth.drop(df_count_smooth.columns[empty_cols],axis=1)

    df_raw=df_count.copy()
    df_count=df_count_smooth.copy()

    # choose a random site
    rndm_site= 5#random.randint(0, df_count.shape[1]-1)
    chsen_data=df_count.iloc[:,rndm_site]

    nam_site=chsen_data.name

    fig, ax = plt.subplots()
    df_raw.iloc[:,rndm_site].plot(ax=ax, style=['-o'],label='Monthly Average People Count',legend=True)
    ax.set_xlim(0,1200)
    chsen_data.plot(ax=ax,style=['-*'],label='Smoothed Monthly Average People Count',legend=True)
    ax.legend(loc='upper right', frameon=1, facecolor='white')
    str_up[rndm_site].plot(ax=ax,color='black',style=['--'], label='_up')
    str_low[rndm_site].plot(ax=ax,color='black',style=['--'],label='_low')
    plt.title("Cleaned Monthly Average People Count Data for a single location")
    hand, labl = ax.get_legend_handles_labels()
    labs= ['Monthly Average People Count', 'Smoothed Monthly Average People Count', 'Range']  
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
       if l not in lablout:
            lablout.append(l)
            handout.append(h)
    ax.legend(handout, labs, loc='upper right', frameon=1, facecolor='white')
    ax.grid(visible=False)
    fig.savefig(f"./outputs/example_cleaned.png", format= 'png', dpi=300, bbox_inches='tight')
    # plt.savefig(f"./outputs/example_cleaned.png", format= 'png', dpi=300, bbox_inches='tight')
    
    

    print('Null values {}'.format(df_count.isnull().sum(axis=0).sum()))
    
    
    return df_count

#  *** NOT USED ***
# def count_pois(df_poi,poi_col):

#     """

        
#     ----------
#         df_poi : Pandas Dataframe
            
#         poi_col : 
#     Returns
#     -------
#         df_poi_cnt: Pandas Dataframe
            
            
#     """
    
#     df_poi_cnt=pd.DataFrame(list(zip(df_poi.groupby(poi_col)[poi_col].count().index,
#                                      df_poi.groupby(poi_col)[poi_col].count().values)),
#                             columns=[poi_col,poi_col+'_count']).set_index(poi_col).T.reset_index(drop=True)
    
#     renm_dict=dict(zip(list(df_poi_cnt.columns),[poi_col+'_' +x for x in list(df_poi_cnt.columns)]))
    
#     df_poi_cnt.columns.name = None
    
#     df_poi_cnt.rename(columns=renm_dict,inplace=True)
    
    
#     return df_poi_cnt


# *** NOT USED ***
# def coord_lister(geom):

#     """
        
#     ----------
#         geom : 
#     -------
#         df_count: Pandas Dataframe
#             Contains cleaned People counter data for each location.
#     """
#     coords = list(geom.coords)
#     return (coords)[0]


def calculate_vif_(df_X, thresh=vif_threshld):

    """
    Calaculates Variance Inflation Factor score for each variable and drops variable with 
    highest score if above threshold value. VIF is then recalculated and the process repeated
    until the highest score is below the threshold value.

    Parameters    
    ----------
        df_X : Pandas Dataframe
            Dataframe containing numerical training features excluding target variable
        thresh: int
            Numerical value as the threshold cut off for VIF score. Any variable
            that scores higher than this will be dropped

    Returns
    -------
        df_X[cols[variables]]: Pandas Dataframe
            Dataframe continaing variables to be kept for analysis     
        drpd_ftr_lst : List
            List of variables that have been dropped by this function.
    """

    cols = df_X.columns
    variables = np.arange(df_X.shape[1])
    dropped=True
    drpd_ftr_lst=[]
    while dropped:
        dropped=False
        c = df_X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        #print(pd.DataFrame(vif).set_index(cols[variables]))
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            #print('dropping \'' + df_X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            drpd_ftr_lst.append(df_X[cols[variables]].columns[maxloc])
            variables = np.delete(variables, maxloc)
            dropped=True
        #print('Remaining variables:')
        #print(X.columns[variables])
    return df_X[cols[variables]],drpd_ftr_lst


def get_proportion(df_x,ftr_x):

    """
    Calaculates the proportion of the value of each census variable for a single people counter.

    Parameters    
    ----------
        df_X : Pandas Dataframe
            Dataframe containing numerical training features excluding target variable
        ftr_x: Column name
            Column to calculate the proportion of

    Returns
    ----------
    prop:
        The proportion of the value for ftr_x for given people counter

    """

    prop=df_x[list(ftr_x)].div(df_x[list(ftr_x)].sum(axis=1), axis=0)
    
    return prop





def get_season(mnth):

    """
    Assigns a season value based on month of the year. 
    Months 1,2,3 = winter
    Months 4,5,6 = spring
    Months 7,8,9 = summer
    Months 10,11,12 = autumn

    Parameters    
    ----------
        mnth : int
            Numerical month of the year

    Returns
    ----------
    season:
        Season corresponding to month of the year. 

    """
    mnth=int(mnth)
    
    if mnth in (1,2,3):
        season = 'winter'
    elif mnth in (4,5,6):
        season = 'spring'
    elif mnth in (7,8,9):
        season = 'summer'
    else:
        season = 'autumn'
    return season


    
    
    
def geod_buffer(gdf, distance, resolution=16, geod = Geodesic.WGS84):
    """
    Creates a buffer of user defined distance around x, y coordinates.

    Parameters
    ----------
    gdf : 
        GeoDataFrame with geometry column
    distance :
         The radius of the buffer in meters
    resolution :
         The resolution of the buffer around each vertex
    geod - Define an ellipsoid

    Returns
    ----------
    buffer : 
        Polygon geometry of created buffer
    """
    buffer = list()
    for index, row in gdf.iterrows():
        lon1, lat1 = row['geometry'].x, row['geometry'].y
        buffer_ = list()
        for azi1 in np.arange(0, 360, 90/resolution):
            properties = geod.Direct(lat1, lon1, azi1, distance)
            buffer_.append([properties['lon2'], properties['lat2']])
        buffer.append(Polygon(buffer_))
    return buffer
    
    
# *** NOT USED ***
# def getXY(pt):

#     return (pt.x, pt.y)


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
        
        sel_colms=["element_type", "osmid", "amenity", "geometry","tourism","highway"]
        
        gdf=gdf[sel_colms]
        print(gdf.shape)
        gdf=gdf.drop_duplicates()
        print(gdf.shape)
        pois=["amenity","tourism","highway"]
        print(df_loc['counter'][posn])
        tmp_pois_df=[]
        for ftr in pois:
            tmp_df=count_pois(gdf,ftr)
            tmp_pois_df.append(tmp_df)
        tmp_pois_df=pd.concat(tmp_pois_df,axis=1)
        tmp_pois_df['site']=df_loc['counter'][posn].replace("  "," ").replace(" ","_")
        str_pois_df.append(tmp_pois_df)
    
    all_sites_pois_df=pd.concat(str_pois_df,axis=0).fillna(0)
    
    all_sites_pois_df.to_pickle(data_folder+'{}_pois_df.pkl'.format(fl_nam))


def get_corr_matrx(df_for_corr,corr_thrslh):

    """
    Creates a correlation matrix visualisation from Variance Inflation Factor data.

    Parameters
    ----------
    df_for_corr : Pandas Dataframe 
        Data frame containing variables to be visualised
    corr_thrslh :
        VIF score threshold. Scores greater than this value will be visualised
    Returns
    ----------
    all_sites_pois_df : 
        Pickle file of POIs in buffer zones surrounding all people counter locations
    """ 
    
    matrix = df_for_corr.corr(method="pearson").dropna(how='all',axis=1).dropna(how='all',axis=0)
    
    matrix=matrix[abs(matrix)>=corr_thrslh].fillna(0)
    
    # Create a mask
    
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    sns.clustermap(matrix, annot=True,figsize=(10,6),dendrogram_ratio=0.1,cmap='RdBu',square=True,fmt=".2f",annot_kws={"size": 8})
    
    plt.show();
    
    

def prepare_strava(loc_names,loc_fil_nam):

    """
    Takes strava data saved individually for each people counter locationa and
    creates a single data frame containing Strava Metro data for each people counter location.

    Parameters
    ----------
    loc_names : List 
        List containing names of people counter locations.
    loc_fil_nam : str
        Path to folder containing Strava data for relevant people counter locations
    Returns
    ----------
    strava_count : Pandas DataFrame
        Dataframe containing Strava Metro data for each people counter location
        
    """ 
    strava_count=[]
   
    for site in loc_names:
         if pathlib.Path(os.getcwd(),loc_fil_nam, site).exists():

             shp_path=glob.glob(loc_fil_nam+site+'/*.shp')[0]

             csv_path=glob.glob(loc_fil_nam+site+'/*.csv')[0]

             shp=gpd.read_file(shp_path)

             shp=shp.drop_duplicates("edgeUID").reset_index(drop=True)

             csv=pd.read_csv(csv_path)
    
             #cj:do we need this ?
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

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if np.round(frac_train + frac_val + frac_test) != 1:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test
    
    
def r2(x, y, ax=None, **kws):
    ax = ax or plt.gca()
    slope, intercept, r_value, p_value, std_err = linregress(x=x, y=y)
    ax.annotate(f'$r^2 = {r_value ** 2:.2f}$\nEq: ${slope:.2f}x{intercept:+.2f}$',
                xy=(.05, .95), xycoords=ax.transAxes, fontsize=8,
                color='darkred', backgroundcolor='#FFFFFF99', ha='left', va='top')    
    
    
def anonymise_coordinates(geo_df, displacement):
    geo_df= geo_df.to_crs(crs_mtr)
    r= displacement
    angle= np.random.random()* 2 * np.pi
    geo_df.latitude = geo_df.geometry.x + (r * np.sin(angle))
    geo_df.longitude = geo_df.geometry.y + (r * np.cos(angle))
    geo_df.geometry= gpd.points_from_xy(geo_df.latitude, geo_df.longitude)
    return geo_df

def create_density_map(df, cols):


    # get min max values to set legends
    summer2022= df.loc[(df['year']==2022) & (df['season']=='Summer')]
    # for people count and predicted count
    summer2022max= summer2022['People Counter Data'].max()
    summer2022min= summer2022['People Counter Data'].min()
    # for strava trip count
    summer2022max_strava= summer2022['Strava Trip Count'].max()
    summer2022min_strava= summer2022['Strava Trip Count'].min()

    for col in cols:
        # if statement to determine which legend min max values to use. Strava data should be plotted on its own scale
        if np.isin(col, ['People Counter Data', 'Predicted Count', 'Error', 'Canals & Rivers Trust Prediction']) ==True:
            # loop through years
            for year in df['year'].unique():
                # loop through seasons
                for season in df['season'].unique():
                    # create df for plotting
                    df_plt= df.loc[(df['year']==year) & (df['season']==season)]
                    # plot density mapbox
                    fig=px.density_mapbox(df_plt, lat='latitude', lon='longitude',z= df_plt[col], hover_name='site',
                        mapbox_style="stamen-terrain", title=f"{col} Density {season} {year}", zoom=4.25, range_color=[summer2022min, summer2022max],
                        labels={
                            col: "People Count"
                        })
                    fig.show()
                    # save figure to images folder
                    # pathlib.Path(f"./images/{year}/").mkdir(parents=True, exist_ok=True)
                    # fig.write_image(f"./images/{year}/{year}_{season}_{col}.png")
        
        elif np.isin(col, ['Strava Trip Count', 'Strava Error'])==True:
            # loop through years
            for year in df['year'].unique():
                # loop through seasons
                for season in df['season'].unique():
                    # create df for plotting
                    df_plt= df.loc[(df['year']==year) & (df['season']==season)]
                    # plot density mapbox
                    fig=px.density_mapbox(df_plt, lat='latitude', lon='longitude',z= df_plt[col], hover_name='site',
                        mapbox_style="stamen-terrain", title=f"{col} Density {season} {year}", zoom=4.25, range_color=[summer2022min_strava, summer2022max_strava],
                        labels={
                            col: "People Count"
                        })
                    fig.show()
                    # save figure to images folder
                    # pathlib.Path(f"./images/{year}/").mkdir(parents=True, exist_ok=True)
                    # fig.write_image(f"./images/{year}/{year}_{season}_{col}.png")
        
        else:
             print(f'{col} variable not plotted with this function')


def create_chlorpleth_map(df, cols, crt):
    
    global train_sites
    global train_val_df
    global points_geom

    if crt == True:
        points= points_geom.to_crs('3857')
    else:
        points= train_sites.to_crs('3857')
    # get min max values to set legends
    summer2022= train_val_df.loc[(df['year']==2022) & (train_val_df['season']=='Summer')]
    # for people count and predicted count
    summer2022max= summer2022['People Counter Data'].max()
    summer2022min= summer2022['People Counter Data'].min()
    # for strava trip count
    summer2022max_strava= summer2022['Strava Trip Count'].max()
    summer2022min_strava= summer2022['Strava Trip Count'].min()

    for col in cols:
        # if statement to determine which legend min max values to use. Strava data should be plotted on its own scale
        if np.isin(col, ['People Counter Data', 'Predicted Count', 'Error', 'Canals & Rivers Trust Prediction']) ==True:
            for year in df['year'].unique():
                # loop through seasons
                for season in df['season'].unique():
                    # create df for plotting
                    plt_df= df.loc[(df['year']==year) & (df['season']==season)]
                    fig, ax = plt.subplots(1, 1)
                    data=gpd.GeoDataFrame(plt_df).to_crs('3857').plot(column=plt_df[col], ax=ax, legend=True,cmap='OrRd', vmin= summer2022min, vmax=summer2022max,
                        legend_kwds={'label':f'Mean of {col}','orientation':'vertical'}
                    )

                    

                    points.plot(
                        # Colour by region label
                        column='NUTS_NAME',
                        # Consider label as categorical
                        categorical=True,
                        legend=True,
                        label='People Counters',
                        # Draw on axis `ax`
                        ax=ax,
                        # Use circle as marker
                        marker="o",
                        # marker size
                        markersize=1,
                        # colours for markers
                        color='slategrey',
                    )

                    # Add basemap
                    contextily.add_basemap(
                        ax,
                        source=contextily.providers.CartoDB.Positron,
                    )

                    # add second legend for people counter locations
                    legend1 = ax.legend(handles=[
                                lines.Line2D(
                                    [],
                                    [],
                                    color="slategrey",
                                    lw=0,
                                    marker="o",
                                    markersize=5,
                                    label='People Counters',
                                    )], 
                            scatterpoints=1, frameon=True,
                            labelspacing=1, loc='upper right', fontsize=8,  
                            title_fontsize=10,
                            labelcolor='black',
                            markerfirst=True,
                            labels=['People Counters']
                            )
                    fig.gca().add_artist(legend1)

                    ax.set_axis_off()

                    ax.set_title(f"{col} {season} {year} regional mean")
                    fig.tight_layout()
                    # pathlib.Path(f"./images/{year}/").mkdir(parents=True, exist_ok=True)
                    # fig.savefig(f'./images/{year}/{year}_chloropleth_{season}_{col}.png')
        
        elif np.isin(col, ['Strava Trip Count', 'Strava Error'])==True:
            for year in df['year'].unique():
                # loop through seasons
                for season in df['season'].unique():
                    # create df for plotting
                    plt_df= df.loc[(df['year']==year) & (df['season']==season)]
                    fig, ax = plt.subplots(1, 1)
                    data=gpd.GeoDataFrame(plt_df).to_crs('3857').plot(column=plt_df[col], ax=ax, legend=True,cmap='OrRd', vmin= summer2022min_strava, vmax=summer2022max_strava,
                        legend_kwds={'label':f'Mean of {col}','orientation':'vertical'}
                    )

                    

                    points.plot(
                    # Colour by region label
                    column='NUTS_NAME',
                    # Consider label as categorical
                    categorical=True,
                    legend=True,
                    label='People Counters',
                    # Draw on axis `ax`
                    ax=ax,
                    # Use circle as marker
                    marker="o",
                    # marker size
                    markersize=1,
                    # colours for markers
                    color='slategrey',
                )

                    # Add basemap
                    contextily.add_basemap(
                        ax,
                        source=contextily.providers.CartoDB.Positron,
                        )

                        # add second legend for peopl counter locations
                    legend1 = ax.legend(handles=[
                                    lines.Line2D(
                                        [],
                                        [],
                                        color="slategrey",
                                        lw=0,
                                        marker="o",
                                        markersize=5,
                                        label='People Counters',
                                        )], 
                                scatterpoints=1, frameon=True,
                                labelspacing=1, loc='upper right', fontsize=8,  
                                title_fontsize=10,
                                labelcolor='black',
                                markerfirst=True,
                                labels=['People Counters']
                                )
                    fig.gca().add_artist(legend1)

                    ax.set_axis_off()

                    ax.set_title(f"{col} {season} {year} regional mean")

                    fig.tight_layout()
                    fig.show()
                    # pathlib.Path(f"./images/{year}/").mkdir(parents=True, exist_ok=True)
                    # fig.savefig(f'./images/{year}/{year}_chloropleth_{season}_{col}.png')
    
        else:
            print(f'{col} variable not plotted with this function')
# Accessing Raw Data

This document explains how to retrieve the raw data used in the project and the folder structure necessary to run the project.



# Counter Location Data


# Strava Data
Downlaoding Strava Metro data can be completed by running the 'download_counter_data.ipynb'. Please note an API key will be required, applications for an API key can be completed [here](https://metro.strava.com).

# Census Data

All census features are downloaded fomr Census 2011 at the Output Area level from [Nomis](https://www.nomisweb.co.uk).

Features Required:

* Output Area shapefiles
* Urban-Rural classification
* Household occupancy
* Age by single year
* Households by deprivation dimensions
* Population density
* Working population
* General health
* Ethnic group
* Car or Van availability



# Accessible Green & Blue Infrastructure 

A series of geodatabases that contain a range of spatial datasets. These datasets describe the location and geographical extent of different types of Green and Blue Infrastructure across England. The datasets highlight accessibility levels, display greenspace provision and natural greenspace standards in a spatial context and present it alongside a wide range of social statistics. 

* [Green and Blue Infrastucture information](https://www.data.gov.uk/dataset/f335ab3a-f670-467f-bedd-80bdd8f1ace6/green-and-blue-infrastructure-england)
* [Green and Blue Infrstructure download link](https://s3.eu-west-1.amazonaws.com/data.defra.gov.uk/Natural_England/Access_Green_Infrastructure/Green_and_Blue_Infrastructure_NE/Green_and_Blue_Infrastructure_Opendata_NE_Geopackage.zip)



# Land Habitat Data

The habitat classification map  uses a machine learning approach to image classification, developed under the Defra Living Maps project (SD1705 – Kilcoyne et al., 2017). The method first clusters homogeneous areas of habitat into segments, then assigns each segment to a defined list of habitat classes using Random Forest (a machine learning algorithm). The habitat probability map displays modelled likely broad habitat classifications, trained on field surveys and earth observation data from 2021 as well as historic data layers. This map is an output from Phase IV of the Living England project, with future work in Phase V (2022-23) intending to standardise the methodology and Phase VI (2023-24) to implement the agreed standardised methods.

* [Landcover (Living England), Living England Habitat Map (Phase 4) | Natural England Open Data Geoportal](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::living-england-habitat-map-phase-4/about)



# People and Nature Survey Data

[The People and Nature Survey](https://www.gov.uk/government/collections/people-and-nature-survey-for-england)for England gathers evidence and trend data through an online survey relating to people’s enjoyment, access, understanding of and attitudes to the natural environment, and it’s contributions to wellbeing. Specifically, we will find mean dog ownership.

* [People and Nature Survey for England - Year 2 - Quarter 1 to Quarter 4 data](https://www.gov.uk/government/statistics/the-people-and-nature-survey-for-england-year-2-annual-report-data-and-publications-april-2021-march-2022-official-statistics-main-findings)



# Weather Data

Historical weather data for each individual people monitoring sites downlaoded using [Meteostat](https://meteostat.net/en/blog/obtain-weather-data-any-location-python) package.

# Folder Structure

The structure of the data folder is as follows. This folder structure is necessary to match the paths defined in modle_config.py

<pre>
data/
├── cenus/
├── strava_and_counter_data_updated/
│   └── data_01-2020_to_11-2022/
└── survey/
</pre>
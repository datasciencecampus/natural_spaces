# Measuring Engagement with Nature

Estimating use of natural spaces based on People Counter data, Strava Metro data, Census data, weather data, Open Street Map data, Green Infrstructure data

________________________________________________________________
The goal of this project is to measure people's engagement with the natural environment. This information supports understanding of progress against Defra’s Environmental Improvement Plan 2023 which sets out key targets and commitments on access and engagement with nature, including a commitment that everyone should live within 15 minutes’ walk of a green or blue space.

Automated people counters are used frequently to monitor pedestrian and cycling activity with a good temporal resolution. However, people counters are expensive to install and maintain and for this reason are only installed in a few strategic locations. 

Introducing an inexpensive and widely applicable data science method for monitoring visitor numbers would considerably enhance Defra’s indicator for tracking nature engagement. This model combines aggregated and anonymised data from Strava Metro with carefully selected open or free-to-access spatial datasets such as automated people counters and indicators of local environmental and social conditions. These results are experimental, in order to produce more robust results to inform outcomes we need to incorporate a larger set of training data and datasets that cover the residential location of visitors. 

A brief introduction to the methodology used and interpretation of results can be found in [this blog post](https://datasciencecampus.ons.gov.uk/using-open-source-data-to-measure-our-engagement-with-the-natural-environment/)


## Requirements
_________________________________________________________________

The project was run on Python version 3.10.9. You can find a list of the direct dependencies, with versions, in the requirements.txt file.
Run the below commands in command line to recreate the environment needed to run the code.

```shell
python3 -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```


## Usage
_________________________________________________________________
Confirm that you have access to the data sets required, details of the data sets and steps to download the data can be found in `data` > `README.md`

Once all data sets are present run python notebooks 1 to 10 found in the `notebooks` folder. 

Steps:

* `1.Prepare_buffer_zones_counter_sites.ipynb` This notebook produces a table of point geometries for all the people monitoring sites for which we have access to. A 5km buffer zone is created around each point, this will later be used to for building our dataset . Currently we have access to training data for Natural England and North Downs Way sites only.

* `2.Prepare_Census_features.ipynb` This notebook retrieves relevant features from the 2011 Census at Output Area level. The 5km buffer zones surrounding each people counter location are then intersected with census data to create a data set relevant to each location. Finally, these data sets are compiled into a single data set containing all census features.

* `3.Green_blue_infrastructure_features.ipynb` Collect green and blue infrasturtcure features including: Accessible woodlands, Public Right of Way and Inland Waterside. Intersect these features with buffer geometries and collate into a single dataset.

* `4.Land_classification_fatures.ipynb` This notebook uses the Urban/Rural land classifications taken from the census in `2.Prepare_Census_features.ipynb`. For each people counter location the sum of the area in the buffer zone that corresponds to each Urban/rural classification is taken. Based on these area values density based clustering is performed to provide a label for each buffer zone and corresponding people counter location. 

* `5.Land_habitat_features.ipynb` This notebook uses habitat classification map developed under the Defra Living Maps project (SD1705 – Kilcoyne et al., 2017). The method first clusters homogeneous areas of habitat into segments, then assigns each segment to a defined list of habitat classes using Random Forest (a machine learning algorithm). The habitat probability map displays modelled likely broad habitat classifications. As in the previous notebook the land habitat classifiction is intersected with the buffer zones around people counter locations. Density based clustering is then performed which then provides a label for each cluster and therfore each people counter location.

* `6.Nature_survey_features.ipynb` We collect relevant features from this the [People and Nature Survey] (https://www.gov.uk/government/collections/people-and-nature-survey-for-england), specifically mean dog occupancy in buffer areas around each counter site.

* `7.Weather_features.ipynb` This notebook pulls historical weather data for each individual people monitoring sites making use of [Meteostat package](https://meteostat.net/en/blog/obtain-weather-data-any-location-python)

* `8.Data_augmentation.ipynb` This notebook deals with preprocessing the collected data sets in prepatation for modelling. Additional point of interest data is obtained from Open Street Map. Exploratory factor analysis is conducted on features that overlap spatially and is used to evaluate the interaction between our features and decide how to account for those as inputs in the mode. Pearson pairwise correlation is use to remove correlated features among points of interest data set. Variance Inflation Factor is then used to further reduce colinearity of variables across all remaining variables in the data set. Finally, static and dynamic features are merged into a single data set to be used in model development.

* `9.Model_development.ipynb` This notebook utilises the single data set of static and dynamic predictors to develop a spatial-temporal model that estimates visitor numbers. The final model is an ensemble regression model with regularisation and is implemented into an automated machine learning workflow to aid in future use.

* `10.Weather_features.ipynb` This notebook explores the effects of the dynamic variables on visitor numbers. A fixed effect approach is utilised to take advantage of the panel data structure of the dynamic features, in this case average monthly temperature and Strava Metro Total Trip Count.


## Limitations

Currently the project is limited by the availablilty of automatic people counter data which is used as the ground truth training data for the model. With more training data available an experiemental statistic that could produce reliable estimates for engagement with nature across a wider area of England could be produced.  In the next phase, we plan to work with a range of partners that maintain and can share people counter data to address this limitation.

## License
_________________________________________________________________


## Credits
_________________________________________________________________

This project was generated from a collaboration with [The Department for Environment, Food and Rural Affairs](https://www.gov.uk/government/organisations/department-for-environment-food-rural-affairs) and [The Data Science Campus](https://datasciencecampus.ons.gov.uk/) at [The Office for National Statistics](https://www.ons.gov.uk/), [Strava Metro] (https://metro.strava.com), [Natural England] (https://www.gov.uk/government/organisations/natural-england) and [The North Downs Way] () 

Developers of this project include Kaveh Jahanshahi (ONS),  Chaitanya Joshi (ONS), Tim Ashelford (DEFRA), Jamie Elliott (ONS) and James Kenyon (DEFRA). 

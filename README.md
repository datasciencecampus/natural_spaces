# Measuring Engagement with Nature

Estimating use of natural spaces based on People Counter data, Strava Metro data, Census data, weather data, Open Street Map data, Green Infrstructure data

________________________________________________________________
The goal of this project is to measure peopple's engagement with the natural environment. This information supports understanding of progress against Defra’s Environmental Improvement Plan 2023 which sets out key targets and commitments on access and engagement with nature, including a commitment that everyone should live within 15 minutes’ walk of a green or blue space.

Automated people counters are used frequently to monitor pedestrian and cycling activity with a good temporal resolution. However, people counters are expensive to install and maintain and for this reason are only installed in a few strategic locations. 

Introducing an inexpensive and widely applicable data science method for monitoring visitor numbers would considerably enhance Defra’s indicator for tracking nature engagement. This model combines aggregated and anonymised data from Strava Metro with carefully selected open or free-to-access spatial datasets such as automated people counters and indicators of local environmental and social conditions. These results are experimental, in order to produce more robust results to inform outcomes we need to incorporate a larger set of training data and datasets that cover the residential location of visitors. 


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
Confirm that you have access to the data sets required, details of the data sets can be found in `data` > `README.md`
## Limitations

Currently the project is limited by the avilablilty of automatic people counter data which is used as the ground truth training data for the model. With more training data available an experiemental statistic that could produce reliable estimates for engagement with nature across a wider area of England could be produced.  In the next phase, we plan to work with a range of partners that maintain and can share people counter data to address this limitation.

## License
_________________________________________________________________


## Credits
_________________________________________________________________

This project was generated from a collaboration with [The Department for Environment, Food and Rural Affairs](https://www.gov.uk/government/organisations/department-for-environment-food-rural-affairs) and [The Data Science Campus](https://datasciencecampus.ons.gov.uk/) at [The Office for National Statistics](https://www.ons.gov.uk/), [Strava Metro] (https://metro.strava.com), [Natural England] (https://www.gov.uk/government/organisations/natural-england) and [The North Downs Way] () 

Developers of this project include Kaveh Jahanshahi (ONS),  Chaitanya Joshi (ONS), Tim Ashelford (DEFRA), Jamie Elliott (ONS) and James Kenyon (DEFRA). 

# journey_plotter
For plotting journeys in .gpx files, making maps and videos of the journeys

## Getting started
1. You may need to add the conda-forge channel to Anaconda:
`conda config --add channels conda-forge` 
2. The folder specified as `data_path` in configs.py should contain:
    1. `cycling_data`, a folder with the .gpx files of the journeys
    2. `images_for_video`, an empty folder
    3. `OS_base_maps`, a folder containing the folders `OS Open Greenspace (ESRI Shape File) TQ` and `OS OpenMap Local (ESRI Shape File) TQ` - these maps can be downloaded [here](https://www.ordnancesurvey.co.uk/opendatadownload/products.html)
    4. `results`, an empty folder
3. Create conda environment from the environment.yml: From the project directory, open a Git bash and execute: `conda env create -f environment.yml`
4. Set `configs.py` as desired
5. Run `main.py`!
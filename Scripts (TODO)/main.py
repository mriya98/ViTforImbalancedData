import datasets
import datasetPreprocessing
import global_variables

# Load dataset
datasetPreprocess = datasetPreprocessing()

# If dataset has csv and not a metadata json, first create the json file
datasetPreprocess.create_metadata(global_variables.DATASET_FOLDER, global_variables.DATASET_CSV_FILE)



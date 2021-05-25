# Natural Language Processing on Prescription Drug Reviews

**Note:** this repository only stores copies of our code.  The data files these codes depend on and generate are stored in a google drive and are far to large to store on git.  As such, these files will not run.  To see functional versions of these scripts, please visit https://drive.google.com/drive/folders/1WfsXz5d1XxHRl9BctP5t18iN-netd2-v?usp=sharing.  All files will be in the directory 2_scripts unless otherwise noted

### File Descriptions
Models.py:  Our LSTM Models
ModelContext.py:  Object that holds everything required to train and evaluate model
Trainer.py:  Object that trains model
Evaluator.py:  Object that executes evaluation steps during and after training and saves in appropriate directory
Plotter.py:  Object that can crawl through directory created by evaluator and create charts
collate.py:  The home of our collate functions
create_dataloader.py:  Dataset objects and a function that returns a DataLoader from a csv 
preprocessing.ipynb:  Executes pre-processing steps and save csv that can be loaded into dataloader objects.  Also creates vocabularies and saves as json objects
run_LSTM_notebook.ipynb:  Runs LSTM models by intiating a ModelContext and calling run().  Includes ability to set paramters for run





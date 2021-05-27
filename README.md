# Natural Language Processing on Prescription Drug Reviews

**Note:** this repository only stores copies of our code.  The data files these codes depend on and generate are stored in a google drive and are far to large to store on git.  As such, these files will not run.  To see functional versions of these scripts, please visit https://drive.google.com/drive/folders/1WfsXz5d1XxHRl9BctP5t18iN-netd2-v?usp=sharing.  All files will be in the directory 2_scripts unless otherwise noted

### File Descriptions
### **2_scripts**:
**Drugs.com_scraper/scrape.py**: This file crawls the drugs.com website letter by letter and then crawls sub letters to scrape individual drug reviews. This script was used to supplement the original dataset which included drug reviews up to January 31, 2017. 
*170 lines of code with documentation.* 

**embeddings/embeddings.ipynb**: This file loads the twitter drug word embedding into a weight matrix and tests that it words with a Vanilla Neural Network. 
*204 lines of code without documentation.*

**viz/results_viz.ipynb**: This file defines two visualizations functions using Altair. The first one is a line chart to plot training, validation, and test accuracies. The second function creates a confusion matrix for the three sentiment categories given a dataframe with columns with predicted classes and true labels to map accuracy. 
*185 lines of code.*

**Create_vocab.ipynb**: This function creates a vocabulary object using the Spacy tokenizer and saves it to the data file, in order to save time in later model creation scripts.
*38 lines of code.*

**Logit_colab**: This script trains three different logits using the google Colab GPU. The first logit uses the original dataset of Grasser et al. which contains duplicates, the second is the same dataset with dropped duplicates, and the third is the non-duplicated dataset containing additional, newly scraped data.
*352 lines of code.*

**Models.py**:  Our LSTM Models that rely only on text-based features.
*352 lines of code.*

**ModelContext.py**:  Object that holds everything required to train and evaluate model
*352 lines of code.*

**Trainer.py**:  Object that trains model
*352 lines of code.*

**Evaluator.py**:  Object that executes evaluation steps during and after training and saves in appropriate directory
*352 lines of code.*

**Plotter.py**:  Object that can crawl through directory created by evaluator and create charts
*352 lines of code.*

**collate.py**:  The home of our collate functions used for text-based feature models
*352 lines of code.*

**create_dataloader.py**:  Dataset objects and a function that returns a DataLoader from a csv 
*352 lines of code.*

**preprocessing.ipynb**:  Executes pre-processing steps and save csv that can be loaded into dataloader objects.  Also creates vocabularies and saves as json objects
*352 lines of code.*

**run_LSTM_notebook.ipynb**:  Runs LSTM models by intiating a ModelContext and calling run().  Includes ability to set paramters for run
*352 lines of code.*



### **3_data**:
**duplicate_exploration/create_datasets.ipynb**: This script creates the three different datasets to use with various models to explore the implications of duplicated datasets in the original datafile on previous research. 
*57 lines of code.*

**duplicate_exploration/duplicate_reviews.ipynb**: This script quantifies the number of duplicates in Grasser et al.’s dataset, and explores the “leakage” between training and testing datasets. 
*36 lines of code.*


- Models.py:  Our LSTM Models
- ModelContext.py:  Object that holds everything required to train and evaluate model
- Trainer.py:  Object that trains model
- Evaluator.py:  Object that executes evaluation steps during and after training and saves in appropriate directory
- Plotter.py:  Object that can crawl through directory created by evaluator and create charts
- collate.py:  The home of our collate functions
- create_dataloader.py:  Dataset objects and a function that returns a DataLoader from a csv 
- preprocessing.ipynb:  Executes pre-processing steps and save csv that can be loaded into dataloader objects.  Also creates vocabularies and saves as json objects
- run_LSTM_notebook.ipynb:  Runs LSTM models by intiating a ModelContext and calling run().  Includes ability to set paramters for run





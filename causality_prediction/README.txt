FILES
=====

create_feature_space.py : compute the feature vector of each given trial
create_level2_dataset.py : collection of functions used for the mapping into the feature space
supervisedNN.py : run the experiment called Supervised[NN] in the paper
supervisedMAR.py : run the experiment called SUpervised[MAR] in the paper
score_function.py : functions used by the classification scripts 



INSTRUCTIONS
============



Here is the code to replicate the results of the paper.

List of the .py files:

- create_feature_space: compute the feature vector of each given trial
- create_level2_dataset: collection of functions used for the mapping into the feature space

- supervisedNN: run the experiment called Supervised[NN] in the paper
- supervisedMAR: run the experiment called SUpervised[MAR] in the paper
- score_function: collection of functions used by the classification scripts 

Folder /data is supposed to contain the list of files in the variable filedata_list in create_feature_space.
Such files are the dataset genereted by the NN model.

Folder /results is supposed to contain the fueture space files of the MAR dataset and 
it will contain the output files of the functions create_feature_space, supervisedMAR and supervisedNN.


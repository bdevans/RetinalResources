# RetinalResources

This code implements experiments described in the paper "The Effects of Neural Resource constraints on Early Visual Representations," to be presented at ICLR 2019 (https://openreview.net/pdf?id=S1xq3oR5tQ).

-- TrainModel.py implements the (parameterizable) model architecture and training scheme.   Trained models are saved in the saved_models directory.  Model performance histories are saved in the Logs directory.

-- TrainAllModels.ipynb provides uses TrainModel.py to train multiple model architectures with specified parameters, for multiple trials (i.e. different random weight initializations).

-- PlotPerformance.ipynb is used to show the performance (on the CIFAR 10 test set, after training) of different model architecture parameter settings.

-- VisualizeRFs.py implements the code that computes and produces visualizations of the receptive fields of trained convolutional channels.  Visualizations are saved in the saved_visualizations directory with file names that indicate the model parameter settings and trial number.  Numpy matrices representing the receptive fields are saved in the saved_filters directory, and relevant model weight matrices (i.e. the "V1" layer weights) are saved in the saved_weights directory.

-- VisualizeRFsRandomInits.py is similar to VisualizeRFs.py, but computes the first-order approximations of convolutional channel receptive fields using gradient ascent starting from multiple (ten) different random initializations rather than from a blank stimulus.  These are used for the simple vs. complex cell analysis described in the paper.

-- ProcessAllRFs.py calls VisualizeRFs.py and VisualizeRFsRandomInits.py for trained models with specified parameters and trial numbers.

-- Quantify_Orientedness.ipynb quantifies and visualizes (in polar plots) the orientedness / isotropy of the relevant model receptive fields, as well as of model weight matrices.

-- Quantify_Complexity.ipynb quantifies the extent to which first-order approximations of model receptive fields vary according to the random initialization used in the gradient ascent approximation.



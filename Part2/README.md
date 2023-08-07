# Barcodes-DNN
This repository contains code for some experiments from the paper. This code is a modified version of some approaches from https://github.com/timgaripov/dnn-mode-connectivity. We put here only the key featuring scripts, for the rest of the code please refer to the https://github.com/timgaripov/dnn-mode-connectivity.

To obtain a curve between two local minima, first find local minima of a model using train.py, second train a curve between minima (train.py with curve option), finally evaluate the curve with eval_curve.py. For more detailed instructions, please refer to https://github.com/timgaripov/dnn-mode-connectivity.

To obtain a triangle between three local minima, run train_triangle.py with checkpoints of three curve models. 
When triangle model is trained, run eval_triangle.py to evaluate the model on the triangle.

Report.ipynb contains examples of plots from the paper.


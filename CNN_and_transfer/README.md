Sample problem in computer vision.


Kaggle competition dataset. The winner of this project obtained an accuracy on the validation data of ~ 90%, here I manage 
an accuracy closer to 92% with custom neural network and ~99% using transfer learning on the validation data. A few techniques to prevent overfitting include: batch normalization, image augmentation, neuron dropout and early stopping. L1 & L2 regularization is also provided in the models as an option
but were not necessary since the aforementioned did a great job at handling the training bias and variance.


- The script catdog_network.py contains the 3 neural network models considered for the project. 
- The jupyter notebook CNN_notebook.ipynb contains our interactive example. 

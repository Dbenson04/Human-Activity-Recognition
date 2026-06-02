#Human-Activity-Recognition
Written in Python and employing transfer learning with the ResNet50 model, our team recorded short gifs of ourselves performing referee signals in Baseball to train our AI model
in hopes of successfully classifying our actions. Our dataset was spliced into individual gifs from one long video, ensuring each splice is of equal length and grouped by their respective action.
When incorporating our transfered model, ResNet50, we froze our layers and added 2 dense layers following it, one with 256 neurons using the ReLu activation function,
and the other with 4 neurons using the softmax activation function, with one neuron for each class. Additionally, a dropout of 40% is used between the layers in attempt to reduce overfitting, and we decided on the Adam optimizer
with "sparse_categorical_crossentropy" as our loss function. Overall, our model performs with a training and validation accuracy of 100% and 80% respectively, pointing towards signs of memorization or overfitting.

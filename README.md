<h1>Machine Learning Framework</h1>
<h3>Overview</h3>
<p>This machine learning framework has been hacked together with Numpy to help solidify my understanding of machine learning. It was tested using data of my favorite spongebob characters. It can classify your favorite characters too, assuming they are the same as mine:</p>
<ul>
    <li>Spongebob Squarepants</li>
    <li>Plankton</li>
    <li>Gary</li>
    <li>Patrick Star</li>
    <li>Sandy Cheeks</li>
    <li>Mr. Krabs</li>
    <li>Squidward</li>
</ul>

<h3>Details</h3>

<p>This project implements a three layer standard neural network. The overall architecture is as follows:</p>
<ul>
    <li>Input Layer --> 98 flattened image examples, each of original shape 100 x 100 x 3 (RGB)</li>
    <li>Hidden Layer --> A fully connected layer of input units 30000 and output units 10</li>
    <li>Activation --> ReLU activation layer</li>
    <li>Output Layer --> A fully connected layer of input units 10 and output units 7 (the number of classes)</li>
    <li>Output Activation --> Softmax activation layer</li>
</ul>

<p>The F1 scores for the datasets are:</p>
<ul>
    <li>Train: 100%
    <li>Cross Validation: 96%
    <li>Test: 99%</li>
</ul>
<p>As you can see, the model does a better job on the test set than on the cross validation set, which, since the datasets are small, implies overfitting.</p>

<h3>Instructions</h3>
<p>To test the tool:</p>
<ol>
  <li>First train the model by running `run_classifier_test.py`</li>
  <li>Then train the generator by running `run_generator_test.py`</li>
  <li>Eventually, it will output it's own circle, one that it drew from the distribution of circle images it was trained on</li>
</ol>

<p>The way I did this was by training the classifier first, then adding a "generator layer" on top of it and attempting to classify the inputs from that layer
as circles. The error initially would be huge as the classifier would be fed just random noise. But back doing a sort of compounding backprop by "shirking" the gradients onto the layer before it until
you get to the generator and applying the amassed gradient from the lower layers. Since those layers know that they are trained to detect circles, they don't need
to update themselves, but they need to send the feedback of how much the layer before needs to change in order for them to make the correct prediction.</p>

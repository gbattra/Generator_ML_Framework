<h1>Generator Model Machine Learning Framework</h1>
<h3>Overview</h3>
<p>This is another framework hacked together to attempt to build a generator model, which, when trained on various shapes can draw its own shapes.
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

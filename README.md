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

It accomplishes this through standard backpropogation, but instead of applying the gradients at each layer, those gradients are "shirked" up to the layer above until reaching the input layer, where the gradients are finally applied. The final input values result in a fuzzy image of the target class.

<img src="https://portfolio-attra.herokuapp.com/assets/images/circle.jpg"/>



class GradientCheckService:

    @staticmethod
    def check_gradients(layer, classifier):
        # get grads from layer
        grads = layer.backward_cache['dW']
        # flatten layer W
        shape = layer.W.shape
        W_flat = layer.W.flatten()

        epsilon = 0.001

        print('Numerical Grad', 'Computed Grad')
        # loop through first few W's
        for i in range(0, 10):
            W_initial = W_flat[i]
            W_plus = W_initial + epsilon
            W_minus = W_initial - epsilon

            W_flat[i] = W_plus
            layer.W = W_flat.reshape(shape)
            pred = classifier.forward_propogate(classifier.data.x)
            cost_plus = classifier.compute_cost(classifier.data.y, pred)

            W_flat[i] = W_minus
            layer.W = W_flat.reshape(shape)
            pred = classifier.forward_propogate(classifier.data.x)
            cost_minus = classifier.compute_cost(classifier.data.y, pred)

            computed_grad = (cost_plus - cost_minus) / (2 * epsilon)

            print(grads.flatten()[i], computed_grad)

            # reset layers W's
            W_flat[i] = W_initial
            layer.W = W_flat.reshape(shape)

        return layer
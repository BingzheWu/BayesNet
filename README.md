### Bayes Neural Network

Implementation of Bayes by Back-propagate.

##### Basic usage
* Building an inference net
    ```Python
    from bayes_network import BayesMLP
    device = 'cuda' 
    net = BayesMLP(device = device)
    net.forward(image, sample = False) 
    ```
Note that the model BayesMLP is designed for the mnist classification. 
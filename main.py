import numpy as np
import random

class Network():
    def __init__(self, layers): # Setup network information
        self.input_size = layers["input"]["length"]
        self.hidden = layers["hidden"]
        self.output = layers["output"]["length"]
    
    def reset(self, settings): # Reset the network by giving it random weights and biases
        def set_weights():
            min_val = settings["weights_range"][0] * 1000
            max_val = settings["weights_range"][1] * 1000

            weights = []

            for index, i in enumerate(self.hidden): # For each hidden layer
                layer_weights = []
                for n in range(i["length"]): # For each neuron of the hidden layer
                    neuron_weights = []
                    if index == 0:
                        for n in range(self.input_size): # For each neuron of the previous layer
                            neuron_weights.append(random.randint(min_val, max_val) / 1000)
                    else:
                        for n in range(self.hidden[index - 1]["length"]):
                            neuron_weights.append(random.randint(min_val, max_val) / 1000)
                    
                    layer_weights.append(neuron_weights)
                
                weights.append(layer_weights)
            
            # Setting output neurons weights
            output_weights = []
            for i in range(self.output):
                neuron_weights = []
                for n in range(self.hidden[-1]["length"]):
                    neuron_weights.append(random.randint(min_val, max_val) / 1000)
                
                output_weights.append(neuron_weights)
            
            weights.append(output_weights)

            return weights
        
        def set_bias():
            min_val = settings["bias_range"][0] * 1000
            max_val = settings["bias_range"][1] * 1000
            bias = []

            for i in self.hidden:
                layer_bias = []
                for n in range(i["length"]):
                    layer_bias.append(random.randint(min_val, max_val) / 1000)
                
                bias.append(layer_bias)
            
            output_bias = []
            for i in range(self.output):
                output_bias.append(random.randint(min_val, max_val) / 1000)
            
            bias.append(output_bias)

            return bias

        self.weights = set_weights()
        self.bias = set_bias()

        print(self.weights)
        print(self.bias)

layers = {
    "input": {
        "length": 2
    },
    "output": {
        "length": 2,
        "activation": "relu"
    },
    "hidden": [
        {
            "length": 3,
            "activation": "relu"
        }
    ]
}
reset_settings = {
    "weights_range": [-1, 1],
    "bias_range": [-1, 1]
}

network = Network(layers)

network.reset(reset_settings)
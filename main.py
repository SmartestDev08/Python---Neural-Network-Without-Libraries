import numpy as np
import random
from math import e
from PIL import Image
import os
import itertools

def copy_array(array):
    result = []
    for i in array:
        if isinstance(i, list):
            result.append(copy_array(i))
        else:
            result.append(i)
    
    return result

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
        
        if settings["preset_weights"] == None:
            self.weights = set_weights()
        else:
            arr = np.load(f"{settings['preset_weights']}.npy")
            arr = arr.tolist()

            self.weights = arr
        self.bias = set_bias()

    def predict(self, input, train = False): # Given an input, predict the output
        if len(input) != self.input_size:
            return

        weights = self.weights
        bias = self.bias
        values = [input]

        for i, layer in enumerate(self.hidden):
            new_values = []

            for n in range(layer["length"]):
                sum1 = 0
                for index, i2 in enumerate(weights[i][n]):
                    sum1 += values[-1][index] * i2
                
                sum1 += bias[i][n]
                new_values.append(max(0, sum1))
            
            values.append(new_values)
        
        new_values = []

        for i in range(self.output):
            sum1 = 0

            for index, n in enumerate(weights[-1][i]):
                sum1 += values[-1][index] * n
            
            sum1 += bias[-1][i]
            sum1 = 1 / (1 + (e**-sum1))

            new_values.append(sum1)
        
        values.append(new_values)

        if not train:
            for i, n in enumerate(values[-1]):
                #print(n)
                #print(round(n, 5))
                values[-1][i] = round(n, 5)
            
            return values[-1]
        else:
            return values

    def train(self, input, expectedOutput, learning_rate): # Adjust weights and biases based off an input and the expected output
        
        def getRelation(layerIndex, neuronIndex): # Calculates a neuron's value relation with the cost
            if layerIndex == len(values):
                pred = expectedOutput[neuronIndex]
                C_a = 2 * (values[layerIndex - 1][neuronIndex] - pred)
                a_r = values[layerIndex - 1][neuronIndex] * (1 - values[layerIndex - 1][neuronIndex])
                return C_a * a_r
            
            relation_allR_a = 0

            for i in range(len(values[layerIndex])):
                new_layer = layerIndex + 1
                a_r = getRelation(new_layer, i)
                #print(layerIndex)
                #print(i)
                #print(neuronIndex)
                #print(weights[layerIndex - 1][i])
                r_preva = weights[layerIndex - 1][i][neuronIndex]

                relation_allR_a += a_r * r_preva
            
            return relation_allR_a
        
        values = self.predict(input, True)
        actual_output = values[-1]
        weights = self.weights
        
        total_error = 0
        for i, expected in enumerate(expectedOutput):
            target = actual_output[i]
            total_error += (target - expected) ** 2
        
        total_error = total_error / len(expectedOutput)
        
        # Updating weights which connect with the output neurons
        new_weights = copy_array(weights)

        for index1, neuronWeights in enumerate(weights[-1]):
            for index2, weight in enumerate(neuronWeights):
                output = actual_output[index1]
                target = expectedOutput[index1]

                C_a = 2 * (output - target)
                a_r = output * (1 - output)
                r_w = values[-2][index2]
                
                C_w = C_a * a_r * r_w
                new_weight_value = weight - ( learning_rate * C_w )
                
                new_weights[-1][index1][index2] = new_weight_value
        
        # Updating weights which don't connect with output neurons

        for index1, layerWeights in enumerate(weights[:-1]): # THESE ARE WEIGHTS - INDEX 1 OF WEIGHTS = INDEX 2 OF VALUES
            for index2, neuronWeights in enumerate(layerWeights):
                for index3, weight in enumerate(neuronWeights):                    
                    r_w = values[index1][index3]
                    a_r = values[index1 + 1][index2] * (1 - values[index1 + 1][index2])
                    allR_a = getRelation(index1 + 2, index2) # Recursion

                    C_w = r_w * a_r * allR_a

                    new_weights[index1][index2][index3] = weight - (learning_rate * C_w)
        self.weights = new_weights

    def saveWeights(self, name):
        weights = self.weights

        arr = np.array(weights, dtype=object)
        np.save(f"{name}.npy", arr)
layers = {
    "input": {
        "length": 256
    },
    "output": {
        "length": 2,
        "activation": "relu"
    },
    "hidden": [
        {
            "length": 23,
            "activation": "sigmoid"
        },
        {
            "length": 8,
            "activation": "sigmoid"
        }
    ]
}
reset_settings = {
    "weights_range": [-1, 1],
    "bias_range": [-1, 1],
    "preset_weights": None
}

def imageToArray(imgPath):
    img = Image.open(imgPath)
    pixels = np.asarray(img)
    pixels = pixels.tolist()

    for i1, row in enumerate(pixels):
        for i2, pix in enumerate(row):
            pixels[i1][i2] = int(pix[0] == 0)
    
    pixels = list(itertools.chain.from_iterable(pixels))

    return pixels

data = []

dataset_directory = "dataset"

for filename in os.listdir(dataset_directory):
    exp_out = [1, 0] # Circle, Cross
    if "cross" in filename:
        exp_out = [0, 1]
    
    data.append([imageToArray(f"dataset/{filename}"), exp_out])

random.shuffle(data)

training_data = data[:int(len(data) / 2)]
validation_data = data[int(len(data) / 2):len(data)]

network = Network(layers)
network.reset(reset_settings)
epochs = 100
learning_rate = 0.03

for i in range(epochs):
    if i % int(epochs / 100) == 0:
        print("epoch " + str(i))
    
    for i, case in enumerate(training_data):
        network.train(case[0], case[1], learning_rate)

network.saveWeights("savedWeights/firstTraining")

print("TRAINING DATA")
for case in training_data:
    predicted = network.predict(case[0])
    print(f"output: {predicted}")
    print(f"expected: {case[1]}")

print("VALIDATION DATA")
for case in validation_data:
    predicted = network.predict(case[0])
    print(f"output: {predicted}")
    print(f"expected: {case[1]}")
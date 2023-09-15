#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <vector>

using namespace std;

typedef struct Neuron {
	float value;
	float bias;
	vector<float> weights;
} Neuron;

float sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

float dotp(vector<float> v1, vector<float> v2) {

	float total = 0;
	for (int i=0; i < v1.size(); i++) {
		total += v1[i] * v2[i];
	}
	return total;
}


float feed_foward(Neuron &n, vector<float> inputs) {
	float total = dotp(n.weights, inputs) + n.bias;
	n.value = sigmoid(total);
	return n.value;
}

float feed_foward_n(Neuron &n, vector<Neuron> inputs) {
	vector<float> values;
	for (int i=0; i<inputs.size(); i++) {
		float value = inputs[i].value;
		values.push_back(value);
	}

	float total = dotp(n.weights, values) + n.bias;
	n.value = sigmoid(total);
	return n.value;
}

float cost_function(float value, float prediction, int count) {
	float error = value - prediction;
	float cost = (error * error) / count;		// the 1 is the amount of inputs
	return cost;
}


float sigmoid_derivative(float x) {
	return exp(-x) / pow(1 + exp(-x), 2);
}

float cost_derivative(float value, float prediction) {
	return -2 * (value - prediction);
}

float feed_foward_weight_derivative(float input) {
	return input;
}

// float feed_foward_bias_derivative(Neuron neuron, float input) {
float feed_foward_bias_derivative(float input) {
	return 1;
}

float weight_change_ratio(Neuron neuron, float value, float prediction, float input) {
	return  cost_derivative(value, prediction) * 
		sigmoid_derivative(prediction) * 
		feed_foward_weight_derivative(input);
}

float bias_change_ratio(Neuron neuron, float value, float prediction, float input) {
	return  cost_derivative(value, prediction) * 
		sigmoid_derivative(prediction) * 
		feed_foward_bias_derivative(input);
}

#define NEURONS_COUNT 10
#define INPUTS_COUNT 10
#define BIAS_LAYER1 10
#define LEARNING_RATE 0.1 

float random_0_1() {
	return (float) rand() / RAND_MAX;
}

vector<float> random_vec(int n) {
	vector<float> vec;
	for (int i=0; i<n; i++) {
		vec.push_back((float) random_0_1());
	}
	return vec;
}

float gradient_change_weight_output_layer(float value, float prediction, float input) {
	return cost_derivative(value, prediction) 
		* sigmoid_derivative(prediction) 
		* feed_foward_weight_derivative(input);
}

float gradient_change_bias_output_layer(float value, float prediction, float input) {
	return cost_derivative(value, prediction) 
		* sigmoid_derivative(prediction) 
		* feed_foward_bias_derivative(input);
}

float feed_foward_weight_derivative_last_layer(float prev_neuron_output) {
	return feed_foward_weight_derivative(prev_neuron_output);
}

float feed_foward_bias_derivative_last_layer(float prev_neuron_output) {
	return feed_foward_bias_derivative(prev_neuron_output);
}

float gradient_change_weight_hidden_layer(float value, float prediction, float prev_neuron_output, float input) {
	return cost_derivative(value,prediction)
		* sigmoid_derivative(prediction)
		* feed_foward_weight_derivative_last_layer(prev_neuron_output)
		* sigmoid_derivative(prev_neuron_output)
		* feed_foward_weight_derivative(input);
}

float gradient_change_bias_hidden_layer(float value, float prediction, float prev_neuron_output, float input) {
	return cost_derivative(value,prediction)
		* sigmoid_derivative(prediction)
		* feed_foward_bias_derivative_last_layer(prev_neuron_output)
		* sigmoid_derivative(prev_neuron_output)
		* feed_foward_bias_derivative(input);
}

int main(int argc, char *argv[]) {

	vector<float> inputs = random_vec(INPUTS_COUNT);

	vector<Neuron> neurons;

	for (int i=0; i<NEURONS_COUNT; i++) {
		Neuron n = {0.0, BIAS_LAYER1, random_vec(INPUTS_COUNT)};
		neurons.push_back(n);
	}

	for (int i=0; i<NEURONS_COUNT; i++) {
		float valor = feed_foward(neurons[i], inputs);
	}

	Neuron output_neuron = {0.0, BIAS_LAYER1, inputs};

	for (int i=0; i < NEURONS_COUNT; i++) {
		feed_foward(neurons[i], inputs);
	}

	
	feed_foward_n(output_neuron, neurons);


	float value = 3;
	float prediction = output_neuron.value;

	for (Neuron n: neurons) {
		int i = 0;
		for (float f: n.weights) {
			f -= LEARNING_RATE * gradient_change_weight_hidden_layer( value,  prediction,  n.value, inputs[i]);
			i++;
		}
	}

	int i = 0;
	for (float f : output_neuron.weights) {
		f -= LEARNING_RATE * gradient_change_weight_output_layer(value, prediction, neurons[i].value);
		i++;
	}


	float cost = cost_function(value, prediction, NEURONS_COUNT);


	// float value = 0.7;
	// float prediction = feed_foward(n, input);
	// float cost = cost_function(value, prediction);
	//
	// float wcr = weight_change_ratio(n,  value,  prediction,  input);
	// float bcr = bias_change_ratio(n,  value,  prediction,  input);

	return 0;
}

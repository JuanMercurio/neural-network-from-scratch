#include <math.h>
#include <stdio.h>

typedef struct Neuron {
	float weight;
	float bias;
} Neuron;

float sigmoid(float x) {
	return 1.0 / (1 + std::exp(-x));
}

float feed_foward(Neuron n, float input) {
	float total = n.weight * input + n.bias;
	return sigmoid(total);
}

float cost_function(float value, float prediction) {
	float error = value - prediction;
	float cost = error * error / 1;		// the 1 is the amount of inputs
	return cost;
}

int main (int argc, char *argv[]) {

	float input = 3.0;
	Neuron n = {3.0, 2.0};

	float value = 3.0;
	float prediction = feed_foward(n, input);
	float cost = cost_function(value, prediction);

	printf("The prediction is: %f\n", prediction);
	printf("the value is %f\n", value);
	printf("The cost is: %f\n", cost);

	return 0;
}

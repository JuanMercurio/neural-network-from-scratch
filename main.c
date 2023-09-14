#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Neuron {
	float weight;
	float bias;
} Neuron;

float sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
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

float sigmoid_derivative(float x) {
	return exp(-x) / pow(1 + exp(-x), 2);
}

float cost_derivative(float value, float prediction) {
	return -2 * (value - prediction);
}

float feed_foward_weight_derivative(Neuron neuron, float input) {
	return input;
}

float feed_foward_bias_derivative(Neuron neuron, float input) {
	return 1;
}

float weight_change_ratio(Neuron neuron, float value, float prediction, float input) {
	return  cost_derivative(value, prediction) * 
		sigmoid_derivative(prediction) * 
		feed_foward_weight_derivative(neuron, input);
}

float bias_change_ratio(Neuron neuron, float value, float prediction, float input) {
	return  cost_derivative(value, prediction) * 
		sigmoid_derivative(prediction) * 
		feed_foward_bias_derivative(neuron, input);
}

float random_from_0_to_1() {
	return (float) rand() / RAND_MAX; 
}

#define MINIMUM_COST .0002

int main (int argc, char *argv[]) {

	time_t seed;
	srand(time(&seed));

	float input = random_from_0_to_1();

	Neuron n = {
		random_from_0_to_1(), 
		random_from_0_to_1()
	};

	float prediction = feed_foward(n, input);
	float value = random_from_0_to_1();
	float cost = cost_function(value, prediction);

	while (cost > MINIMUM_COST) {

		float wcr = weight_change_ratio(n,  value,  prediction,  input);
		float bcr = bias_change_ratio(n,  value,  prediction,  input);

		printf("The prediction is: %f\n", prediction);
		printf("the value is %f\n", value);
		printf("The cost is: %f\n", cost);
		printf("The weight change ratio is: %f\n", wcr);
		printf("The bias change ratio is: %f\n", bcr);

		float weight_change;
		float bias_change;

		printf("Enter a value to add to the weight: ");
		scanf("%f", &weight_change);
		printf("Enter a value to add to the weight: ");
		scanf("%f", &bias_change);

		printf("\n");
		printf("----------------------------\n");
		printf("\n");

		n.bias += bias_change;
		n.weight += weight_change;
		
		prediction = feed_foward(n, input);
		cost = cost_function(value, prediction);
	}

		printf("The cost is: %f\n", cost);
		printf("the value is %f\n", value);
		printf("The prediction is: %f\n", prediction);
		printf("\n");


	return 0;
}

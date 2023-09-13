#include <stdio.h>

typedef struct Neuron {
	float weight;
	float bias;
} Neuron;


int main (int argc, char *argv[]) {
	float input = 3.0;
	Neuron n = {3.2, 2};
	
	printf("Termino el programa");
	return 0;
}

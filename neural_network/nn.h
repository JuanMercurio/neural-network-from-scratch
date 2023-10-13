#ifndef _NN_H_
#define _NN_H_

#include "../matrix/matrix.h"

typedef struct {
  int matrix_weight_count;
  Matrix **activations;
  Matrix **list_weights;
  Matrix **deltas;
  char *desc;
  float learning_rate;
} NeuralNetwork;

NeuralNetwork *neural_network_create(int layers[], int len_layers,
                                     float learning_rate);

void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) ;
void print_desc(NeuralNetwork *nn);
void NN_free(NeuralNetwork *nn);

#endif // !_NN_H_

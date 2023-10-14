#ifndef _NN_H_
#define _NN_H_

#include "../matrix/matrix.h"

typedef struct NeuralNetwork NeuralNetwork;

struct NeuralNetwork {
  int matrix_weight_count;
  Matrix **list_weights;
  char *desc;
  float learning_rate;
  float (*loss_function)(NeuralNetwork *nn, Matrix* Y, Matrix * target );
};

NeuralNetwork *neural_network_create(int layers[], int len_layers,
                                     float learning_rate);

void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) ;
void print_desc(NeuralNetwork *nn);
void NN_free(NeuralNetwork *nn);

#endif // !_NN_H_

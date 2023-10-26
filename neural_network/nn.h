#ifndef _NN_H_
#define _NN_H_

#include "../matrix/matrix.h"

typedef struct NeuralNetwork NeuralNetwork;

typedef float (*ActivationFunction)(float z);
typedef Matrix* (*ActivationDerivativeFunction)(Matrix* A);
typedef float (*ErrorFunction)(NeuralNetwork*nn, Matrix * prediction, Matrix*target);
typedef Matrix* (*LossFunctionDerivative)(Matrix* prediction, Matrix* target);

struct NeuralNetwork {
  int matrix_weight_count;
  Matrix **list_weights;
  char *desc;
  float learning_rate;
  Matrix** Zs;
  ActivationFunction *activators;
  ActivationDerivativeFunction *activators_derivatives;
  ErrorFunction loss_function;
  LossFunctionDerivative loss_function_derivative;
  // ErrorFunction *error_derivative;
};

typedef enum LossFunction {
  MSE,
  CROSS_ENTROPY,
} LossFunction;

typedef enum Activations {
  SIGMOID,
  SOFTMAX,
  RELU,
} Activations;

NeuralNetwork *neural_network_create(int layers[], int len_layers,
                                     float learning_rate);

Matrix * predict(NeuralNetwork*nn, Matrix* X);
void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) ;
void print_desc(NeuralNetwork *nn);
void NN_free(NeuralNetwork *nn);
void NN_set_layer_activation(NeuralNetwork *nn, int layer, Activations act);
void NN_set_loss_function(NeuralNetwork *nn, LossFunction function);

#endif // !_NN_H_

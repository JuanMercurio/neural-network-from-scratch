#include "matrix/matrix.h"
#include "neural_network/nn.h"

int main(int argc, char *argv[]) {

  float learning_rate = 0.2;
  int epochs = 1;
  int display_update = 200;
  int layers[] = {2, 2, 1};

  Matrix *inputs = matrix_create(4, 2);
  inputs->data[0][0] = 0;
  inputs->data[0][1] = 0;
  inputs->data[1][0] = 0;
  inputs->data[1][1] = 1;
  inputs->data[2][0] = 1;
  inputs->data[2][1] = 0;
  inputs->data[3][0] = 1;
  inputs->data[3][1] = 1;
  Matrix *outputs = matrix_create(4, 1);
  outputs->data[0][0] = 0;
  outputs->data[1][0] = 1;
  outputs->data[2][0] = 1;
  outputs->data[3][0] = 0;

  int len_layers = sizeof(layers) / sizeof(layers[0]);

  NeuralNetwork *nn = neural_network_create(layers, len_layers, learning_rate);
  fit(nn, inputs, outputs, epochs, display_update);

  print_desc(nn);

  matrix_free(outputs);
  matrix_free(inputs);
  NN_free(nn);

  return 0;
}

#include "matrix/matrix.h"
#include "mnist/mnist.h"
#include "neural_network/nn.h"

// int main(int argc, char *argv[]) {
int main() {

  float learning_rate = 0.2;
  int epochs = 20000;
  int display_update = 200;

  // Matrix *inputs = mnist_get_input_data();
  // Matrix *outputs = mnist_get_output_data();

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

  // we need on hidden relu and on the last softmax
  // int layers[] = {inputs->cols, 10, 10, outputs->cols};
  int layers[] = {2, 2, 1};

  int len_layers = sizeof(layers) / sizeof(layers[0]);

  NeuralNetwork *nn = neural_network_create(layers, len_layers, learning_rate);
  fit(nn, inputs, outputs, epochs, display_update);

  print_desc(nn);

  matrix_free(outputs);
  matrix_free(inputs);
  NN_free(nn);

  return 0;
}

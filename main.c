#include "matrix/matrix.h"
#include "mnist/mnist.h"
#include "neural_network/nn.h"

// int main(int argc, char *argv[]) {
int main() {

  // float learning_rate = 0.2;
  // int epochs = 20000;
  // int display_update = 200;

  Matrix *inputs = mnist_get_input_data();
  Matrix *outputs = mnist_get_output_data();

  int layers[] = {inputs->cols, 2, outputs->cols};

  // int len_layers = sizeof(layers) / sizeof(layers[0]);

  // NeuralNetwork *nn = neural_network_create(layers, len_layers,
  // learning_rate); fit(nn, inputs, outputs, epochs, display_update);
  //
  // print_desc(nn);
  //
  // matrix_free(outputs);
  // matrix_free(inputs);
  // NN_free(nn);

  return 0;
}

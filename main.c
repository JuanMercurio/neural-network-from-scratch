#include <stdio.h>
#include <stdlib.h>

float random_number() {
  return rand() / (float)RAND_MAX; // createa a 0.12341234 random number
}

typedef struct {
  int cols;
  int rows;
  float **data;
} Matrix;

typedef struct {
  Matrix **weights;
} NeuralNetwork;

// should be freed
Matrix *matrix_allocate(int row_count, int col_count) {
  Matrix *matrix = malloc(sizeof(Matrix));

  matrix->data = malloc(row_count * sizeof(float));

  for (int i = 0; i < row_count; i++) {
    matrix->data[i] = malloc(sizeof(float) * col_count);
  }

  return matrix;
}

//  -----------------------------------------------------

void weights_initilize(float **weights, int row_count, int col_count) {

  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++) {
      weights[i][j] = 1.1; // random_number();
    }
  }
}

NeuralNetwork create_neural_network(int layers[], int len_layers,
                                    float learning_rate) {

  if (len_layers < 3)
    exit(1);

  Matrix **weights = malloc(sizeof(Matrix) * (len_layers - 1));

  for (int i = 0; i < len_layers - 2; i++) {
    int row_count = layers[i];
    int col_count =
        layers[i + 1] + 1; // this + 1 is because of the bias methos that we use

    Matrix *m = matrix_allocate(row_count, col_count);

    weights_initilize(m->data, m->rows,
                      m->cols); // +1 because of bias
    weights[i] = m;
  }

  // the last 2 layers (last hidden to output)
  int row_count = layers[len_layers - 2]; // +1 because of bias
  int col_count = layers[len_layers - 1];

  Matrix *last_weights = matrix_allocate(row_count, col_count);

  weights_initilize(last_weights->data, last_weights->rows, last_weights->cols);

  NeuralNetwork nn;
  nn.weights = weights;

  return nn;
}

void print_weights(NeuralNetwork nn) {
  printf("%f\n", nn.weights[0]->data[0][0]);
}

int main(int argc, char *argv[]) {

  float learning_rate = 0.1;
  int layers[] = {2, 2, 1};
  int len_layers = sizeof(layers) / sizeof(layers[0]);

  NeuralNetwork nn = create_neural_network(layers, len_layers, learning_rate);
  print_weights(nn);

  return 0;
}

#include <math.h>
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
  int matrix_weight_count;
} NeuralNetwork;

void matrix_times_scalar(Matrix *matrix, float scalar) {
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->rows; j++) {
      matrix->data[i][j] /= scalar;
    }
  }
}

// should be freed
Matrix *matrix_allocate(int row_count, int col_count) {
  Matrix *matrix = malloc(sizeof(Matrix));

  matrix->data = malloc(row_count * sizeof(float *));
  matrix->rows = row_count;
  matrix->cols = col_count;

  for (int i = 0; i < row_count; i++) {
    matrix->data[i] = malloc(sizeof(float) * col_count);
  }

  return matrix;
}

//  -----------------------------------------------------

void weights_initilize(Matrix *matrix) {

  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      matrix->data[i][j] =
          (float)rand() / RAND_MAX; // random_number between 0 and 1 ;
    }
  }

  // normalize values with the sqrt of the amount of weights
  matrix_times_scalar(matrix, sqrt((double)matrix->rows * matrix->cols));
}

NeuralNetwork create_neural_network(int layers[], int len_layers,
                                    float learning_rate) {

  int count_matrix = 0;
  if (len_layers < 3)
    exit(1);

  Matrix **weights = malloc(sizeof(Matrix *) * (len_layers - 1));

  for (int i = 0; i < len_layers - 2; i++) {
    int row_count = layers[i] + 1;
    int col_count = layers[i + 1] + 1; // this + 1 is because of the bias

    Matrix *m = matrix_allocate(row_count, col_count);

    weights_initilize(m);
    weights[i] = m;

    count_matrix++;
  }

  // the last 2 layers (last hidden to output)
  int row_count = layers[len_layers - 2] + 1; // +1 because of bias
  int col_count = layers[len_layers - 1];

  Matrix *last_weights = matrix_allocate(row_count, col_count);

  weights_initilize(last_weights);
  weights[len_layers - 2] = last_weights;
  count_matrix++;

  NeuralNetwork nn = {weights, count_matrix};

  return nn;
}

void print_weights(NeuralNetwork nn) {

  printf("\n");
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    for (int j = 0; j < nn.weights[i]->rows; j++) {
      for (int k = 0; k < nn.weights[i]->cols; k++) {
        printf("%-10g ", nn.weights[i]->data[j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  float learning_rate = 0.1;
  int layers[] = {2, 2, 1};
  int len_layers = sizeof(layers) / sizeof(layers[0]);

  NeuralNetwork nn = create_neural_network(layers, len_layers, learning_rate);
  // printf("%d\n", nn.matrix_weight_count);
  print_weights(nn);

  return 0;
}

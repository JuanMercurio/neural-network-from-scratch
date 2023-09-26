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
  Matrix **list_weights; // its a list of matrices
  int matrix_weight_count;
  Matrix **neurons;
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

void insert_ones_column(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    m->data[i] = realloc(m->data[i], (m->rows + 1) * sizeof(float *));
    m->data[i][m->cols] = 1.0;
  }
  m->cols++;
}

void weights_initilize(Matrix *matrix) {

  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      matrix->data[i][j] = (float)rand() / RAND_MAX;
      // random_number between 0 and 1 ;
      // TODO: BEST random between
    }
  }
}

float sigmoid(float x) { return 1.0 / (1 - exp(-x)); }
float sigmoid_derivative(float x) { return x * (1 - x); }

Matrix *matrix_dot(Matrix *m1, Matrix *m2) {

  Matrix *final = matrix_allocate(m1->rows, m2->cols);

  for (int i = 0; i < final->rows; i++) {
    for (int j = 0; j < final->cols; j++) {
      float value = 0;
      for (int k = 0; k < final->cols; k++) {
        value += m1->data[i][k] * m2->data[k][j];
      }
      final->data[i][j] = value;
    }
  }
  return final;
}

// void fit_partial(NeuralNetwork nn, Matrix *input, float *output) {
//
//   int neurons_count = nn.list_weights[0]->cols; // this is static for now
//   float *activations = malloc(sizeof(float) * neurons_count);
//   *activations = *input;
//
//   for (int i = 0; i < nn.matrix_weight_count; i++) {
//   }
// }
//
// void fit(NeuralNetwork nn, Matrix *inputs, Matrix *outputs, int epochs,
//          int display_update) {
//   insert_ones_column(inputs);
//
//   for (int i = 0; i < epochs; i++) {
//     fit_partial(nn, inputs, outputs);
//   }
//
//   if (epochs == 0 || epochs + 1 % display_update == 0) {
//     printf("saturnoooooooooooooo\n");
//   }
// }
//
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
    // normalize values with the sqrt of the amount of weights
    matrix_times_scalar(m, sqrt((double)m->rows * m->cols));
    weights[i] = m;

    count_matrix++;
  }

  // the last 2 layers (last hidden to output)
  int row_count = layers[len_layers - 2] + 1; // +1 because of bias
  int col_count = layers[len_layers - 1];

  Matrix *last_weights = matrix_allocate(row_count, col_count);

  weights_initilize(last_weights);
  // normalize values with the sqrt of the amount of weights
  matrix_times_scalar(last_weights,
                      sqrt((double)last_weights->rows * last_weights->cols));
  weights[len_layers - 2] = last_weights;
  count_matrix++;

  NeuralNetwork nn = {weights, count_matrix};

  return nn;
}

Matrix *matrix_create_random(int rows, int cols) {
  Matrix *m = matrix_allocate(rows, cols);
  weights_initilize(m);
  return m;
}

void matrix_print(Matrix *m) {
  for (int j = 0; j < m->rows; j++) {
    for (int k = 0; k < m->cols; k++) {
      printf("%-10g ", m->data[j][k]);
    }
    printf("\n");
  }
}

void print_weights(NeuralNetwork nn) {

  printf("\n");
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    matrix_print(nn.list_weights[i]);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  float learning_rate = 0.1;
  int layers[] = {2, 2, 1};

  Matrix *inputs = matrix_create_random(2, 2);
  Matrix *outputs = matrix_create_random(2, 2);

  int len_layers = sizeof(layers) / sizeof(layers[0]);

  // NeuralNetwork nn = create_neural_network(layers, len_layers,
  // learning_rate);

  Matrix *a = matrix_allocate(2, 2);
  a->data[0][0] = 0.0;
  a->data[0][1] = 1.0;
  a->data[1][0] = 1.0;
  a->data[1][1] = 0.0;

  Matrix *b = matrix_allocate(1, 2);
  b->data[0][0] = 1.0;
  b->data[0][1] = 1.0;

  Matrix *result = matrix_dot(b, a);
  matrix_print(result);

  // print_weights(nn);
  // fit(nn, inputs, outputs, 1, 100);

  return 0;
}

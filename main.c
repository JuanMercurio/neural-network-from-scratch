#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float random_number() {
  return rand() / (float)RAND_MAX; // createa a 0.12341234 random number
}

typedef struct {
  int cols;
  int rows;
  float **data;
} Matrix;

typedef struct {
  int matrix_weight_count;
  Matrix **activations;
  Matrix **list_weights; // its a list of matrices
  Matrix **deltas;
  char *desc;
  float alpha;
} NeuralNetwork;

void matrix_print(Matrix *m) {
  for (int j = 0; j < m->rows; j++) {
    for (int k = 0; k < m->cols; k++) {
      printf("%-10g ", m->data[j][k]);
    }
    printf("\n");
  }
}

void matrix_free(Matrix *m) {
  free(m->data);
  free(m);
}

void print_deltas(NeuralNetwork nn) {
  printf("Deltas:\n");
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    matrix_print(nn.deltas[i]);
  }
}

void print_desc(NeuralNetwork nn) {
  printf("Shape of NN\n");
  printf("%s\n", nn.desc);
}

void print_weights(NeuralNetwork nn) {

  printf("\n");
  printf("----------------WEIGHTS----------------------\n");
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    matrix_print(nn.list_weights[i]);
    printf("\n");
  }
}

void print_activations(NeuralNetwork nn) {

  printf("\n");
  printf("----------------ACTIVATIONS----------------------\n");
  for (int i = 0; i < nn.matrix_weight_count + 1; i++) {
    matrix_print(nn.activations[i]);
    printf("\n");
  }
}
void matrix_times_scalar_transform(Matrix *matrix, float scalar) {
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
double random_normal_distribution() {

  double u1 = ((double)rand() / RAND_MAX);
  double u2 = ((double)rand() / RAND_MAX);

  double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  return z0;
}

void weights_initilize(Matrix *matrix) {

  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      matrix->data[i][j] = (float)random_normal_distribution();
      // random_number between 0 and 1 ;
      // TODO: BEST random between
    }
  }
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float sigmoid_derivative(float x) { return x * (1 - x); }

// this destroys the first matrix
Matrix *matrix_diff(Matrix *m1, Matrix *m2) {

  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    printf("matrix_diff\n");
    fflush(stdout);
    abort();
  }

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      m1->data[i][j] = m1->data[i][j] - m2->data[i][j];
    }
  }

  return m1;
}
Matrix *matrix_operation_single(Matrix *m, float(operation(float))) {

  Matrix *final = matrix_allocate(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = operation(m->data[i][j]);
    }
  }

  return final;
}

Matrix *matrix_operation(Matrix *m1, Matrix *m2,
                         float (*operation)(float, float)) {

  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    printf("matrix_operation\n");
    fflush(stdout);
    abort();
  }

  Matrix *final = matrix_allocate(m1->rows, m1->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      final->data[i][j] = operation(m1->data[i][j], m2->data[i][j]);
    }
  }

  return final;
}

Matrix *matrix_transform(Matrix *m, float (*transfomer)(float)) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->data[i][j] = transfomer(m->data[i][j]);
    }
  }

  return m;
}

Matrix *matrix_times_scalar(Matrix *m, float scalar) {

  Matrix *final = matrix_allocate(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {

    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = m->data[i][j] * scalar;
    }
  }

  return final;
}

Matrix *matrix_dot(Matrix *m1, Matrix *m2) {

  if (m2->cols == 1 && m2->rows == 1) {
    return matrix_times_scalar(m1, m2->data[0][0]);
  }

  if (m1->cols == 1 && m1->rows == 1) {
    return matrix_times_scalar(m2, m1->data[0][0]);
  }

  if (m1->cols == 1 && m2->rows == 1) {
    Matrix *final = matrix_allocate(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++) {
      for (int j = 0; j < m2->cols; j++) {
        final->data[i][j] = m1->data[i][0] * m2->data[0][j];
      }
    }
    return final;
  }

  if (m1->cols != m2->rows) {
    printf("matrix_dot\n");
    fflush(stdout);
    abort();
  }
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

void matrix_change_size(Matrix *m, int rows, int cols) {
  m->cols = cols;
  m->rows = rows;
  m->data = realloc(m->data, sizeof(float *) * rows);

  for (int i = 0; i < rows; i++) {
    m->data[i] = realloc(m->data[i], sizeof(float) * cols);
  }
}

// for (int i = 0; i < m->rows; i++) {
//   for (int j = i + 1; j < m->cols; j++) {
//     float temp = m->data[i][j];
//     m->data[i][j] = m->data[j][i];
//     m->data[j][i] = temp;
//   }
// }
//
Matrix *matrix_t(Matrix *m) {

  Matrix *transpose = matrix_allocate(m->cols, m->rows);

  for (int i = 0; i < transpose->rows; i++) {
    for (int j = 0; j < transpose->cols; j++) {
      transpose->data[i][j] = m->data[j][i];
    }
  }

  return transpose;
}

float linear_times(float x, float y) { return x * y; }

void nn_free_activations(NeuralNetwork *nn) {
  for (int i = 0; i < nn->matrix_weight_count; i++) {
    free(nn->activations[i]->data);
    free(nn->activations[i]);
  }
}

void matrix_update_sum(Matrix *to_update, Matrix *updater) {

  for (int i = 0; i < to_update->rows; i++)
    for (int j = 0; j < to_update->cols; j++) {
      to_update->data[i][j] += updater->data[i][j];
    }
}

void nn_free_deltas(NeuralNetwork *nn) {
  for (int i = 0; i < nn->matrix_weight_count; i++) {
    free(nn->deltas[i]->data);
    free(nn->deltas[i]);
  }
}

void fit_partial(NeuralNetwork *nn, Matrix *input, Matrix *outputs) {

  int activation_count = nn->matrix_weight_count + 1;
  int activation_size = (sizeof(Matrix *) * activation_count);
  nn->activations = malloc(activation_size);
  nn->activations[0] = input;

  // feedfoward
  for (int i = 0; i < nn->matrix_weight_count; i++) {

    Matrix *net = matrix_dot(nn->activations[i], nn->list_weights[i]);
    Matrix *out = matrix_transform(net, sigmoid);

    nn->activations[i + 1] = out;
  }

  // deltas
  Matrix *error_derivative =
      matrix_diff(nn->activations[nn->matrix_weight_count], outputs);

  Matrix **deltas = malloc(sizeof(Matrix *) * activation_count);

  Matrix *d = matrix_operation_single(nn->activations[nn->matrix_weight_count],
                                      sigmoid_derivative);
  Matrix *delta = matrix_operation(d, error_derivative, linear_times);
  matrix_free(d);
  deltas[0] = delta;

  for (int i = 0; i < activation_count - 2; i++) {

    Matrix *transpose_weight =
        matrix_t(nn->list_weights[nn->matrix_weight_count - i - 1]);
    Matrix *delta = matrix_dot(deltas[i], transpose_weight);
    Matrix *d = matrix_operation_single(
        nn->activations[activation_count - 2 - i], sigmoid_derivative);
    delta = matrix_operation(delta, d, linear_times);
    deltas[i + 1] = delta;

    matrix_free(d);
    matrix_free(transpose_weight);
  }

  nn->deltas = deltas;
  // printf("IMPRIMO DELTAS\n");
  // for (int i = 0; i < nn->matrix_weight_count; i++) {
  //
  //   matrix_print(deltas[i]);
  // }

  for (int i = 0; i < nn->matrix_weight_count; i++) {
    Matrix *activations_transpose = matrix_t(nn->activations[i]);
    Matrix *dot = matrix_dot(activations_transpose,
                             deltas[nn->matrix_weight_count - 1 - i]);

    Matrix *final_change = matrix_times_scalar(dot, -1.0 * nn->alpha);
    matrix_update_sum(nn->list_weights[i], final_change);

    matrix_free(activations_transpose);
    matrix_free(final_change);
    matrix_free(dot);
  }

  // cambiar weights

  // free todo
  //  free(nn->deltas);
  //  free(nn->activations);

  nn_free_deltas(nn);
  nn_free_activations(nn);
}

Matrix *predict(NeuralNetwork *nn, Matrix *x) {
  // insert_ones_column(x); // esto por ahora nomas
  Matrix *p = NULL;
  p = matrix_transform(matrix_dot(x, nn->list_weights[0]), sigmoid);
  x = p;

  for (int i = 1; i < nn->matrix_weight_count; i++) {
    p = matrix_transform(matrix_dot(x, nn->list_weights[i]), sigmoid);
    matrix_free(x);
    x = p;
  }

  return p;
}

float *predict_list(NeuralNetwork *nn, Matrix *inputs) {
  float *predictions = malloc(sizeof(float *) * inputs->rows);
  Matrix *input = NULL;

  for (int i = 0; i < inputs->rows; i++) {
    input = matrix_allocate(1, inputs->cols);
    for (int j = 0; j < inputs->rows; j++) {
      input->data[0][j] = inputs->data[i][j];
    }
    Matrix *pred = predict(nn, input);
    predictions[i] = pred->data[0][0];
  }
  return predictions;
}
float mse(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs) {
  float *predictions = predict_list(nn, inputs);
  float mse = 0;
  for (int i = 0; i < outputs->rows; i++) {
    float error = predictions[i] - outputs->data[i][0];
    mse += error * error; // Square the error and add it to mse
  }

  mse /= outputs->rows; // Divide by 2 for the mean squared error

  return mse;
}

void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) {

  insert_ones_column(inputs);

  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < inputs->rows; j++) {

      Matrix *input = matrix_allocate(1, inputs->cols);
      input->data[0] = inputs->data[j];
      Matrix *target = matrix_allocate(1, outputs->cols);
      target->data[0] = outputs->data[j];

      fit_partial(nn, input, target);

      // matrix_free(input); this is freed when we free the activations
      matrix_free(target);
    }
    if (i % 1 == 0) {
      float loss = mse(nn, inputs, outputs);
      printf("el loss esta siendo %g\n", loss);
    }
  }
}

char *description_create(int *layers, int len_layers) {

  char *desc = malloc(sizeof(char) * 2 * len_layers);
  char *a = malloc(sizeof(char));

  sprintf(a, "%d", layers[0]);
  strcat(desc, a);

  for (int i = 1; i < len_layers; i++) {
    sprintf(a, "%d", layers[i]);
    strcat(desc, "-");
    strcat(desc, a);
  }
  return desc;
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
    // normalize values with the sqrt of the amount of weights
    matrix_times_scalar_transform(m, sqrt(1 / (double)m->rows));
    weights[i] = m;

    count_matrix++;
  }

  // the last 2 layers (last hidden to output)
  int row_count = layers[len_layers - 2] + 1; // +1 because of bias
  int col_count = layers[len_layers - 1];

  Matrix *last_weights = matrix_allocate(row_count, col_count);

  weights_initilize(last_weights);
  // normalize values with the sqrt of the amount of weights
  matrix_times_scalar_transform(last_weights,
                                sqrt(1 / (double)last_weights->rows));
  weights[len_layers - 2] = last_weights;
  count_matrix++;

  char *desc = description_create(layers, len_layers);

  NeuralNetwork nn = {count_matrix, NULL, weights, NULL, desc, 0.3};

  return nn;
}

Matrix *matrix_create_random(int rows, int cols) {
  Matrix *m = matrix_allocate(rows, cols);
  weights_initilize(m);
  return m;
}

int main(int argc, char *argv[]) {

  float learning_rate = 0.1;
  int layers[] = {2, 2, 1};

  Matrix *inputs = matrix_allocate(4, 2);
  inputs->data[0][0] = 0;
  inputs->data[0][1] = 0;
  inputs->data[1][0] = 1;
  inputs->data[1][1] = 0;
  inputs->data[2][0] = 0;
  inputs->data[2][1] = 1;
  inputs->data[3][0] = 1;
  inputs->data[3][1] = 1;
  Matrix *outputs = matrix_allocate(4, 1);
  outputs->data[0][0] = 0;
  outputs->data[1][0] = 1;
  outputs->data[2][0] = 1;
  outputs->data[3][0] = 0;
  // matrix_print(inputs);
  // matrix_print(outputs);
  //
  // Matrix *inputs = matrix_create_random(4, 2);
  // Matrix *outputs = matrix_create_random(4, 1);

  int len_layers = sizeof(layers) / sizeof(layers[0]);

  // float(*s)[col] = (float(*)[col])n;
  // Cast a block of memory that containt a static matrix of floats
  // into a matrix that can be accesed like this: s[0][0]

  NeuralNetwork nn = create_neural_network(layers, len_layers, learning_rate);
  fit(&nn, inputs, outputs, 100, 10);

  Matrix *x0 = matrix_allocate(1, 2);
  x0->data[0][0] = 0;
  x0->data[0][1] = 0;
  Matrix *x2 = matrix_allocate(1, 2);
  x2->data[0][0] = 1;
  x2->data[0][1] = 0;
  Matrix *x1 = matrix_allocate(1, 2);
  x1->data[0][0] = 0;
  x1->data[0][1] = 1;
  Matrix *x3 = matrix_allocate(1, 2);
  x3->data[0][0] = 1;
  x3->data[0][1] = 1;

  // Matrix *p0 = predict(&nn, x0);
  // Matrix *p1 = predict(&nn, x1);
  // Matrix *p2 = predict(&nn, x2);
  // Matrix *p3 = predict(&nn, x3);
  //
  // int result0 = (p0->data[0][0] < 0.5) ? 0 : 1;
  // int result1 = (p1->data[0][0] < 0.5) ? 0 : 1;
  // int result2 = (p2->data[0][0] < 0.5) ? 0 : 1;
  // int result3 = (p3->data[0][0] < 0.5) ? 0 : 1;
  //
  // float result_float_0 = p0->data[0][0];
  // float result_float_1 = p1->data[0][0];
  // float result_float_2 = p2->data[0][0];
  // float result_float_3 = p3->data[0][0];
  //
  // printf("resultados: \n");
  // printf("%d- %d - %d - %d\n", result0, result1, result2, result3);
  // printf("%g- %g - %g - %g\n", result_float_0, result_float_1,
  // result_float_2,
  //        result_float_3);

  // // print_activations(nn);
  print_desc(nn);
  // print_deltas(nn);
  //

  return 0;
}

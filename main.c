#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix/matrix.h"

int print;

typedef struct {
  int matrix_weight_count;
  Matrix **activations;
  Matrix **list_weights;
  Matrix **deltas;
  char *desc;
  float learning_rate;
} NeuralNetwork;

void insert_ones_column(Matrix *m) { matrix_add_column(m, 1.0); }

double random_uniform_distribution_number(float min, float max) {

  float random = (float)rand() / RAND_MAX;
  double final = min + random * (max - min);

  // return random;
  return final;
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
void weights_initilize(Matrix *matrix) {

  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      matrix->data[i][j] = (float)random_uniform_distribution_number(
          -1.0 /* / sqrt(matrix->cols) */, 1.0 /* / sqrt(matrix->cols) */);
    }
  }
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float sigmoid_derivative(float x) { return x * (1 - x); }

float relu(float x) {
  if (x > 0)
    return x;
  else
    return 0;
}

float relu_derivative(float x) {
  if (x > 0)
    return 1;
  else
    return 0;
}

void NN_free_activations(NeuralNetwork *nn) {
  for (int i = 0; i < nn->matrix_weight_count; i++) {
    free(nn->activations[i]->data);
    free(nn->activations[i]);
  }
}

void NN_free_deltas(NeuralNetwork *nn) {
  for (int i = 0; i < nn->matrix_weight_count; i++) {
    free(nn->deltas[i]->data);
    free(nn->deltas[i]);
  }
}

Matrix *predict(NeuralNetwork *, Matrix *);

float *predict_list(NeuralNetwork *nn, Matrix *inputs) {
  float *predictions = malloc(sizeof(float *) * inputs->rows);
  Matrix *input = NULL;

  for (int i = 0; i < inputs->rows; i++) {
    input = matrix_create(1, inputs->cols);
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
    mse += error * error;
  }

  mse /= outputs->rows;

  return mse;
}

void suffle_data(Matrix *m, Matrix *m2) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      int index = ((int)rand() / INT_MAX) % m->rows;
      float *temp = m->data[i];
      m->data[i] = m->data[index];
      m->data[index] = temp;
      temp = m2->data[i];
      m2->data[i] = m2->data[index];
      m2->data[index] = temp;
    }
  }
}

void NN_create_activations(NeuralNetwork *nn, int activation_count,
                           Matrix *input) {

  int activation_size = (sizeof(Matrix *) * activation_count);
  nn->activations = malloc(activation_size);
  nn->activations[0] = matrix_dup(input);

  // feedfoward
  for (int i = 0; i < nn->matrix_weight_count; i++) {

    Matrix *net = matrix_dot(nn->activations[i], nn->list_weights[i]);
    Matrix *out = matrix_transform(net, sigmoid);
    // Matrix *out = matrix_transform(net, relu);

    nn->activations[i + 1] = out;
  }

  if (print == 1) {
    printf("Before training weights:\n");
    print_weights(*nn);
  }

  if (print == 1) {
    print_activations(*nn);
  }
}

void NN_create_deltas(NeuralNetwork *nn, int activation_count,
                      Matrix *outputs) {

  Matrix *error_derivative =
      matrix_substract(nn->activations[nn->matrix_weight_count], outputs);
  // matrix_operation(nn->activations[nn->matrix_weight_count], outputs,
  // substract);
  // matrix_diff(outputs, nn->activations[nn->matrix_weight_count]);

  Matrix **deltas = malloc(sizeof(Matrix *) * activation_count);

  Matrix *d = matrix_element_operation(nn->activations[nn->matrix_weight_count],
                                       sigmoid_derivative);
  // Matrix *d =
  // matrix_operation_single(nn->activations[nn->matrix_weight_count],
  //                                     relu_derivative);
  // Matrix *delta =
  // matrix_element_to_element_operation(d, error_derivative, times);
  Matrix *delta = matrix_times(d, error_derivative);
  matrix_free(d);
  deltas[0] = delta;

  for (int i = 0; i < activation_count - 2; i++) {

    Matrix *transpose_weight =
        matrix_transpose(nn->list_weights[nn->matrix_weight_count - i - 1]);
    Matrix *delta = matrix_dot(deltas[i], transpose_weight);
    Matrix *d = matrix_element_operation(
        nn->activations[activation_count - 2 - i], sigmoid_derivative);
    // nn->activations[activation_count - 2 - i], relu_derivative);
    // delta = matrix_element_to_element_operation(delta, d, times);
    delta = matrix_times(delta, d);
    deltas[i + 1] = delta;

    matrix_free(d);
    matrix_free(transpose_weight);
  }

  nn->deltas = deltas;
  if (print == 1) {
    print_deltas(*nn);
  }
}

void NN_udpate_weights(NeuralNetwork *nn) {

  for (int i = 0; i < nn->matrix_weight_count; i++) {
    Matrix *activations_transpose = matrix_transpose(nn->activations[i]);
    Matrix *dot = matrix_dot(activations_transpose,
                             nn->deltas[nn->matrix_weight_count - 1 - i]);

    Matrix *final_change = matrix_scale(dot, -nn->learning_rate);
    if (print == 1) {
      printf("final changes to weights");
      matrix_print(final_change);
    }
    matrix_add_inplace(nn->list_weights[i], final_change);

    matrix_free(activations_transpose);
    matrix_free(final_change);
    matrix_free(dot);
  }
  if (print == 1) {
    print_weights(*nn);
  }
}

Matrix *predict(NeuralNetwork *nn, Matrix *x) {

  if (x->cols != nn->list_weights[0]->rows) {
    insert_ones_column(x);
  }

  Matrix *p = NULL;
  Matrix *dot = matrix_dot(x, nn->list_weights[0]);
  p = matrix_transform(dot, sigmoid);
  // p = matrix_transform(dot,  relu);
  matrix_free(dot);
  // here we dont free x because x is the inputs
  x = p;

  for (int i = 1; i < nn->matrix_weight_count; i++) {
    dot = matrix_dot(x, nn->list_weights[i]);
    p = matrix_transform(dot, sigmoid);
    matrix_free(dot);
    // p = matrix_transform(dot, relu);
    matrix_free(x);
    x = p;
  }

  return p;
}

void fit_partial(NeuralNetwork *nn, Matrix *input, Matrix *outputs) {

  int activation_count = nn->matrix_weight_count + 1;

  // fow prop
  NN_create_activations(nn, activation_count, input);

  // back prop
  NN_create_deltas(nn, activation_count, outputs);
  NN_udpate_weights(nn);

  // debug
  // Matrix *prediction = predict(nn, input);
  // printf("%f %f = %f | Predicted: %f\n", input->data[0][0],
  // input->data[0][1],
  //        outputs->data[0][0], prediction->data[0][0]);
  //
  NN_free_deltas(nn);
  NN_free_activations(nn);
}

void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) {

  insert_ones_column(inputs);

  for (int i = 0; i < epochs; i++) {
    suffle_data(inputs, outputs);
    for (int j = 0; j < inputs->rows; j++) {

      Matrix *input = matrix_create(1, inputs->cols);
      input->data[0] = inputs->data[j];
      Matrix *target = matrix_create(1, outputs->cols);
      target->data[0] = outputs->data[j];

      fit_partial(nn, input, target);

      // matrix_free(input); this is freed when we free the activations
      matrix_free(target);
    }
    if (i % display_update == 0) {
      float loss = mse(nn, inputs, outputs);
      printf("Current loss: %g\n", loss);
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

    Matrix *m = matrix_create(row_count, col_count);

    weights_initilize(m);
    // normalize values with the sqrt of the amount of neurons in layer
    // matrix_times_scalar_transform(m, sqrt(1 / (double)m->cols));
    weights[i] = m;

    count_matrix++;
  }

  // the last 2 layers (last hidden to output)
  int row_count = layers[len_layers - 2] + 1; // +1 because of bias
  int col_count = layers[len_layers - 1];

  Matrix *last_weights = matrix_create(row_count, col_count);

  weights_initilize(last_weights);
  // normalize values with the sqrt of the amount of neurons in layer
  // matrix_times_scalar_transform(last_weights, sqrt(1 /
  // (double)last_weights->cols));
  weights[len_layers - 2] = last_weights;
  count_matrix++;

  char *desc = description_create(layers, len_layers);

  NeuralNetwork nn = {count_matrix, NULL, weights, NULL, desc, learning_rate};

  if (print == 1) {
    print_weights(nn);
  }

  return nn;
}

int main(int argc, char *argv[]) {

  float learning_rate = 0.2;
  int epochs = 20000;
  int display_update = 100;
  int layers[] = {2, 2, 1};
  print = 0;

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

  NeuralNetwork nn = create_neural_network(layers, len_layers, learning_rate);
  fit(&nn, inputs, outputs, epochs, display_update);

  return 0;
}

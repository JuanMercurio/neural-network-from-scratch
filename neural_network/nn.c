
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

int print = 1;

void list_forEach(Matrix **list, int size, void (*procedure)(Matrix *)) {
  for (int i = 0; i < size; i++) {
    procedure(list[i]);
  }
}

void list_free(Matrix **list, int size, void (*destructor)(Matrix *)) {
  for (int i = 0; i < size; i++) {
    destructor(list[i]);
  }

  free(list);
}

void insert_ones_column(Matrix *m) { matrix_add_column(m, 1.0); }

double random_uniform_distribution_number(float min, float max) {

  float random = (float)rand() / RAND_MAX;
  double final = min + random * (max - min);

  // return random;
  return final;
}

void print_desc(NeuralNetwork *nn) {
  printf("Shape of NN\n");
  printf("%s\n", nn->desc);
}

void print_weights(NeuralNetwork *nn) {

  printf("\n");
  printf("----------------WEIGHTS----------------------\n");
  for (int i = 0; i < nn->matrix_weight_count; i++) {
    matrix_print(nn->list_weights[i]);
    printf("\n");
  }
  list_forEach(nn->list_weights, nn->matrix_weight_count, matrix_print);
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

Matrix *predict(NeuralNetwork *, Matrix *);

float *predict_list(NeuralNetwork *nn, Matrix *inputs) {
  float *predictions = malloc(sizeof(float *) * inputs->rows);
  Matrix *input = NULL;

  for (int i = 0; i < inputs->rows; i++) {
    input = matrix_create(1, inputs->cols);
    for (int j = 0; j < inputs->cols; j++) {
      input->data[0][j] = inputs->data[i][j];
    }
    Matrix *pred = predict(nn, input);
    matrix_free(input);
    predictions[i] = pred->data[0][0];
    matrix_free(pred);
  }
  return predictions;
}

float mse(NeuralNetwork *nn, Matrix *X, Matrix *Y) {
  Matrix *prediction = predict(nn, X);
  float error = 0;
  for (int i = 0; i < prediction->rows; i++) {
    error += pow(prediction->data[i][0] - Y->data[i][0], 2);
  }

  error /= prediction->cols;
  return error;
}

float calculate_loss(NeuralNetwork *nn, Matrix *inputs, Matrix *targets) {

  float network_error = 0;
  for (int i = 0; i < inputs->rows; i++) {
    Matrix *input = matrix_get_rows(inputs, i, 1);
    Matrix *target = matrix_get_rows(targets, i, 1);
    network_error += nn->loss_function(nn, input, target);
    // network_error += mse(nn, input, target);
  }

  network_error /= targets->rows;

  return network_error;
}

void suffle_data(Matrix *m, Matrix *m2) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      // this is wrong we need to allcoate the *temp
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

// this can reuse memory and be more optimal (as well as continuous memory in
// all matrix)
Matrix **NN_create_activations(NeuralNetwork *nn, int activation_count,
                               Matrix *input) {

  int activation_size = activation_count * sizeof(Matrix *);
  Matrix **activations = malloc(activation_size);
  // nn->activations = activations;
  // nn->activations[0] = matrix_dup(input);
  activations[0] = matrix_dup(input);
  // feedfoward
  for (int i = 0; i < nn->matrix_weight_count; i++) {

    Matrix *net = matrix_dot(activations[i], nn->list_weights[i]);
    Matrix *out = matrix_transform(net, sigmoid);
    matrix_free(net);
    // Matrix *out = matrix_transform(net, relu);

    // nn->activations[i + 1] = out;
    activations[i + 1] = out;
  }

  if (print == 1) {
    printf("Before training weights:\n");
    print_weights(nn);
  }

  if (print == 1) {
    printf("Activations\n");
    list_forEach(activations, activation_count, matrix_print);
    list_forEach(activations, activation_count, matrix_print);
  }
  return activations;
}

Matrix **NN_create_deltas(NeuralNetwork *nn, Matrix **activations,
                          int activation_count, Matrix *outputs) {

  int deltas_size = (activation_count - 1) * sizeof(Matrix *);

  Matrix **deltas = malloc(deltas_size);
  Matrix *error_derivative =
      matrix_substract(activations[nn->matrix_weight_count], outputs);

  for (int i = 0; i < nn->matrix_weight_count; i++) {

    Matrix *activation_derivative = matrix_element_operation(
        activations[nn->matrix_weight_count - i], sigmoid_derivative);
    Matrix *act_err_derivative =
        matrix_times(activation_derivative, error_derivative);
    Matrix *activations_l_minus_1_T =
        matrix_transpose(activations[nn->matrix_weight_count - i - 1]);
    Matrix *delta = matrix_dot(activations_l_minus_1_T, act_err_derivative);

    // matrix_free(error_derivative);
    error_derivative = matrix_transpose(delta);
    matrix_free(activation_derivative);
    matrix_free(activations_l_minus_1_T);
    deltas[nn->matrix_weight_count - 1 - i] = delta;
  }

  if (print == 1) {
    list_forEach(deltas, nn->matrix_weight_count, matrix_print);
  }

  return deltas;
}

void NN_udpate_weights(NeuralNetwork *nn, Matrix **activations,
                       Matrix **deltas) {

  for (int i = 0; i < nn->matrix_weight_count; i++) {

    Matrix *final_change = matrix_scale(deltas[i], -nn->learning_rate);
    if (print == 1) {
      printf("final changes to weights");
      matrix_print(final_change);
    }
    matrix_add_inplace(nn->list_weights[i], final_change);

    matrix_free(final_change);
  }
  if (print == 1) {
    print_weights(nn);
  }
}

Matrix *predict(NeuralNetwork *nn, Matrix *x) {

  if (x->cols != nn->list_weights[0]->rows) {
    insert_ones_column(x);
  }

  Matrix *activations = matrix_dup(x);

  for (int i = 0; i < nn->matrix_weight_count; i++) {
    Matrix *dot = matrix_dot(activations, nn->list_weights[i]);
    matrix_free(activations);
    activations = matrix_element_operation(dot, sigmoid);
    matrix_free(dot);
  }

  return activations;
}

void fit_partial(NeuralNetwork *nn, Matrix *input, Matrix *outputs) {

  int activation_count = nn->matrix_weight_count + 1;

  // fow prop
  Matrix **activations = NN_create_activations(nn, activation_count, input);

  // back prop
  Matrix **deltas =
      NN_create_deltas(nn, activations, activation_count, outputs);
  NN_udpate_weights(nn, activations, deltas);

  // debug
  if (print == 1) {
    Matrix *prediction = predict(nn, input);
    printf("%f %f = %f | Predicted: %f\n", input->data[0][0], input->data[0][1],
           outputs->data[0][0], prediction->data[0][0]);
  }

  list_free(activations, nn->matrix_weight_count + 1, matrix_free);
  list_free(deltas, nn->matrix_weight_count, matrix_free);
}

void fit(NeuralNetwork *nn, Matrix *inputs, Matrix *outputs, int epochs,
         int display_update) {

  insert_ones_column(inputs);

  for (int i = 0; i < epochs; i++) {
    suffle_data(inputs, outputs);
    for (int j = 0; j < inputs->rows; j++) {

      Matrix *input = matrix_get_rows(inputs, j, j + 1);
      Matrix *target = matrix_get_rows(outputs, j, j + 1);

      fit_partial(nn, input, target);

      matrix_free(input);
      matrix_free(target);
    }
    if (i % display_update == 0) {
      float loss = calculate_loss(nn, inputs, outputs);
      printf("Current loss: %g\n", loss);
    }
  }
}

char *description_create(int *layers, int len_layers) {

  char *desc = malloc(sizeof(char) * 2 * len_layers);
  desc[0] = '\0';
  char a[10] = " ";

  sprintf(a, "%d", layers[0]);
  strcat(desc, a);

  for (int i = 1; i < len_layers; i++) {
    sprintf(a, "%d", layers[i]);
    strcat(desc, "-");
    strcat(desc, a);
  }
  return desc;
}

NeuralNetwork *neural_network_create(int layers[], int len_layers,
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

  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  //  {count_matrix, NULL, weights, NULL, desc, learning_rate};
  nn->desc = desc;
  nn->list_weights = weights;
  nn->learning_rate = learning_rate;
  nn->matrix_weight_count = count_matrix;
  nn->loss_function = mse;

  if (print == 1) {
    print_weights(nn);
  }

  return nn;
}

void NN_free_weights(NeuralNetwork nn) {
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    matrix_free(nn.list_weights[i]);
  }
  free(nn.list_weights);
}

void NN_free(NeuralNetwork *nn) {
  free(nn->desc);
  NN_free_weights(*nn);
  free(nn);
  nn = NULL;
}

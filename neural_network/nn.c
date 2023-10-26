
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

int print = 0;

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
float sigmoid_derivative_single(float x) { return x * (1 - x); }

Matrix *mse_derivative(Matrix *prediction, Matrix *target) {
  return matrix_subtract(prediction, target);
}

Matrix *sigmoid_derivative(Matrix *A) {
  Matrix *final = matrix_element_operation(A, sigmoid_derivative_single);
  return final;
}

float euler(float a) { return (float)exp(a); }

Matrix *softmax(Matrix *z) {
  float denominator = 0;
  for (int i = 0; i < z->cols; i++) {
    denominator += exp(z->data[0][i]);
  }

  Matrix *numerator = matrix_transform(z, euler);
  Matrix *probability = matrix_scale(numerator, 1.0 / denominator);
  free(numerator);

  return probability;
}

float one(float a) { return 1 * a / a; }

Matrix *softmax_derivative(Matrix *A) {

  Matrix *probability = softmax(A);
  Matrix *ones = matrix_create(A->rows, A->cols);
  matrix_inplace_element_operation(ones, one); // fill with ones
  Matrix *temp = matrix_subtract(ones, probability);
  Matrix *final = matrix_times(probability, temp);

  matrix_free(ones);
  matrix_free(temp);

  return final;
}

float relu(float x) {
  if (x > 0)
    return x;
  else
    return 0;
}

float relu_derivative_single(float x) {
  if (x >= 0)
    return 1;
  else
    return 0;
}

Matrix *relu_derivative(Matrix *z) {
  Matrix *final = matrix_element_operation(z, relu_derivative_single);
  return final;
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

float ln(float a) { return log(a); }

// TODO Y not used
float cross_entropy(NeuralNetwork *nn, Matrix *X, Matrix *Y) {
  Matrix *prediction = predict(nn, X);
  Matrix *probability = softmax(prediction);
  Matrix *lns = matrix_transform(probability, ln);
  Matrix *errors = matrix_scale(lns, -1);

  matrix_free(probability);
  matrix_free(lns);
  matrix_free(prediction);

  float error = 0;

  for (int i = 0; i < prediction->cols; i++) {
    error += errors->data[0][i];
  }

  return error;
}

// TODO
Matrix *cross_entropy_derivative(Matrix *A) {
  // Matrix *sd = softmax_derivative(A);
  // Matrix *derivative = matrix_subtract(sd, A);

  // this two lines are for no warnings
  A = NULL;
  return A;
}

float mse(NeuralNetwork *nn, Matrix *X, Matrix *Y) {
  Matrix *prediction = predict(nn, X);
  float error = 0;
  for (int i = 0; i < prediction->rows; i++) {
    error += pow(prediction->data[i][0] - Y->data[i][0], 2);
  }

  error /= prediction->cols;

  matrix_free(prediction);
  return error;
}

float calculate_loss(NeuralNetwork *nn, Matrix *inputs, Matrix *targets) {

  float network_error = 0;
  for (int i = 0; i < inputs->rows; i++) {
    Matrix *input = matrix_get_rows(inputs, i, i + 1);
    Matrix *target = matrix_get_rows(targets, i, i + 1);
    network_error += nn->loss_function(nn, input, target);
    // network_error += mse(nn, input, target);
    matrix_free(input);
    matrix_free(target);
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
    Matrix *out = matrix_transform(net, nn->activators[i]);
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
  Matrix *error_derivative = nn->loss_function_derivative(activations[nn->matrix_weight_count], outputs);
  // matrix_subtract(activations[nn->matrix_weight_count], outputs);

  Matrix **deltas = malloc(deltas_size);

  Matrix *d = nn->activators_derivatives[nn->matrix_weight_count - 1](
      activations[nn->matrix_weight_count]);

  Matrix *delta = matrix_times(d, error_derivative);
  matrix_free(error_derivative);
  matrix_free(d);

  deltas[0] = delta;

  for (int i = 0; i < activation_count - 2; i++) {

    Matrix *transpose_weight =
        matrix_transpose(nn->list_weights[nn->matrix_weight_count - i - 1]);
    Matrix *delta = matrix_dot(deltas[i], transpose_weight);

    Matrix *d = nn->activators_derivatives[nn->matrix_weight_count - 1 - 1 - i](
        activations[nn->matrix_weight_count - 1 - i]);

    // Matrix *d = nn->activators_derivatives[nn->matrix_weight_count - 1 -
    // 1 - i](
    //     activations[activation_count - 2 - i]);
    // nn->activations[activation_count - 2 - i], relu_derivative);
    // delta = matrix_element_to_element_operation(delta, d, times);
    Matrix *new_delta = matrix_times(delta, d);
    matrix_free(delta);
    deltas[i + 1] = new_delta;

    matrix_free(d);
    matrix_free(transpose_weight);
  }

  // nn->deltas = deltas;
  if (print == 1) {
    list_forEach(deltas, nn->matrix_weight_count, matrix_print);
  }

  return deltas;
}

void NN_udpate_weights(NeuralNetwork *nn, Matrix **activations,
                       Matrix **deltas) {

  for (int i = 0; i < nn->matrix_weight_count; i++) {
    Matrix *activations_transpose = matrix_transpose(activations[i]);
    Matrix *dot = matrix_dot(activations_transpose,
                             deltas[nn->matrix_weight_count - 1 - i]);

    Matrix *final_change = matrix_scale(dot, -nn->learning_rate);
    if (print == 1) {
      printf("final changes to weights\n");
      matrix_print(final_change);
    }
    matrix_add_inplace(nn->list_weights[i], final_change);

    matrix_free(activations_transpose);
    matrix_free(final_change);
    matrix_free(dot);
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

void fit_partial(NeuralNetwork *nn, Matrix *input, Matrix *output) {

  int activation_count = nn->matrix_weight_count + 1;

  // fow prop
  Matrix **activations = NN_create_activations(nn, activation_count, input);

  // back prop
  Matrix **deltas = NN_create_deltas(nn, activations, activation_count, output);
  NN_udpate_weights(nn, activations, deltas);

  // debug
  if (print == 1) {
    Matrix *prediction = predict(nn, input);
    printf("%f %f = %f | Predicted: %f\n", input->data[0][0], input->data[0][1],
           output->data[0][0], prediction->data[0][0]);
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

  nn->activators = malloc(sizeof(ActivationFunction) * (len_layers - 1));
  // for (int i = 0; i < len_layers - 1; i++) {
  //   nn->activators[i] = sigmoid;
  // }

  nn->activators_derivatives =
      malloc(sizeof(ActivationDerivativeFunction) * (len_layers - 1));
  // for (int i = 0; i < len_layers - 1; i++) {
  //   nn->activators_derivatives[i] = sigmoid_derivative;
  // }

  if (print == 1) {
    print_weights(nn);
  }

  return nn;
}

void NN_set_loss_function(NeuralNetwork *nn, LossFunction function) {
  switch (function) {
  case MSE:
    nn->loss_function = mse;
    nn->loss_function_derivative = mse_derivative;
    break;
  case CROSS_ENTROPY:
    nn->loss_function = cross_entropy; // todo: create coss_entropy
    break;
  }
}

void NN_set_layer_activation(NeuralNetwork *nn, int layer,
                             Activations function) {

  if (layer == 0) {
    printf("Cant set activation function to the inputs\n");
    fflush(stdout);
    abort();
  }

  switch (function) {
  case RELU:
    nn->activators[layer - 1] = relu;
    nn->activators_derivatives[layer - 1] = relu_derivative;
    break;

  case SIGMOID:
    nn->activators[layer - 1] = sigmoid;
    nn->activators_derivatives[layer - 1] = sigmoid_derivative;
    break;

  case SOFTMAX:
    // todo
    //  nn->activators[layer] = softmax;
    //  nn->activators_derivatives[layer] = softmax_derivative;
    break;
  }
}

void NN_free_weights(NeuralNetwork nn) {
  for (int i = 0; i < nn.matrix_weight_count; i++) {
    matrix_free(nn.list_weights[i]);
  }
  free(nn.list_weights);
}

void NN_free(NeuralNetwork *nn) {
  free(nn->desc);
  free(nn->activators_derivatives);
  free(nn->activators);
  NN_free_weights(*nn);
  free(nn);
  nn = NULL;
}

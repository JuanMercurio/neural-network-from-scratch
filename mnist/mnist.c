#include "../matrix/matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mnist_pretty_print(Matrix *drawing) {

  Matrix *nice = matrix_create(28, 28);

  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      nice->data[i][j] = drawing->data[0][(28 * i) + j];
    }
  }

  matrix_print(nice);
  free(nice);
}

float *one_hot(int output_count, int value) {
  if (output_count < value || value < 0) {
    printf("el output es: %d y el valor es: %d\n", output_count, value);
    fflush(stdout);
    exit(1);
  }

  float *output = malloc(sizeof(float) * output_count);
  for (int i = 0; i < output_count; i++) {
    output[i] = 0;
  }

  output[value] = 1;

  return output;
}

Matrix *mnist_get_output_data() {
  FILE *file = fopen("mnist/mnist_train.csv", "r");

  int i = 0;
  char c = getc(file);
  while (c != EOF) {
    if (c == '\n')
      i++;
    c = getc(file);
  }

  fseek(file, 0, SEEK_SET);

  Matrix *outputs = malloc(sizeof(Matrix *));
  outputs->rows = i - 1;
  outputs->cols = 10;
  outputs->data = malloc(sizeof(float *) * i);

  for (int j = 0; j < i - 1; j++) {
    c = getc(file);
    float *output = one_hot(10, c - '0');
    outputs->data[j] = output;
    while (getc(file) != '\n')
      ;
  }

  fclose(file);
  return outputs;
}

Matrix *mnist_get_input_data() {
  FILE *file = fopen("mnist/mnist_train.csv", "r");

  if (file == NULL) {
    perror("Unable to open file!");
    abort();
  }

  int i = 0;

  char c = getc(file);
  while (c != EOF) {
    if (c == '\n')
      i++;
    c = getc(file);
  }

  fseek(file, 0, SEEK_SET);

  Matrix *inputs = malloc(sizeof(Matrix *));
  inputs->cols = 28 * 28;
  inputs->rows = i - 1;
  inputs->data = malloc(sizeof(float *) * i);

  if (inputs == NULL)
    abort();

  c = getc(file);
  for (int j = 0; j < i - 1; j++) {

    float *input = malloc(sizeof(float) * 28 * 28);

    c = getc(file);
    c = getc(file);
    for (int k = 0; k < 784; k++) {

      char value[100] = "";

      while (c != ',') {
        char s[2];
        sprintf(s, "%c", c);
        strcat(value, s);
        c = getc(file);
      }

      char *err[10];
      input[k] = strtol(value, err, 10);

      c = getc(file);
    }

    inputs->data[j] = input;
  }
  fclose(file);

  return inputs;
}

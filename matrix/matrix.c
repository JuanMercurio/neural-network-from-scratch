#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

float times(float x, float y) { return x * y; }
float add(float x, float y) { return x + y; }
float identity(float x) { return x; }
float substract(float a, float b) { return a - b; }
float zero(float z) { return 0 * z; }

Matrix *matrix_create(int row_count, int col_count) {
  Matrix *matrix = calloc(1, sizeof(Matrix));

  matrix->data = malloc(row_count * sizeof(float *));
  matrix->rows = row_count;
  matrix->cols = col_count;

  for (int i = 0; i < row_count; i++) {
    matrix->data[i] = malloc(sizeof(float) * col_count);
  }

  // matrix_inplace_element_operation(matrix, zero);

  return matrix;
}

void matrix_free(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    free(m->data[i]);
  }
  free(m->data);
  free(m);
  m = NULL;
}

Matrix *matrix_dup(Matrix *m) { return matrix_element_operation(m, identity); }

void matrix_add_inplace(Matrix *to_update, Matrix *updater) {
  matrix_inplace_element_to_element_operation(to_update, updater, add);
}

Matrix *matrix_subtract(Matrix *m1, Matrix *m2) {
  return matrix_element_to_element_operation(m1, m2, substract);
}

Matrix *matrix_times(Matrix *m1, Matrix *m2) {
  return matrix_element_to_element_operation(m1, m2, times);
}

void matrix_add_column(Matrix *m, int value) {
  for (int i = 0; i < m->rows; i++) {
    m->data[i] = realloc(m->data[i], (m->rows + 1) * sizeof(float *));
    m->data[i][m->cols] = value;
  }
  m->cols++;
}

void matrix_print(Matrix *m) {
  for (int j = 0; j < m->rows; j++) {
    for (int k = 0; k < m->cols; k++) {
      printf("%-3g ", m->data[j][k]);
    }
    printf("\n");
  }
}

Matrix *matrix_transform(Matrix *m, float (*transfomer)(float)) {

  Matrix *final = matrix_create(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = transfomer(m->data[i][j]);
    }
  }

  return final;
}

Matrix *matrix_scale(Matrix *m, float scalar) {

  Matrix *final = matrix_create(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {

    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = m->data[i][j] * scalar;
    }
  }

  return final;
}

Matrix *matrix_dot(Matrix *m1, Matrix *m2) {

  if (m1->cols != m2->rows) {
    printf("ERROR: matrix_dot");
    printf("The amount of columns in the first matrix do not equal the "
           "amount of rows in the second column");
    char error_location[100];
    sprintf(error_location, "File: %s | Line: %d", __FILE__, __LINE__);
    printf("%s\n", error_location);
    fflush(stdout);
    exit(1);
  }

  Matrix *final = matrix_create(m1->rows, m2->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      float value = 0;
      for (int k = 0; k < m2->rows; k++) {
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

Matrix *matrix_transpose(Matrix *m) {

  Matrix *transpose = matrix_create(m->cols, m->rows);

  for (int i = 0; i < transpose->rows; i++) {
    for (int j = 0; j < transpose->cols; j++) {
      transpose->data[i][j] = m->data[j][i];
    }
  }

  return transpose;
}

Matrix *matrix_element_operation(Matrix *m, float(operation(float))) {

  Matrix *final = matrix_create(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = operation(m->data[i][j]);
    }
  }

  return final;
}

Matrix *matrix_get_rows(Matrix *m, int begining, int end) {
  Matrix *final = matrix_create(end - begining, m->cols);
  for (int i = 0; i < end - begining; i++) {
    for (int j = 0; j < m->cols; j++) {
      final->data[i][j] = m->data[begining + i][j];
    }
  }
  return final;
}

Matrix *matrix_element_to_element_operation(Matrix *m1, Matrix *m2,
                                            float (*operation)(float, float)) {

  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    char error_location[100];
    char error[] =
        "matrix_operation error: The number of rows or columns differs";
    sprintf(error_location, "File: %s | Line: %d ", __FILE__, __LINE__);
    printf("%s\n", error);
    printf("%s\n", error_location);
    exit(1);
  }

  Matrix *final = matrix_create(m1->rows, m1->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      final->data[i][j] = operation(m1->data[i][j], m2->data[i][j]);
    }
  }

  return final;
}

void matrix_inplace_element_to_element_operation(
    Matrix *to_update, Matrix *updater, float(operation(float, float))) {

  if (to_update->rows != updater->rows || to_update->cols != updater->cols) {
    char error_location[100];
    char error[] =
        "matrix_operation error: The number of rows or columns differs";
    sprintf(error_location, "File: %s | Line: %d ", __FILE__, __LINE__);
    printf("%s\n", error);
    printf("%s\n", error_location);
    exit(1);
  }

  for (int i = 0; i < to_update->rows; i++)
    for (int j = 0; j < to_update->cols; j++) {
      to_update->data[i][j] =
          operation(to_update->data[i][j], updater->data[i][j]);
    }
}

void matrix_inplace_element_operation(Matrix *m, float(operation(float))) {
  for (int i = 0; i < m->rows; i++)
    for (int j = 0; j < m->cols; j++) {
      m->data[i][j] = operation(m->data[i][j]);
    }
}

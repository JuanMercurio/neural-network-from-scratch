#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  int cols;
  int rows;
  float **data;
} Matrix;

Matrix *matrix_create(int row_count, int col_count);
Matrix *matrix_subtract(Matrix *m1, Matrix *m2);
Matrix *matrix_dot(Matrix *m1, Matrix *m2);
Matrix *matrix_times(Matrix*m1, Matrix* m2);
Matrix *matrix_transform(Matrix *m, float (*transfomer)(float));
Matrix *matrix_scale(Matrix *m, float scalar);
Matrix *matrix_transpose(Matrix *m);
Matrix *matrix_dup(Matrix *m);
Matrix *matrix_get_rows(Matrix *m, int begining, int end);

void matrix_add_column(Matrix *m, int value);
void matrix_free(Matrix*);
void matrix_print(Matrix*);
void matrix_add_inplace(Matrix *to_update, Matrix *updater);


Matrix *matrix_element_to_element_operation(Matrix *m1, Matrix *m2,
                                            float (*operation)(float, float));
Matrix *matrix_element_operation(Matrix *m, float(operation(float)));
void matrix_inplace_element_to_element_operation(
    Matrix *to_update, Matrix *updater, float(operation(float, float)));
void matrix_inplace_element_operation(Matrix *m, float(operation(float)));

#endif // !MATRIX_H

#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            result->data[i][j] = rand_double(low, high);
        }
    }
}


void copy_matrix(matrix *result, matrix *mat) {
    #pragma omp parallel for
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[i][j] = mat->data[i][j];
        }
    }
}
/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    //Initialize mat space and set values to remove garbage
    matrix *temp = (struct matrix*)malloc(sizeof(struct matrix));
    if (!temp) {
        return -1;
    }
    //IS_1D
    temp->is_1d = 0;
    if (rows == 1 || cols == 1) {
        temp->is_1d = 1;
    }
    //PARENT
    temp->parent = NULL;
    //ROWS AND COLS
    temp->rows = rows;
    temp->cols = cols;
    //REF COUNT
    temp->ref_cnt = (int*)malloc(sizeof(int));
    if (!temp->ref_cnt) {
        return -1;
    }
    temp->ref_cnt[0] = 1;
    //DATA
    temp->data = (double**)malloc(rows*sizeof(double*));
    if (!temp->data) {
        return -1;
    }
    double *row_data = (double*)calloc(rows * cols, sizeof(double));
    if (!row_data) {
        return -1;
    }
    temp->row_major = NULL;
    temp->row_major = row_data;
    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        temp->data[i] = row_data + (i * cols);
    }
    //RESULTS
    (*mat) = temp;
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {

    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    if(rows + row_offset > from->rows || cols + col_offset > from->cols) {
        return -1;
    }
    matrix *temp = (struct matrix*)malloc(sizeof(struct matrix));
    if (!temp) {
        return -1;
    }
    //CLEAR GARBAGE VALS
    temp->data = (double**)malloc(rows*sizeof(double*));
    if (temp->data == NULL) {
        return -1;
    }
    temp->row_major = NULL;
    temp->rows = 0;
    temp->cols = 0;
    temp->is_1d = 0;
    if (rows == 1 || cols == 1){
        temp->is_1d = 1;
    }
    temp->ref_cnt = NULL;
    temp->parent = NULL;
    //PUT IN CORRECT VALS
    temp->rows = rows;
    temp->cols = cols;
    temp->row_major = from->row_major;
    if (from->parent == NULL) {
       temp->parent = from;
    } else {
        temp->parent = from->parent;
    }
    //Data pointers point to the same parent matrix
    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        temp->data[i] = from->data[i + row_offset] + col_offset; // Start pointer at beginning of array (row) and increment by column offset
    }
    //Increase parent ref count by 1 b/c the matrix was successfully filled
    from->ref_cnt[0] += 1;
    temp->ref_cnt = from->ref_cnt;
    //Return
    (*mat) = temp;
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (!mat){
        return;
    }
    mat->ref_cnt[0] -= 1;
    // if mat does not share data with any other matrices then free data
    if (mat->ref_cnt[0] == 0) {

        free(mat->row_major);

        free(mat->ref_cnt);
    }
    free(mat->data);
    free(mat);

}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    #pragma omp parallel for
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i][j] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
  int mat1_row = mat1->rows;
  int mat1_col = mat1->cols;
    if (!mat1 || !mat2 || !result) {
        return -1;
    }
    if (mat1_row != mat2->rows || mat1_col != mat2->cols || result->rows != mat1_row || result->cols != mat1_col) {
        return -1;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < mat1_row; i++) {
        for (size_t j = 0; j < mat1_col / 4; j++) {
            _mm256_storeu_pd(&result->data[i][j*4], _mm256_add_pd(_mm256_loadu_pd(&mat1->data[i][j*4]), _mm256_loadu_pd(&mat2->data[i][j*4])));
        }
        for (size_t k = ((mat1_col) / 4) * 4; k < mat1_col; k++) {
            result->data[i][k] = mat1->data[i][k] + mat2->data[i][k];
        }
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int mat1_row = mat1->rows;
    int mat1_col = mat1->cols;
    if (!mat1 || !mat2 || !result) {
        return -1;
    }
    if (mat1_row != mat2->rows || mat1_col != mat2->cols || result->rows != mat1_row || result->cols != mat1_col) {
        return -1;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < mat1_row; i++) {
        for (size_t j = 0; j < mat1_col; j++)
        {
            result->data[i][j] = (mat1->data[i][j] - mat2->data[i][j]);
        }
    }
    return -1;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int mat1_row = mat1->rows;
    int mat1_col = mat1->cols;
    int mat2_col = mat2->cols;
    int mat2_row = mat2->rows;
    double **result_data = result->data;
    double **mat1_data = mat1->data;
    double **mat2_data = mat2->data;
    if (!mat1 || !mat2 || !result) {
        return -1;
    }
    if (mat1_col != mat2_row || result->rows != mat1_row || result->cols != mat2_col) {
        return -1;
    }
    /*
    double *col_major_mat2 = calloc(mat2_col*mat2_row, sizeof(double));
    size_t blocksize = 16;
    #pragma omp parallel for
    for (size_t i = 0; i < mat2_row; i += blocksize) {
        for (size_t j = 0; j < mat2_col; j += blocksize) {
        // transpose the block beginning at [i,j]
            for (size_t k = i; k < i + blocksize && k < mat2_row; ++k) {
                int kOff = k * mat2_col;
                for (size_t l = j; l < j + blocksize && l < mat2_col; ++l) {
                    col_major_mat2[l*mat2_row + k] = mat2->row_major[kOff + l];
                }
            }
        }
    }
    */
    fill_matrix(result, 0);
    size_t i;
    size_t j;
    size_t k;
    #pragma omp parallel for
    for (i = 0; i < mat1_row; i++) {
        for (j = 0; j < mat1_col; j++) {
            for (k = 0; k < mat2_col; k++) {
                result_data[i][k] += mat1_data[i][j] * mat2_data[j][k];
            }
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int mat_row = mat->rows;
    int mat_col = mat->cols;
    if (pow < 0 || !mat || !result || mat_row != mat_col ||
    mat_row != result->rows || mat_col != result->cols) {
        return -1;
    }
    matrix *temp;
    allocate_matrix(&temp, mat_row, mat_col);
    matrix *mat_copy;
    allocate_matrix(&mat_copy, mat_row, mat_col);
    copy_matrix(mat_copy, mat);
    fill_matrix(result, 0);
    #pragma omp parallel for
    for (size_t i = 0; i < mat_row; i++){
        result->data[i][i] = 1;
    }
    while (pow > 0) {
        if (pow % 2 == 1) {
            mul_matrix(temp, result, mat_copy);
            //printf("temp with pow = %d: \n", pow);
            //print_mat(temp);
            copy_matrix(result, temp);
            //printf("result with pow = %d: \n", pow);
            //print_mat(result);
            pow -= 1;
        }  else {
            mul_matrix(temp, mat_copy, mat_copy);
            //printf("temp with pow = %d: \n", pow);
            //print_mat(temp);
            copy_matrix(mat_copy, temp);
            //printf("mat_copy with pow = %d: \n", pow);
            //print_mat(mat_copy);
            pow = pow / 2;
        }
    }
    deallocate_matrix(temp);
    deallocate_matrix(mat_copy);
    return 0;
}


void print_mat(matrix* mat) { //for debuggin
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%f, ", mat->data[i][j]);
        }
        printf("\n");
    }

}

/*
    matrix *temp;
    allocate_matrix(&temp, mat->rows, mat->cols);
    copy_matrix(temp, mat);
    if (pow == 1) {
        copy_matrix(result, mat);
    }
    while (pow > 1) {
        mul_matrix(result, temp, mat);
        copy_matrix(temp, result);
        pow -= 1;
    }
    deallocate_matrix(temp);
    return 0;
    *
    * /
}




/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if (!mat || !result) {
        return -1;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[i][j] =  -1*(mat->data[i][j]);
        }
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if (!mat || !result) {
        return -1;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            double temp = mat->data[i][j];
            if (temp < 0) {
                result->data[i][j] =  -temp;
            } else {
                result->data[i][j] =  temp;
            }

        }
    }
    return 0;
}

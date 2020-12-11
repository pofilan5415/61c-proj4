#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(self->mat->data[i][j]));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(self->mat->data[i][j]));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
       PyErr_SetString(PyExc_TypeError, "Args must be of matrix type.");
       return NULL;
    }
    Matrix61c* mat2 = (Matrix61c*)args;
    if (self->mat->rows != mat2->mat->rows || self->mat->cols != mat2->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match.");
        return NULL;
    }
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, self->mat->cols);
    matrix *temp;
    if (allocate_matrix(&temp, self->mat->rows, self->mat->cols) == 0) {
        add_matrix(temp, self->mat, mat2->mat);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
    
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
       PyErr_SetString(PyExc_TypeError, "Args must be of matrix type.");
       return NULL;
    }
    Matrix61c* mat2 = (Matrix61c*)args;
    if (self->mat->rows != mat2->mat->rows || self->mat->cols != mat2->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match.");
        return NULL;
    }
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, self->mat->cols);
    matrix *temp;
    if (allocate_matrix(&temp, self->mat->rows, self->mat->cols) == 0) {
        sub_matrix(temp, self->mat, mat2->mat);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
       PyErr_SetString(PyExc_TypeError, "Args must be of matrix type.");
       return NULL;
    }
    Matrix61c* mat2 = (Matrix61c*)args;
    if (mat2->mat->rows != self->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Dimensions do not match.");
        return NULL;
    }
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, mat2->mat->cols);
    matrix *temp;
    if (allocate_matrix(&temp, self->mat->rows, mat2->mat->cols) == 0) {
        mul_matrix(temp, self->mat, mat2->mat);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
    
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, self->mat->cols);
    matrix *temp;
    if (allocate_matrix(&temp, self->mat->rows, self->mat->cols) == 0) {
        neg_matrix(temp, self->mat);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, self->mat->cols);
    matrix *temp;
    if(allocate_matrix(&temp, self->mat->rows, self->mat->cols) == 0) {
        abs_matrix(temp, self->mat);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    if (!PyLong_Check(pow)) {
       PyErr_SetString(PyExc_TypeError, "Pow must be of int type.");
       return NULL;
    }
    int pow_as_int = PyLong_AsLong(pow);
    if (self->mat->rows != self->mat->cols || pow_as_int < 0) {
        PyErr_SetString(PyExc_ValueError, "Pow must be greater than 0.");
       return NULL;
    }
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->shape = get_shape(self->mat->rows, self->mat->cols);
    matrix *temp;
    if(allocate_matrix(&temp, self->mat->rows, self->mat->cols) == 0) {
        pow_matrix(temp, self->mat, pow_as_int);
        result->mat = temp;
        return result;
    }
    PyErr_SetString(PyExc_RuntimeError, "Allocation Unsuccessful.");
    return NULL;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    .nb_add = Matrix61c_add,
    .nb_subtract = Matrix61c_sub,
    .nb_multiply = Matrix61c_multiply,
    .nb_negative = Matrix61c_neg,
    .nb_absolute = Matrix61c_abs,
    .nb_power = Matrix61c_pow
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    int row;
    int col;
    double val;
    if (!PyTuple_Check(args) || PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "Incorrect Args.");
        return NULL;
    }
    if (!PyLong_Check(PyTuple_GetItem(args, 0)) || !PyLong_Check(PyTuple_GetItem(args, 1))) {
        PyErr_SetString(PyExc_TypeError, "Args must be ints");
        return NULL; 
    }
    if (!PyLong_Check(PyTuple_GetItem(args, 2)) && !PyFloat_Check(PyTuple_GetItem(args, 2))) {
        PyErr_SetString(PyExc_TypeError, "Val must be float or int");
        return NULL;
    }
    row = PyLong_AsLong(PyTuple_GetItem(args, 0));
    col = PyLong_AsLong(PyTuple_GetItem(args, 1));
    val = PyFloat_AsDouble(PyTuple_GetItem(args, 2));
    if (row >= self->mat->rows || col >= self->mat->cols || row < 0 || col < 0) {
        PyErr_SetString(PyExc_IndexError, "Trying to index out of range.");
        return NULL;
    }
    self->mat->data[row][col] = val;
    return Py_None;
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    int row;
    int col;
    if (!PyTuple_Check(args) || PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect Args");
        return NULL;
    }
    if (!PyLong_Check(PyTuple_GetItem(args, 0)) || !PyLong_Check(PyTuple_GetItem(args, 1))) {
        PyErr_SetString(PyExc_TypeError, "Args must be ints");
        return NULL; 
    }
    row = PyLong_AsLong(PyTuple_GetItem(args, 0));
    col = PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (row > self->mat->rows || col > self->mat->cols || row < 0 || col < 0) {
        PyErr_SetString(PyExc_IndexError, "Trying to index out of range.");
        return NULL;
    }
    return PyFloat_FromDouble(get(self->mat, row, col));
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    {"get", (PyCFunction)Matrix61c_get_value, METH_VARARGS, "Returns the value at that location in the matrix."}, 
    {"set", (PyCFunction)Matrix61c_set_value, METH_VARARGS, "Sets the value at that location in the matrix."},
    {NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    if (self->mat->is_1d) {
        //printf("is 1_d\n");
        Py_ssize_t length;
        int is_vert;
        if (self->mat->rows == 1) {
            length = self->mat->cols;
            is_vert = 0;
            //printf("is not vert\n");
        } else {
            length = self->mat->rows;
            is_vert = 1;
            //printf("is vert\n");
        }

        if (PySlice_Check(key)) {
            //printf("key is a slice\n");
            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slicelength;

            PySlice_GetIndicesEx(key, length, &start, &stop, &step, &slicelength);
            if (step != 1 || slicelength < 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }
            //printf("before creating new matrix\n");
            Matrix61c *new_1d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
            //printf("after creating new matrix\n");
            matrix *new_mat_1d;
            int rows = 1;
            int cols = 1;
            int row_offset = 0;
            int col_offset = 0;
            if (is_vert) {
                row_offset = start;
                rows = slicelength;
            } else {
                col_offset = start;
                cols = slicelength;
            }
            //printf("length: %d, start: %d, stop: %d, slicelength: %d \n", length, start, stop, slicelength);
            //printf("before calling all_mat_ref\n");
            allocate_matrix_ref(&new_mat_1d, self->mat, row_offset, col_offset, rows, cols);
            //printf("after calling all_mat_ref\n");
            new_1d->mat = new_mat_1d;
            new_1d->shape = get_shape(rows, cols);
            return new_1d;
        } else if (PyLong_Check(key)) {
            int key_as_int = PyLong_AsLong(key);
            //printf("key: %d \n", key_as_int);
            if ((key_as_int >= self->mat->cols && key_as_int >= self->mat->rows) || key_as_int < 0) {
                //printf("rip \n");
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                return NULL;
            }
            //printf("program continued \n");

            double val;
            if (is_vert) {
                val = self->mat->data[key_as_int][0];
            } else {
                val = self->mat->data[0][key_as_int];
            }
            return PyFloat_FromDouble(val);
        } else {
            PyErr_SetString(PyExc_TypeError, "key must be slice or integer for 1D array");
            return NULL;
        }
    } else {
        if (PySlice_Check(key)) {
            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slicelength;
            PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
            if (step != 1 || slicelength < 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }
            Matrix61c *new_2d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
            matrix *new_mat_2d;
            allocate_matrix_ref(&new_mat_2d, self->mat, start, 0, slicelength, self->mat->cols);                
            new_2d->mat = new_mat_2d;
            new_2d->shape = get_shape(slicelength, self->mat->cols);
            return new_2d;
        } else if (PyTuple_Check(key)) {
            //printf("0\n");
            //printf("int/slice: %d \n", PyLong_Check(PyTuple_GetItem(key, 0)) && PySlice_Check(PyTuple_GetItem(key, 1)));
            //printf("int/int: %d \n", PyLong_Check(PyTuple_GetItem(key, 0)) && PyLong_Check(PyTuple_GetItem(key, 1)));
            //printf("slice/long: %d \n", PySlice_Check(PyTuple_GetItem(key, 0)) && PyLong_Check(PyTuple_GetItem(key, 1)));
            //printf("slice/slice: %d \n", PySlice_Check(PyTuple_GetItem(key, 0)) && PySlice_Check(PyTuple_GetItem(key, 1)));
            if (PyLong_Check(PyTuple_GetItem(key, 0)) && PyLong_Check(PyTuple_GetItem(key, 1))) {
                int key_as_int1 = PyLong_AsLong(PyTuple_GetItem(key, 0));
                int key_as_int2 = PyLong_AsLong(PyTuple_GetItem(key, 1));

                if (key_as_int1 >= self->mat->rows || key_as_int2 >= self->mat->cols || key_as_int1 < 0 || key_as_int2 < 0) {
                    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                    return NULL;
                }
                double val = get(self->mat, key_as_int1, key_as_int2);
                return PyFloat_FromDouble(val);
            } else if (PyLong_Check(PyTuple_GetItem(key, 0)) && PySlice_Check(PyTuple_GetItem(key, 1))) {
                //printf("1\n");
                int key_as_int = PyLong_AsLong(PyTuple_GetItem(key, 0));
                //printf("2\n");
                if (key_as_int >= self->mat->rows || key_as_int < 0) {
                    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                    return NULL;
                } 
                Py_ssize_t start;
                Py_ssize_t stop;
                Py_ssize_t step;
                Py_ssize_t slicelength;
                //printf("3\n");
                PySlice_GetIndicesEx(PyTuple_GetItem(key, 1), self->mat->cols, &start, &stop, &step, &slicelength);
                //printf("4\n");
                if (step != 1 || slicelength < 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }
                //printf("5\n");
                Matrix61c *new_2d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                matrix *new_mat_2d;
                //printf("6\n");
                allocate_matrix_ref(&new_mat_2d, self->mat, key_as_int, start, 1, slicelength);
                new_2d->mat = new_mat_2d;
                new_2d->shape = get_shape(1, slicelength);
                //printf("7\n");
                return new_2d;
            } else if (PySlice_Check(PyTuple_GetItem(key, 0)) && PyLong_Check(PyTuple_GetItem(key, 1))) {
                int key_as_int = PyLong_AsLong(PyTuple_GetItem(key, 1));
                if (key_as_int >= self->mat->cols || key_as_int < 0) {
                    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                    return NULL;
                } 
                Py_ssize_t start;
                Py_ssize_t stop;
                Py_ssize_t step;
                Py_ssize_t slicelength;
                PySlice_GetIndicesEx(PyTuple_GetItem(key, 0), self->mat->rows, &start, &stop, &step, &slicelength);
                if (step != 1 || slicelength < 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }
                Matrix61c *new_2d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                matrix *new_mat_2d;
                allocate_matrix_ref(&new_mat_2d, self->mat, start, key_as_int, slicelength, 1);
                new_2d->mat = new_mat_2d;
                new_2d->shape = get_shape(slicelength, 1);
                return new_2d;
            } else {
                Py_ssize_t start1;
                Py_ssize_t stop1;
                Py_ssize_t step1;
                Py_ssize_t slicelength1;
                PySlice_GetIndicesEx(PyTuple_GetItem(key, 0), self->mat->rows, &start1, &stop1, &step1, &slicelength1);
                if (step1 != 1 || slicelength1 < 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }
                Py_ssize_t start2;
                Py_ssize_t stop2;
                Py_ssize_t step2;
                Py_ssize_t slicelength2;
                PySlice_GetIndicesEx(PyTuple_GetItem(key, 1), self->mat->cols, &start2, &stop2, &step2, &slicelength2);
                if (step2 != 1 || slicelength2 < 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }
                Matrix61c *new_2d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                matrix *new_mat_2d;
                allocate_matrix_ref(&new_mat_2d, self->mat, start1, start2, slicelength1, slicelength2);
                new_2d->mat = new_mat_2d;
                new_2d->shape = get_shape(slicelength1, slicelength2);
                return new_2d;
            }

        } else if (PyLong_Check(key)) {
            int key_as_int = PyLong_AsLong(key);

            if (key_as_int >= self->mat->rows || key_as_int < 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                return NULL;
            }
            Matrix61c *new_2d = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
            matrix *new_mat_2d;
            allocate_matrix_ref(&new_mat_2d, self->mat, key_as_int, 0, 1, self->mat->cols);
            new_2d->mat = new_mat_2d;
            new_2d->shape = get_shape(1, self->mat->cols);
            return new_2d;

        } else {
            PyErr_SetString(PyExc_TypeError, "key must be tuple, slice, or integer for 2D array");
            return NULL;
        }
    }
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    PyObject *slice = Matrix61c_subscript(self, key);
    if (self->mat->is_1d) {
        if (PyFloat_Check(slice)) {
            if (!PyLong_Check(v) && !PyFloat_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "val must be an int or double");
                return -1;
            }
            int key_as_int = PyLong_AsLong(key);
            double v_as_int = PyFloat_AsDouble(v);
            if (self->mat->rows == 1) {
                self->mat->data[0][key_as_int] = v_as_int;
            } else {
                self->mat->data[key_as_int][0] = v_as_int;
            }
        } else if (PyObject_TypeCheck(slice, &Matrix61cType)) {
            Matrix61c* slice_mat = (Matrix61c*)slice;
            if (!PyList_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "val must be a list");
                return -1;
            } 
            if (slice_mat->mat->rows == 1) {
                if (slice_mat->mat->cols != PyList_Size(v)) {
                    PyErr_SetString(PyExc_ValueError, "val has the wrong length for the selected slice");
                    return -1;
                }
                for (size_t i = 0; i < self->mat->cols; i++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, i));
                    slice_mat->mat->data[0][i] = val;
                }
            } else {
                if (slice_mat->mat->rows != PyList_Size(v)) {
                    PyErr_SetString(PyExc_ValueError, "val has the wrong length for the selected slice");
                    return -1;
                }
                for (size_t i = 0; i < self->mat->rows; i++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, i));
                    slice_mat->mat->data[i][0] = val;
                }
            }
        }
    } else {
        if (PyFloat_Check(slice)) {
            if (!PyLong_Check(v) && !PyFloat_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "val must be an int or double");
                return -1;
            }
            int key_as_int1 = PyLong_AsLong(PyTuple_GetItem(key, 0));
            int key_as_int2 = PyLong_AsLong(PyTuple_GetItem(key, 1));
            self->mat->data[key_as_int1][key_as_int2] = PyFloat_AsDouble(v);
        } else if (PyObject_TypeCheck(slice, &Matrix61cType)){
            Matrix61c* slice_mat = (Matrix61c*)slice;
            if (!PyList_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "val must be a list");
                return -1;
            } 
            if (slice_mat->mat->is_1d) {
                if (slice_mat->mat->rows == 1) {
                    if (slice_mat->mat->cols != PyList_Size(v)) {
                        PyErr_SetString(PyExc_ValueError, "val has the wrong length for the selected slice");
                        return -1;
                    }
                    for (size_t i = 0; i < self->mat->cols; i++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(v, i));
                        slice_mat->mat->data[0][i] = val;
                    }
                } else {
                    if (slice_mat->mat->rows != PyList_Size(v)) {
                        PyErr_SetString(PyExc_ValueError, "val has the wrong length for the selected slice");
                        return -1;
                    }
                    for (size_t i = 0; i < self->mat->rows; i++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(v, i));
                        slice_mat->mat->data[i][0] = val;
                    }
                }
            } else {
                if (!PyList_Check(PyList_GetItem(v, 0))) {
                    PyErr_SetString(PyExc_ValueError, "val must be a 2d list");
                    return -1;
                } 
                if (slice_mat->mat->rows != PyList_Size(v)) {
                    PyErr_SetString(PyExc_ValueError, "val has too many rows for the selected slice");
                    return -1;
                }
                if (slice_mat->mat->cols != PyList_Size(PyList_GetItem(v, 0))) {
                    PyErr_SetString(PyExc_ValueError, "val has too many cols for the selected slice");
                    return -1;
                }
            
                for (size_t i = 0; i < slice_mat->mat->rows; i++) {
                    for (size_t j = 0; j < slice_mat->mat->cols; j++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(v, i), j));
                        slice_mat->mat->data[i][j] = val;
                    }       
                }  
            }
        }
    } 
    return 0;
}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}
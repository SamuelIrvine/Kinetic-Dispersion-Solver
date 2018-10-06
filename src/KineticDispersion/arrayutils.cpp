//
// Created by h on 20/07/18.
//

#include "arrayutils.h"

arr1d arrayutils::extract1d(const p::api::const_object_attribute &obj) {
    np::ndarray arr = p::extract<np::ndarray>(obj);
    if (arr.get_nd() != 1)
        throw std::runtime_error("Range dimensionality must equal 1. ");
    Py_intptr_t const *shape = arr.get_shape();
    Py_intptr_t const *stride = arr.get_strides();
    size_t n = (size_t) shape[0];

    arr1d val(n);

    for (size_t i = 0; i < n; i++) {
        val[i] = *reinterpret_cast<double const *>(arr.get_data() + i * stride[0]);
    }
    return val;
}

arr2d arrayutils::extract2d(const p::api::const_object_attribute &obj, const long nx, const long ny) {
    np::ndarray arr = p::extract<np::ndarray>(obj);
    if (arr.get_nd() != 2)
        throw std::runtime_error("Distribution dimensionality must equal 2. ");
    Py_intptr_t const *shape = arr.get_shape();
    Py_intptr_t const *stride = arr.get_strides();
    if (shape[0] != nx || shape[1] != ny)
        throw std::runtime_error("Distribution dimensions do not match. ");
    arr2d val((size_t) nx, arr1d((size_t) ny));
    for (long i = 0; i < nx; i++) {
        for (long j = 0; j < ny; j++) {
            val[i][j] = *reinterpret_cast<double const *>(arr.get_data() + i * stride[0] + j * stride[1]);
        }
    }
    return val;
}

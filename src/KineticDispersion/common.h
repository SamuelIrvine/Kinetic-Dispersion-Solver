//
// Created by h on 20/07/18.
//

#ifndef KINETIC_DISPERSION_SOLVER_COMMON_H
#define KINETIC_DISPERSION_SOLVER_COMMON_H

#include <iostream>
#include <boost/math/complex.hpp>
#include <vector>

namespace bm = boost::math;

using namespace std;

typedef complex<double> cdouble;
typedef vector<double> arr1d;
typedef vector<vector<double>> arr2d;

const double e0 = 8.85E-12; //TODO: increase precision
const double cl = 3E8;
const cdouble I{0.0, 1.0};
//const double mu0 = cl*cl*e0;

template<size_t N, typename T, typename T2>
const inline array<T, N> &operator+=(array<T, N> &a, const T2 &b) {
    for (size_t i = 0; i < N; i++) {
        a[i] += b[i];
    }
    return a;
};

template<size_t N, typename T, typename T2>
const inline array<T, N> &operator-=(array<T, N> &a, const T2 &b) {
    for (size_t i = 0; i < N; i++) {
        a[i] -= b[i];
    }
    return a;
};

template<size_t N, typename T, typename T2>
const inline array<T, N> &operator*=(array<T, N> &a, const T2 &b) {
    for (size_t i = 0; i < N; i++) {
        a[i] *= b;
    }
    return a;
};
#endif //KINETIC_DISPERSION_SOLVER_COMMON_H



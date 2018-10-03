//
// Created by h on 20/07/18.
//

#ifndef KINETIC_DISPERSION_SOLVER_ARRAYUTILS_H
#define KINETIC_DISPERSION_SOLVER_ARRAYUTILS_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "common.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

class arrayutils {

public:

    static arr1d extract1d(const p::api::const_object_attribute &obj);

    static arr2d extract2d(const p::api::const_object_attribute &obj, const long nx, const long ny);
};


#endif //KINETIC_DISPERSION_SOLVER_ARRAYUTILS_H

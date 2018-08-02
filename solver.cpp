#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>
#include "species.h"


using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;
namespace bm = boost::math;


template<typename T>
static T det(const array<array<T, 3>, 3> &p) {
    T sum{};
    for (size_t i = 0; i < 3; i++) {
        sum += p[1][(i + 1) % 3] * p[2][(i + 2) % 3] * p[0][i];
        sum -= p[1][(i + 2) % 3] * p[2][(i + 1) % 3] * p[0][i];
    }
    return sum;
};


class Solver {

    double kperp, kpara;

    vector<Species> species;

public:

    Solver(const p::list &list, const double B) {
        Py_Initialize();
        np::initialize();
        for (int i = 0; i < len(list); i++) {
            species.push_back(Species(list[i], B));
        }
    }

    void push_kperp(const double kperp) {
        this->kperp = kperp;
        for (Species &spec: species) {
            spec.push_kperp(kperp);
        }
    }

    void push_kpara(const double kpara) {
        this->kpara = kpara;
    }

    array<array<cdouble, 3>, 3> evaluateM(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> M{};
        for (Species &spec: species) {
            M += spec.push_omega(kpara, kperp, wr, wi);
        }
        double kperp2 = kperp*kperp;
        double kperp3 = kperp2*kperp;
        double kperp4 = kperp2*kperp2;
        double kpara2 = kpara*kpara;
        double kpara4 = kpara2*kpara2;
        double kpara5 = kpara4*kpara;
        double kpara6 = kpara4*kpara2;

        M[0][0] += kperp2*kpara4;
        M[1][1] += kperp2*kpara4;
        M[2][2] += kperp2*kpara4;

        cdouble c2_w2 = pow(cl / cdouble{wr, wi}, 2);

        M[0][0] -= c2_w2*kperp2*kpara6;
        M[0][2] += c2_w2*kperp3*kpara5;
        M[1][1] -= c2_w2*kperp2*kpara6;
        M[1][1] -= c2_w2*kperp4*kpara4;
        M[2][0] += c2_w2*kperp3*kpara5;
        M[2][2] -= c2_w2*kperp4*kpara4;

        M *= (1.0/kpara4*kperp2);



        return M;
    };

    np::ndarray convertM(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> M = evaluateM(wr, wi);

        p::tuple shape = p::make_tuple(3, 3);
        np::ndarray npM = np::zeros(shape, np::dtype::get_builtin<cdouble>());

        for (size_t i=0;i<3;i++){
            for (size_t j=0;j<3;j++){
                npM[i][j] = M[i][j];
            }
        }

        return npM;
    }

    cdouble evaluateDetM(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> M = evaluateM(wr, wi);
        cdouble detM = det(M);
        double wa = sqrt(wr*wr + wi*wi);
        return detM * pow(wa, 4) / pow((kpara * kpara + kperp * kperp) * cl * cl, 2);
    }

    cdouble evaluateDetMLongitudinal(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> M = evaluateM(wr, wi);
        array<cdouble, 3> ML{};
        ML[0] = kperp*M[0][0] + kpara*M[0][2];
        ML[2] = kperp*M[2][0] + kpara*M[2][2];

        cdouble detML = (ML[0]*kperp + ML[2]*kpara)/(kperp*kperp + kpara*kpara);
        return detML;
    }

    np::ndarray marginalize(const np::ndarray &arr) {
        int nd = arr.get_nd();
        Py_intptr_t const *shape = arr.get_shape();
        Py_intptr_t const *stride = arr.get_strides();
        np::ndarray result = np::zeros(nd, shape, arr.get_dtype());
        if (nd == 1) {
            for (size_t i = 0; i < (size_t) shape[0]; i++) {
                cdouble w = *reinterpret_cast<cdouble const *>(arr.get_data() + i * stride[0]);
                cdouble res = evaluateDetM(w.real(), w.imag());
                *reinterpret_cast<cdouble *>(result.get_data() + i * stride[0]) = res;
            }
        } else if (nd == 2) {
            for (size_t i = 0; i < (size_t) shape[0]; i++) {
                for (size_t j = 0; j < (size_t) shape[1]; j++) {
                    cdouble w = *reinterpret_cast<cdouble const *>(arr.get_data() + i * stride[0] + j * stride[1]);
                    cdouble res = evaluateDetM(w.real(), w.imag());
                    *reinterpret_cast<cdouble *>(result.get_data() + i * stride[0] + j * stride[1]) = res;
                }
            }
        } else
            throw std::runtime_error("Unsupported marginalization dimensionality. ");
        return result;
    }

};


BOOST_PYTHON_MODULE (libSolver) {
    p::class_<Solver>("Solver", p::init<const boost::python::list, double>())
            .def("push_kperp", &Solver::push_kperp)
            .def("push_kpara", &Solver::push_kpara)
            .def("evaluateDetM", &Solver::evaluateDetM)
            .def("evaluateDetMLongitudinal", &Solver::evaluateDetMLongitudinal)
            .def("evaluateM", &Solver::convertM)
            .def("marginalize", &Solver::marginalize);
}




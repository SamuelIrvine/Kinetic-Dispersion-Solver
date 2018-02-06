
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/multi_array.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>
#include <iostream>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;
namespace bm = boost::math;

typedef complex<double> cdouble;
typedef vector<double> arr1d;
typedef vector<vector<double>> arr2d;

const double e0 = 8.85E-12; //TODO: increase precision
const double cl = 3E8;
const cdouble I{0.0, 1.0};

static arr1d extract1d(const p::api::const_object_attribute &obj) {
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

static arr2d extract2d(const p::api::const_object_attribute &obj, const long nx, const long ny) {
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

template<size_t M1, size_t M2, size_t N1, size_t N2, typename T>
static array<array<T, N1 + N2 - 1>, M1 + M2 - 1>
polyMul(const array<array<T, N1>, M1> &a1, const array<array<T, N2>, M2> &a2) {
    array<array<T, N1 + N2 - 1>, M1 + M2 - 1> p{};
    for (size_t i = 0; i < M1; i++) {
        for (size_t j = 0; j < N1; j++) {
            for (size_t k = 0; k < M2; k++) {
                for (size_t l = 0; l < N2; l++) {
                    p[i + k][j + l] += a1[i][j] * a2[k][l];
                }
            }
        }
    }
    return p;
};

template<size_t M, size_t N, typename T>
static T polyEval(const array<array<T, N>, M> &p, const double &x, const double &y) {
    T sum{};
    double py{1.0};
    for (size_t i = 0; i < M; i++) {
        double px{1.0};
        for (size_t j = 0; j < N; j++) {
            sum += px * py * p[i][j];
            px *= x;
        }
        py *= y;
    }
    return sum;
};



class TaylorSum {

    struct Element{
        arr2d sum1, sum2, sum3, sum4, sum5;

        Element(){};

        Element(size_t n_s, size_t n_a){
            sum1 = arr2d(n_s, arr1d(n_a, 0.0));
            sum2 = arr2d(n_s, arr1d(n_a, 0.0));
            sum3 = arr2d(n_s, arr1d(n_a, 0.0));
            sum4 = arr2d(n_s, arr1d(n_a, 0.0));
            sum5 = arr2d(n_s, arr1d(n_a, 0.0));
        }
    };

public:
    arr2d a0;
    size_t i, j, ni, ai;
    double a1, b, c, d;
    cdouble c1, c2, c3, c4, c5;

    Element sX00a, sX01a, sX02a, sX11a, sX12a, sX22a;
    Element sX00b, sX01b, sX02b, sX11b, sX12b, sX22b;

    vector<vector<vector<pair<size_t, size_t>>>> mapping;

    TaylorSum(){
    }

    TaylorSum(size_t n_a, vector<int> &ns) {
        size_t n_s = ns.size();
        a0 = arr2d(n_s, arr1d(n_a, 0.0));
        sX00a = Element(n_s, n_a);
        sX01a = Element(n_s, n_a);
        sX02a = Element(n_s, n_a);
        sX11a = Element(n_s, n_a);
        sX12a = Element(n_s, n_a);
        sX22a = Element(n_s, n_a);
        sX00b = Element(n_s, n_a);
        sX01b = Element(n_s, n_a);
        sX02b = Element(n_s, n_a);
        sX11b = Element(n_s, n_a);
        sX12b = Element(n_s, n_a);
        sX22b = Element(n_s, n_a);
        mapping = vector<vector<vector<pair<size_t, size_t>>>>(n_s);
        for (size_t k=0;k<ns.size();k++){
            mapping[k] = vector<vector<pair<size_t, size_t>>>(n_a);
        }
    }

    inline void setA(const size_t ni, const double a_min, const double da){
        for (size_t ai=0;ai<a0[ni].size();ai++){
            a0[ni][ai] = a_min + ai*da;
        }
    }

    inline void pushDenominator(const size_t ni, const size_t i, const size_t j, const size_t ai,
                                const double a1, const double b, const double c, const double d){
        this->ni = ni;
        this->i = i;
        this->j = j;
        this->ai = ai;
        this->a1 = a1;
        this->b = b;
        this->c = c;
        this->d = d;
        mapping[ni][ai].push_back(pair<size_t, size_t>(i, j));
    }

    inline void accumulate(Element &X, const arr2d &p, const arr1d &q, const double r){
        double z0 = p[i*2][j*2]*q[i*2]*r;
        double z1 = p[i*2][j*2+1]*q[i*2]*r;
        double z2 = p[i*2][j*2+2]*q[i*2]*r;
        double z3 = p[i*2+1][j*2]*q[i*2+1]*r;
        double z4 = p[i*2+1][j*2+1]*q[i*2+1]*r;
        double z5 = p[i*2+1][j*2+2]*q[i*2+1]*r;
        double z6 = p[i*2+2][j*2]*q[i*2+2]*r;
        double z7 = p[i*2+2][j*2+1]*q[i*2+2]*r;
        double z8 = p[i*2+2][j*2+2]*q[i*2+2]*r;

        double Z0 = (z0 + 4 * z1 + z2 + 4 * z3 + 16 * z4 + 4 * z5 + z6 + 4 * z7 + z8);
        double Z1 = (2 * z1 + z2 + 8 * z4 + 4 * z5 + 2 * z7 + z8);
        double Z2 = (2 * z3 + 8 * z4 + 2 * z5 + z6 + 4 * z7 + z8);
        double Z3 = (4 * z4 + 2 * z5 + 2 * z7 + z8);
        double Z4 = (z0 + 4 * z1 + z2 - 12 * z3 - 48 * z4 - 12 * z5 - 9 * z6 - 36 * z7 - 9 * z8);
        double Z5 = (z0 - 12 * z1 - 9 * z2 + 4 * z3 - 48 * z4 - 36 * z5 + z6 - 12 * z7 - 9 * z8);
        double Z6 = (2 * z1 + z2 - 24 * z4 + 12 * z5 + 18 * z7 + 9 * z8);
        double Z7 = (2 * z3 - 24 * z4 - 18 * z5 + z6 - 12 * z7 - 9 * z8);
        double Z8 = (z0 - 3 *(4 *z1 + 3 * z2 + 4 * z3 - 48 * z4 - 36 * z5 + 3 * z6 - 36 * z7 - 27 * z8));
        double Z9 = (2 * z3 - 16 * z4 - 16 * z5 + z6 - 8 * z7 - 8 * z8);
        double Z10 = (z0 - 8 * z1 - 8 * z2 + 4 * z3 - 32 * z4 - 32 * z5 + z6 - 8 * z7 - 8 * z8);
        double Z11 = (z0 - 8 * z1 - 8 * z2 - 12 * z3 + 96 * z4 + 96 * z5 - 9 * z6 + 72 * z7 + 72 * z8);
        double Z12 = (z0 + 4 * z1 + z2 - 8 * z3 - 32 * z4 - 8 * z5 - 8 * z6 - 32 * z7 - 8 * z8);
        double Z13 = (2 * z1 + z2 - 8 * (2 * z4 + z5 + 2 * z7 + z8));
        double Z14 = (z0 - 12 * z1 - 9 * z2 - 8 * z3 + 96 * z4 + 72 * z5 - 8 * z6 + 96 * z7 + 72 * z8);
        double Z15 = (3 * z0 - 20 * z1 - 25 * z2 + 12 * z3 - 80 * z4 - 100 * z5 + 3 * z6 - 20 * z7 - 25 * z8);
        double Z16 = (6 * z3 - 40 * z4 - 50 * z5 + 3 * z6 - 20 * z7 - 25 * z8);
        double Z17 = (3 * z0 - 20 * z1 - 25 * z2 - 36 * z3 + 240 * z4 + 300 * z5 - 27 * z6 + 180 * z7 + 225 * z8);
        double Z18 = (z0 - 8 * (z1 + z2 + z3 - 8 * z4 - 8 * z5 + z6 - 8 * z7 - 8 * z8));
        double Z19 = (3 * z0 + 12 * z1 + 3 * z2 - 20 * z3 - 80 * z4 - 20 * z5 - 25 * z6 - 100 * z7 - 25 * z8);
        double Z20 = (6 * z1 + 3 * z2 - 5 * (8 * z4 + 4 * z5 + 10 * z7 + 5 * z8));
        double Z21 = (3 * z0 - 36 * z1 - 27 * z2 - 20 * z3 + 240 * z4 + 180 * z5 - 25 * z6 + 300 * z7 + 225 * z8);

        X.sum1[ni][ai]+=Z0*(1.0/36.0);
        X.sum2[ni][ai]+=-(1.0/36.0) * (a1 * Z0 + b * Z1 + c * Z2 + d * Z3);
        X.sum3[ni][ai]+=(1.0/3600.0) * (100 * a1*a1 * Z0 + 200 * a1 * (b * Z1 + c * Z2 + d * Z3) -
                                        10 * (c*c * Z4 + b * (b * Z5 - 20 * c * Z3 + 2 * d * Z7) + 2 * c * d * Z6) + d * d * Z8);
        X.sum4[ni][ai]+=(1.0/3600.0) * (-100 * a1*a1*a1 * Z0 + c * (10 * c * (c * Z12 + 3 * d * Z13) - 3 * d * d * Z14) -
                                        300 * a1*a1 * (b * Z1 + c * Z2 + d * Z3) +
                                        30 * a1 * (c * c * Z4 + b * (b * Z5 - 20 * c * Z3 + 2 * d * Z7) + 2 * c * d * Z6) -
                                        3 * a1 * d*d * Z8 - 3 * b * (d*d * Z11 - 10 * c * c * Z6 + 2 * c * d * Z8) + 10 * b * b * b * Z10 +
                                        30 * b * b * (c * Z7 + d * Z9));
        X.sum5[ni][ai]+=(1.0/12600.0) * (350 * a1*a1*a1*a1 * Z0 - 10 * b*b*b*b * Z15 +
                                         14 * b * c * (-10 * c*c * Z13 + 3 * c * d * Z14 + 3 * d*d * Z18) +
                                         2 * c*c * (-5 * c * (c * Z19 + 4 * d * Z20) + 3 * d*d * Z21) +
                                         1400 * a1*a1*a1 * (b * Z1 + c * Z2 + d * Z3) +
                                         3 * b*b * (2 * d * (7 * c * Z11 + d * Z17) + 7 * c*c * Z8) +
                                         21 * a1*a1 * (-10 * (c*c * Z4 + b * (b * Z5 - 20 * c * Z3 + 2 * d * Z7) + 2 * c * d * Z6) +
                                                       d*d * Z8) - 20 * b*b*b * (2 * d * Z16 + 7 * c * Z9) -
                                         14 * a1 * (c * (10 * c * (c * Z12 + 3 * d * Z13) - 3 * d*d * Z14) -
                                                    3 * b * (d*d * Z11 - 10 * c*c * Z6 + 2 * c * d * Z8) + 10 * b*b*b * Z10 +
                                                    30 * b*b * (c * Z7 + d * Z9)));
    }

    void pushW(cdouble w, const size_t ni, const size_t ai){
        this->ni = ni;
        this->ai = ai;
        c1 = 1.0/(w + a0[ni][ai]);
        c2 = c1*c1;
        c3 = c2*c1;
        c4 = c2*c2;
        c5 = c4*c1;
    }

    cdouble evaluate(Element &X){
        return c1*X.sum1[ni][ai] + c2*X.sum2[ni][ai] + c3*X.sum3[ni][ai] + c4*X.sum4[ni][ai] + c5*X.sum5[ni][ai];
    }

};

class TriangleIntegrator{

public:

    cdouble evaluate(const size_t i, const size_t j, const arr2d &P, const arr1d &Q, const double R){
        double z0, z1, z2, z3, z4, z5, z6, z7, z8;
        double c0, c1, c2, c3, c4, c5, c6, c7, c8;

        z0 = P[i*2][j*2]*Q[i*2]*R;
        z1 = P[i*2][j*2+1]*Q[i*2]*R;
        z2 = P[i*2][j*2+2]*Q[i*2]*R;
        z3 = P[i*2+1][j*2]*Q[i*2+1]*R;
        z4 = P[i*2+1][j*2+1]*Q[i*2+1]*R;
        z5 = P[i*2+1][j*2+2]*Q[i*2+1]*R;
        z6 = P[i*2+2][j*2]*Q[i*2+2]*R;
        z7 = P[i*2+2][j*2+1]*Q[i*2+2]*R;
        z8 = P[i*2+2][j*2+2]*Q[i*2+2]*R;

        c0 = z0;
        c1 = -3*z0 + 4*z1 - z2;
        c2 = 2*(z0 - 2*z1 + z2);
        c3 = -3*z0 + 4*z3 - z6;
        c4 = 9*z0 - 12*z1 + 3*z2 - 12*z3 + 16*z4 - 4*z5 + 3*z6 -4*z7 + z8;
        c5 = -2*(3*z0 - 6*z1 + 3*z2 - 4*z3 + 8*z4 - 4*z5 + z6 - 2*z7 + z8);
        c6 = 2*(z0 - 2*z3 + z6);
        c7 = -2*(3*z0 - 4*z1 +z2 - 6*z3 + 8*z4 -2*z5 +3*z6 -4*z7 + z8);
        c8 = 4*(z0 -2*z1 + z2 - 2*z3 + 4*z4 -2*z5 + z6 - 2*z7 + z8);

        return evaluate(c0, c1, c2, c3, c4, c5, c6, c7, c8, triA) +
                evaluate(c0, c1, c2, c3, c4, c5, c6, c7, c8, triB) +
                evaluate(c0, c1, c2, c3, c4, c5, c6, c7, c8, triC) +
                evaluate(c0, c1, c2, c3, c4, c5, c6, c7, c8, triD);
    }

    void pushTriangles(const double wi, const double z0, const double z1, const double z2, const double z3){
        cdouble a0, a3;
        double a1, a2, a4, a5;
        a0 = z0 + I*wi;
        a1 = z1 - z0;
        a2 = z2 - z0;
        a4 = z3 - z2;
        a5 = z3 - z1;
        a3 = z3 - a4 - a5 + I*wi;

        setupHalfTriangles(a0, a1, a2, false);
        setupHalfTriangles(a3, a4, a5, true);
    }

private:

    struct Triangle{
        cdouble p0, p1, p2, p3, p4, p5, p6, p7, p8;
        array<double, 2> QA, QB, QC;
    };

    Triangle triA, triB, triC, triD;

    cdouble evaluate(const double c0, const double c1, const double c2, const double c3, const double c4,
                     const double c5, const double c6, const double c7, const double c8, const Triangle &t){
        return c0*t.p0 + c1*t.p1 + c2*t.p2 + c3*t.p3 + c4*t.p4 + c5*t.p5 + c6*t.p6 + c7*t.p7 + c8*t.p8;
    }

    static array<double, 2> intersect(const array<double, 2> &P1, const array<double, 2> &P2, const double x){
        double m = (P1[1] - P2[1])/(P1[0] - P2[0]);
        double c = P1[1] - m*P1[0];
        return array<double, 2>{x, m*x + c};
    };

    void setupHalfTriangles(const cdouble a0, const double a1, const double a2, const bool upper){
        array<double, 2> Q0, Q1, Q2, Q4, Q5, Q6, Q7;
        cdouble a;
        double theta = atan2(a2, a1);
        double R0 = cos(theta);
        double R1 = sin(theta);
        a = (R0 * a1 + R1 * a2) / a0;
        if (upper){
            Q0[0] = 1.0*R0 + 1.0*R1;
            Q0[1] = 1.0*-R1 + 1.0*R0;
            Q1[0] = 1.0*R0 + 0.0*R1;
            Q1[1] = 1.0*-R1 + 0.0*R0;
            Q2[0] = 0.0*R0 + 1.0*R1;
            Q2[1] = 0.0*-R1 + 1.0*R0;
        }else{
            Q0[0] = 0.0*R0 + 0.0*R1;
            Q0[1] = 0.0*-R1 + 0.0*R0;
            Q1[0] = 1.0*R0 + 0.0*R1;
            Q1[1] = 1.0*-R1 + 0.0*R0;
            Q2[0] = 0.0*R0 + 1.0*R1;
            Q2[1] = 0.0*-R1 + 1.0*R0;
        }
        if ((Q1[0] < Q0[0] && Q0[0] < Q2[0]) || (Q2[0] < Q0[0] && Q0[0] < Q1[0])){
            Q4 = Q2;
            Q5 = Q0;
            Q6 = intersect(Q1, Q2, Q0[0]);
            Q7 = Q1;
        }
        else if ((Q0[0] < Q1[0] && Q1[0] < Q2[0]) || (Q2[0] < Q1[0] && Q1[0] < Q0[0])) {
            Q4 = Q2;
            Q5 = Q1;
            Q6 = intersect(Q0, Q2, Q1[0]);
            Q7 = Q0;
        }
        else{
            Q4 = Q0;
            Q5 = Q2;
            Q6 = intersect(Q0, Q1, Q2[0]);
            Q7 = Q1;
        }

        Triangle &tri1 = upper?triD:triB;
        Triangle &tri2 = upper?triC:triA;

        tri1.QA = Q4;
        tri1.QB = Q5;
        tri1.QC = Q6;
        tri2.QA = Q7;
        tri2.QB = Q5;
        tri2.QC = Q6;

        setupTriangle(tri1, R0, R1, a, a0);
        setupTriangle(tri2, R0, R1, a, a0);

    }

    void setupTriangle(Triangle &t, const double R0, const double R1, const cdouble a1, const cdouble a0){
        double mg0, mg1, cg0, cg1;
        double x0, x1;
        double mAB = (t.QA[1] - t.QB[1]) / (t.QA[0] - t.QB[0]);
        double cAB = t.QA[1] - mAB * t.QA[0];
        double mAC = (t.QA[1] - t.QC[1]) / (t.QA[0] - t.QC[0]);
        double cAC = t.QA[1] - mAC * t.QA[0];

        if (t.QC[1] > t.QB[1]) {
            mg0 = mAB;
            cg0 = cAB;
            mg1 = mAC;
            cg1 = cAC;
        } else {
            mg1 = mAB;
            cg1 = cAB;
            mg0 = mAC;
            cg0 = cAC;
        }

        if (t.QA[0] < t.QB[0]) {
            x0 = t.QA[0];
            x1 = t.QB[0];
        }
        else {
            x0 = t.QB[0];
            x1 = t.QA[0];
        }

        if (fabs(mg0>1E6)||fabs(mg1)>1E6){
            t.p0 = t.p1 = t.p2 = t.p3 = t.p4 = t.p5 = t.p6 = t.p7 = t.p8 = 0.0;
        }else{
            computeTriangleIntegral(mg0, mg1, cg0, cg1, x0, x1, R0, R1, a1, t);
            cdouble a0i = 1.0/a0;
            t.p0*=a0i;
            t.p1*=a0i;
            t.p2*=a0i;
            t.p3*=a0i;
            t.p4*=a0i;
            t.p5*=a0i;
            t.p6*=a0i;
            t.p7*=a0i;
            t.p8*=a0i;
        }
    }

    void computeTriangleIntegral(const double mg0, const double mg1, const double cg0, const double cg1,
                      const double x0, const double x1, const double R0, const double R1, const cdouble a1, Triangle& t){
        cdouble logarg = (a1 * (x0 - x1)) / (1.0 + a1 * x1);
        cdouble logfac = 0.5*log1p(((a1.real()*a1.real() + a1.imag()*a1.imag())*(x0*x0 -x1*x1) +
                2.0*a1.real()*(x0 - x1)) /(1.0 + (a1.real()*a1.real() + a1.imag()*a1.imag())*x1*x1 +
                2.0*a1.real()*x1)) + I*atan2(logarg.imag(), 1.0+logarg.real());
        cdouble a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, a1_8;
        cdouble o1_a1, o1_a1_2, o1_a1_3, o1_a1_4, o1_a1_5, o1_a1_6;
        cdouble logfac_a1, logfac_a1_2, logfac_a1_3, logfac_a1_4, logfac_a1_5, logfac_a1_6;
        double x0_2, x0_3, x0_4, x0_5;
        double x1_2, x1_3, x1_4, x1_5;
        double R0_2, R1_2, R0_3, R1_3, R0_4, R1_4;
        double mg0_2, mg1_2, mg0_3, mg1_3, mg0_4, mg1_4, mg0_5, mg1_5;
        double cg0_2, cg1_2, cg0_3, cg1_3, cg0_4, cg1_4, cg0_5, cg1_5;
        a1_2 = a1 * a1;
        a1_3 = a1_2 * a1;
        a1_4 = a1_3 * a1;
        a1_5 = a1_4 * a1;
        a1_6 = a1_5 * a1;
        a1_7 = a1_6 * a1;
        a1_8 = a1_7 * a1;
        o1_a1 = 1.0/a1;
        o1_a1_2 = o1_a1*o1_a1;
        o1_a1_3 = o1_a1_2*o1_a1;
        o1_a1_4 = o1_a1_3*o1_a1;
        o1_a1_5 = o1_a1_4*o1_a1;
        o1_a1_6 = o1_a1_5*o1_a1;
        logfac_a1 = logfac*o1_a1;
        logfac_a1_2 = logfac*o1_a1_2;
        logfac_a1_3 = logfac*o1_a1_3;
        logfac_a1_4 = logfac*o1_a1_4;
        logfac_a1_5 = logfac*o1_a1_5;
        logfac_a1_6 = logfac*o1_a1_6;
        x0_2 = x0*x0;
        x0_3 = x0*x0_2;
        x0_4 = x0*x0_3;
        x0_5 = x0*x0_4;
        x1_2 = x1*x1;
        x1_3 = x1*x1_2;
        x1_4 = x1*x1_3;
        x1_5 = x1*x1_4;
        R0_2 = R0*R0;
        R0_3 = R0_2*R0;
        R0_4 = R0_3*R0;
        R1_2 = R1*R1;
        R1_3 = R1_2*R1;
        R1_4 = R1_3*R1;
        mg0_2 = mg0*mg0;
        mg0_3 = mg0_2*mg0;
        mg0_4 = mg0_3*mg0;
        mg0_5 = mg0_4*mg0;
        mg1_2 = mg1*mg1;
        mg1_3 = mg1_2*mg1;
        mg1_4 = mg1_3*mg1;
        mg1_5 = mg1_4*mg1;
        cg0_2 = cg0*cg0;
        cg0_3 = cg0_2*cg0;
        cg0_4 = cg0_3*cg0;
        cg0_5 = cg0_4*cg0;
        cg1_2 = cg1*cg1;
        cg1_3 = cg1_2*cg1;
        cg1_4 = cg1_3*cg1;
        cg1_5 = cg1_4*cg1;


        t.p0 = logfac_a1 * (cg0 - cg1)  +
                logfac_a1_2 * (-mg0 + mg1) +
                o1_a1 * ((mg0 - mg1) * (x0 - x1));
                
        t.p1 = logfac_a1 * ((-cg0_2 + cg1_2) * R1*0.5) +
                logfac_a1_2 * ((-cg0 * R0 + cg1 * R0 + cg0 * mg0 * R1 - cg1 * mg1 * R1)) -
                logfac_a1_3 * ((mg0 - mg1) * (-2 * R0 + (mg0 + mg1) * R1))*0.5 +
                o1_a1 * ((x0 - x1) * (4 * cg0 * (R0 - mg0 * R1) - 4 * cg1 * (R0 - mg1 * R1) - (mg0 - mg1) * (-2 * R0 + (mg0 + mg1) * R1) * (x0 + x1)))*0.25 +
                o1_a1_2 * ((mg0 - mg1) * (-2 * R0 + (mg0 + mg1) * R1) * (x0 - x1))*0.5;

        t.p2 = logfac_a1 * ((cg0_3 - cg1_3) * R1_2*(1.0/3.0)) +
                logfac_a1_2 * (R1 * (cg0_2 * (R0 - mg0 * R1) + cg1_2 * (-R0 + mg1 * R1))) +
                logfac_a1_3 * ((cg0 * (R0 - mg0 * R1)*(R0 - mg0 * R1) - cg1 * (R0 - mg1 * R1)*(R0 - mg1 * R1))) +
                logfac_a1_4 * ((mg0 - mg1) * (3 * R0_2 - 3 * (mg0 + mg1) * R0 * R1 + (mg0_2 + mg0 * mg1 + mg1_2) * R1_2)*(-1.0/3.0)) +
                o1_a1_3 * (((mg0 - mg1) * (3 * R0_2 - 3 * (mg0 + mg1) * R0 * R1 + (mg0_2 + mg0 * mg1 + mg1_2) * R1_2) * (x0 - x1))*(1.0/3.0)) +
                o1_a1_2 * ((x0 - x1) * (6 * cg0 * (R0 - mg0 * R1)*(R0 - mg0 * R1) -
                6 * cg1 * (R0 - mg1 * R1) * (R0 - mg1 * R1) + (mg0 - mg1) * (3 * R0_2 -
                3 * (mg0 + mg1) * R0 * R1 + (mg0_2 + mg0 * mg1 + mg1_2) * R1_2) * (x0 + x1)) * (-1.0/6.0)) +
                o1_a1 * ((1.0/18.0) * (x0 - x1) * (18 * cg0_2 * R1 * (-R0 + mg0 * R1) +
                18 * cg1_2 * R1 * (R0 - mg1 * R1) + 9 * cg0 * (R0 - mg0 * R1)*(R0 - mg0 * R1) * (x0 + x1) -
                9 * cg1 * (R0 - mg1 * R1) * (R0 - mg1 * R1) * (x0 + x1) +
                2 * (mg0 - mg1) * (3 * R0_2 -
                3 * (mg0 + mg1) * R0 * R1 + (mg0_2 + mg0 * mg1 + mg1_2) * R1_2) * (x0_2 + x0 * x1 + x1_2)));

        t.p3 = logfac_a1*(((cg0 - cg1) * (cg0 + cg1) * R0) * 0.5) +
                logfac_a1_2 * (-cg0 * (mg0 * R0 + R1) + cg1 * (mg1 * R0 + R1)) +
                logfac_a1_3 * ((mg0 - mg1) * ((mg0 + mg1) * R0 + 2 * R1) * 0.5) +
                o1_a1_2 * ((mg0 - mg1) * ((mg0 + mg1) * R0 + 2 * R1) * (x0 - x1)*-0.5) +
                o1_a1 * (((x0 - x1) * (4 * cg0 * (mg0 * R0 + R1) -
                4 * cg1 * (mg1 * R0 + R1) + (mg0 - mg1) * ((mg0 + mg1) * R0 + 2 * R1) * (x0 +
                x1)))*0.25);

        t.p4 = logfac_a1 * (((-cg0_3 + cg1_3) * R0 * R1) * (1.0/3.0)) +
                logfac_a1_4 * ((mg0 - mg1) * (-3 * (mg0 + mg1) * R0_2 +
                2 * (-3 + mg0_2 + mg0 * mg1 + mg1_2) * R0 * R1 + 3 * (mg0 + mg1) * R1_2)*(1.0/6.0)) +
                logfac_a1_3 * (-cg0 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) +
                cg1 * (mg1 * R0 + R1) * (-R0 + mg1 * R1)) +
                logfac_a1_2 * (cg1_2 * (R0_2 - 2 * mg1 * R0 * R1 - R1_2) +
                cg0_2 * (-R0_2 + 2 * mg0 * R0 * R1 + R1_2))*0.5 +
                o1_a1_3 * (- ((mg0 - mg1) * (-3 * (mg0 + mg1) * R0_2 +
                2 * (-3 + mg0_2 + mg0 * mg1 + mg1_2) * R0 * R1 + 3 * (mg0 + mg1) * R1_2) * (x0 -
                x1)) * (1.0/6.0)) +
                o1_a1_2 * ((1.0/12.0) * (x0 - x1) * (12 * cg0 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) -
                12 * cg1 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) + (mg0 -
                mg1) * (-3 * (mg0 + mg1) * R0_2 +
                2 * (-3 + mg0_2 + mg0 * mg1 + mg1_2) * R0 * R1 +
                3 * (mg0 + mg1) * R1_2) * (x0 + x1))) +
                o1_a1*((1.0/18.0) * (x0 - x1) * (9 * cg0_2 * (R0_2 - 2 * mg0 * R0 * R1 - R1_2) +
                9 * cg1_2 * (-R0_2 + 2 * mg1 * R0 * R1 + R1_2) -
                9 * cg0 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (x0 + x1) +
                9 * cg1 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (x0 + x1) - (mg0 -
                mg1) * (-3 * (mg0 + mg1) * R0_2 +
                2 * (-3 + mg0_2 + mg0 * mg1 + mg1_2) * R0 * R1 +
                3 * (mg0 + mg1) * R1_2) * (x0_2 + x0 * x1 + x1_2)));

        t.p5 = logfac_a1 * (((cg0_4 - cg1_4) * R0 * R1_2)*0.25) +
                logfac_a1_5 * (1.0/12.0) * (6 * (mg0 - mg1) * (mg0 + mg1) * R0_3 +
                4 * (3 * mg0 - 2 * mg0_3 - 3 * mg1 + 2 * mg1_3) * R0_2 * R1 +
                3 * (mg0 - mg1) * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0 * R1_2 +
                4 * (mg0_3 - mg1_3) * R1_3) +
                logfac_a1_4 * - ((cg0 * (mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) -
                cg1 * (mg1 * R0 + R1) * (R0 - mg1 * R1)*(R0 - mg1 * R1))) +
                logfac_a1_3 * (((cg0_2 * (R0 - mg0 * R1) * (R0_2 - 3 * mg0 * R0 * R1 - 2 * R1_2) -
                cg1_2 * (R0 - mg1 * R1) * (R0_2 - 3 * mg1 * R0 * R1 - 2 * R1_2)))*0.5) +
                logfac_a1_2 * ((R1 * (cg0_3 * (2 * R0_2 - 3 * mg0 * R0 * R1 - R1_2) +
                cg1_3 * (-2 * R0_2 + 3 * mg1 * R0 * R1 + R1_2)))*(1.0/3.0)) +
                o1_a1_4 * (- (1.0/12.0) * (mg0 - mg1) * (6 * (mg0 + mg1) * R0_3 -
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0_2 * R1 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0 * R1_2 +
                4 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_3) * (x0 - x1)) +
                o1_a1_3 * ((1.0/24.0) * (x0 - x1) * (24 * cg0 * (mg0 * R0 + R1) * (R0 - mg0 * R1)*(R0 - mg0 * R1) -
                24 * cg1 * (mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1) + (mg0 -
                mg1) * (6 * (mg0 + mg1) * R0_3 -
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0_2 * R1 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0 * R1_2 +
                4 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_3) * (x0 + x1))) +
                o1_a1_2 * (-(1.0/36.0) * (x0 - x1) * (18 * cg0_2 * (R0 - mg0 * R1) * (R0_2 - 3 * mg0 * R0 * R1 - 2 * R1_2) -
                18 * cg1_2 * (R0 - mg1 * R1) * (R0_2 - 3 * mg1 * R0 * R1 - 2 * R1_2) +
                18 * cg0 * (mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) * (x0 + x1) -
                18 * cg1 * (mg1 * R0 + R1) * (R0 - mg1 * R1)*(R0 - mg1 * R1) * (x0 + x1) + (mg0 -
                mg1) * (6 * (mg0 + mg1) * R0_3 - 4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0_2 * R1 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0 * R1_2 +
                4 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_3) * (x0_2 + x0 * x1 + x1_2))) +
                o1_a1 * ((1.0/48.0) * (16 * cg0_3 * R1 * (-2 * R0_2 + 3 * mg0 * R0 * R1 + R1_2) * (x0 - x1) -
                16 * cg1_3 * R1 * (-2 * R0_2 + 3 * mg1 * R0 * R1 + R1_2) * (x0 - x1) +
                12 * cg0_2 * (R0 - mg0 * R1) * (R0_2 - 3 * mg0 * R0 * R1 - 2 * R1_2) * (x0 -
                x1) * (x0 + x1) - 12 * cg1_2 * (R0 - mg1 * R1) * (R0_2 - 3 * mg1 * R0 * R1 - 2 * R1_2) * (x0 -
                x1) * (x0 + x1) + 16 * cg0 * (mg0 * R0 + R1) * (R0 - mg0 * R1)*(R0 - mg0 * R1) * (x0_3 - x1_3) -
                16 * cg1 * (mg1 * R0 + R1) * (R0 - mg1 * R1)*(R0 - mg1 * R1) * (x0_3 - x1_3) + (mg0 -
                mg1) * (6 * (mg0 + mg1) * R0_3 - 4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0_2 * R1 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0 * R1_2 +
                4 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_3) * (x0_4 - x1_4)));

        t.p6 = logfac_a1 * ((cg0_3 - cg1_3) * R0_2*(1.0/3.0)) +
                logfac_a1_4 * (-(1.0/3.0) * ((mg0 - mg1) * ((mg0_2 + mg0 * mg1 + mg1_2) * R0_2 +
                3 * (mg0 + mg1) * R0 * R1 + 3 * R1_2))) +
                logfac_a1_2 * (R0 * (-cg0_2 * (mg0 * R0 + R1) + cg1_2 * (mg1 * R0 + R1))) +
                logfac_a1_3 * ((cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) - cg1 * (mg1 * R0 + R1) * (mg1 * R0
                + R1))) +
                o1_a1_3 * (((mg0 - mg1) * ((mg0_2 + mg0 * mg1 + mg1_2) * R0_2 + 3 * (mg0 + mg1) * R0 * R1 +
                3 * R1_2) * (x0 - x1)) * (1.0/3.0)) +
                o1_a1_2 * (- (1.0/6.0) * (x0 - x1) * (6 * cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) -
                6 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) + (mg0 -
                mg1) * ((mg0_2 + mg0 * mg1 + mg1_2) * R0_2 + 3 * (mg0 + mg1) * R0 * R1 +
                3 * R1_2) * (x0 + x1))) +
                (o1_a1) * ((1.0/18.0) * (x0 - x1) * (18 * cg0_2 * R0 * (mg0 * R0 + R1) -
                18 * cg1_2 * R0 * (mg1 * R0 + R1) + 9 * cg0 * (mg0 * R0 + R1)*(mg0 * R0 + R1) * (x0 + x1) -
                9 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (x0 + x1) +
                2 * (mg0 - mg1) * ((mg0_2 + mg0 * mg1 + mg1_2) * R0_2 +
                3 * (mg0 + mg1) * R0 * R1 + 3 * R1_2) * (x0_2 + x0 * x1 + x1_2)));

        t.p7 = logfac_a1 * (((-cg0_4 + cg1_4) * R0_2 * R1)*0.25) +
                logfac_a1_5 * ((1.0/12.0) * (4 * (mg0_3 - mg1_3) * R0_3 -
                3 * (mg0 - mg1) * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0_2 * R1 +
                4 * (3 * mg0 - 2 * mg0_3 - 3 * mg1 + 2 * mg1_3) * R0 * R1_2 +
                6 * (-mg0_2 + mg1_2) * R1_3)) +
                logfac_a1_4 * ((cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) * (-R0 + mg0 * R1) -
                cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (-R0 + mg1 * R1))) +
                logfac_a1_3 * (((-cg0_2 * (mg0 * R0 + R1) * (-2 * R0_2 + 3 * mg0 * R0 * R1 + R1_2) +
                cg1_2 * (mg1 * R0 + R1) * (-2 * R0_2 + 3 * mg1 * R0 * R1 + R1_2)))*0.5) +
                logfac_a1_2 * ((R0 * (cg1_3 * (R0_2 - 3 * mg1 * R0 * R1 - 2 * R1_2) +
                cg0_3 * (-R0_2 + 3 * mg0 * R0 * R1 + 2 * R1_2))) * (1.0/3.0)) +
                o1_a1_4 * ((1.0/12.0) * (mg0 - mg1) * (-4 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_3 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0_2 * R1 +
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0 * R1_2 +
                6 * (mg0 + mg1) * R1_3) * (x0 - x1)) +
                o1_a1_3 * ((-1.0/24.0) * (x0 - x1) * (24 * cg0 * (mg0 * R0 + R1)*(mg0 * R0 + R1) * (-R0 + mg0 * R1) -
                24 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (-R0 + mg1 * R1) + (mg0 -
                mg1) * (-4 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_3 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0_2 * R1 +
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0 * R1_2 +
                6 * (mg0 + mg1) * R1_3) * (x0 + x1))) +
                o1_a1_2 * ((1.0/36.0) * (x0 - x1) * (18 * cg0_2 * (mg0 * R0 + R1) * (-2 * R0_2 + 3 * mg0 * R0 * R1 + R1_2) -
                18 * cg1_2 * (mg1 * R0 + R1) * (-2 * R0_2 + 3 * mg1 * R0 * R1 + R1_2) +
                18 * cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (x0 + x1) -
                18 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (x0 + x1) + (mg0 -
                mg1) * (-4 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_3 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0_2 * R1 +
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0 * R1_2 +
                6 * (mg0 + mg1) * R1_3) * (x0_2 + x0 * x1 + x1_2))) +
                o1_a1 * ((1.0/48.0) * (16 * cg0_3 * R0 * (R0_2 - 3 * mg0 * R0 * R1 - 2 * R1_2) * (x0 - x1) -
                16 * cg1_3 * R0 * (R0_2 - 3 * mg1 * R0 * R1 - 2 * R1_2) * (x0 - x1) -
                12 * cg0_2 * (mg0 * R0 + R1) * (-2 * R0_2 + 3 * mg0 * R0 * R1 + R1_2) * (x0 -
                x1) * (x0 + x1) +
                12 * cg1_2 * (mg1 * R0 + R1) * (-2 * R0_2 + 3 * mg1 * R0 * R1 + R1_2) * (x0 -
                x1) * (x0 + x1) -
                16 * cg0 * (mg0 * R0 + R1)*(mg0 * R0 + R1) * (-R0 + mg0 * R1) * (x0_3 - x1_3) +
                16 * cg1 * (mg1 * R0 + R1)*(mg1 * R0 + R1) * (-R0 + mg1 * R1) * (x0_3 - x1_3) - (mg0 -
                mg1) * (-4 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_3 +
                3 * (mg0 + mg1) * (-4 + mg0_2 + mg1_2) * R0_2 * R1 +
                4 * (-3 + 2 * (mg0_2 + mg0 * mg1 + mg1_2)) * R0 * R1_2 +
                6 * (mg0 + mg1) * R1_3) * (x0_4 - x1_4)));


        t.p8 = logfac_a1 * (((cg0_5 - cg1_5) * R0_2 * R1_2)*0.2) +
                logfac_a1_6 * ((1.0/30.0) * (10 * (-mg0_3 + mg1_3) * R0_4 +
                15 * (mg0 - mg1) * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (-15 * mg0 + 20 * mg0_3 - 3 * mg0_5 + 15 * mg1 - 20 * mg1_3 +
                3 * mg1_5) * R0_2 * R1_2 - 15 * (mg0 - mg1) * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (-mg0_3 + mg1_3) * R1_4)) +
                logfac_a1_5 * ((cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) -
                cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1))) +
                logfac_a1_2 * ((R0 * R1 * (cg0_4 * (R0_2 - 2 * mg0 * R0 * R1 - R1_2) +
                cg1_4 * (-R0_2 + 2 * mg1 * R0 * R1 + R1_2)))*0.5) +
                logfac_a1_4 * (-cg0_2 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (-R0_2 + 2 * mg0 * R0 * R1 +
                R1_2) + cg1_2 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (-R0_2 +
                2 * mg1 * R0 * R1 + R1_2)) +
                logfac_a1_3 * ((1.0/3.0) * (cg0_3 * (R0_4 - 6 * mg0 * R0_3 * R1 + 2 * (-2 + 3 * mg0_2) * R0_2 * R1_2 +
                6 * mg0 * R0 * R1_3 + R1_4) -
                cg1_3 * (R0_4 - 6 * mg1 * R0_3 * R1 + 2 * (-2 + 3 * mg1_2) * R0_2 * R1_2 +
                6 * mg1 * R0 * R1_3 + R1_4))) +
                o1_a1_5 * ((1.0/30.0) * (mg0 - mg1) * (10 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_4 -
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (15 + 3 * mg0_4 + 3 * mg0_3 * mg1 - 20 * mg1_2 + 3 * mg1_4 +
                mg0_2 * (-20 + 3 * mg1_2) + mg0 * mg1 * (-20 + 3 * mg1_2)) * R0_2 * R1_2 +
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_4) * (x0 - x1)) +
                o1_a1_4 * ((-1.0/60.0) * (x0 - x1) * (60 * cg0 * (mg0 * R0 + R1)*(mg0 * R0 + R1) * (R0 - mg0 * R1)*(R0 - mg0 * R1) -
                60 * cg1 * (mg1 * R0 + R1)*(mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1) + (mg0 -
                mg1) * (10 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_4 -
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (15 - 20 * mg0_2 + 3 * mg0_4 +
                mg0 * (-20 + 3 * mg0_2) * mg1 + (-20 + 3 * mg0_2) * mg1_2 +
                3 * mg0 * mg1_3 + 3 * mg1_4) * R0_2 * R1_2 +
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_4) * (x0 + x1))) +
                o1_a1_3 * ((1.0/90.0) * (x0 -
                x1) * (90 * cg0_2 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (-R0_2 + 2 * mg0 * R0 * R1 +
                R1_2) - 90 * cg1_2 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (-R0_2 +
                2 * mg1 * R0 * R1 + R1_2) +
                45 * cg0 * (mg0 * R0 + R1)*(mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) * (x0 + x1) -
                45 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1) * (x0 + x1) + (mg0 -
                mg1) * (10 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_4 -
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (15 - 20 * mg0_2 + 3 * mg0_4 +
                mg0 * (-20 + 3 * mg0_2) * mg1 + (-20 + 3 * mg0_2) * mg1_2 +
                3 * mg0 * mg1_3 + 3 * mg1_4) * R0_2 * R1_2 +
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_4) * (x0_2 + x0 * x1 + x1_2))) +
                o1_a1_2 * ((1.0/120.0) * (-40 * cg0_3 * (R0_4 - 6 * mg0 * R0_3 * R1 +
                2 * (-2 + 3 * mg0_2) * R0_2 * R1_2 + 6 * mg0 * R0 * R1_3 + R1_4) * (x0 - x1) +
                40 * cg1_3 * (R0_4 - 6 * mg1 * R0_3 * R1 + 2 * (-2 + 3 * mg1_2) * R0_2 * R1_2 +
                6 * mg1 * R0 * R1_3 + R1_4) * (x0 - x1) +
                60 * cg1_2 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (-R0_2 + 2 * mg1 * R0 * R1 +
                R1_2) * (x0 - x1) * (x0 + x1) +
                60 * cg0_2 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (-R0_2 + 2 * mg0 * R0 * R1 +
                R1_2) * (-x0_2 + x1_2) - 40 * cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) * (x0_3 - x1_3) +
                40 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1) * (x0_3 - x1_3) - (mg0 -
                mg1) * (10 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_4 -
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (15 + 3 * mg0_4 + 3 * mg0_3 * mg1 - 20 * mg1_2 + 3 * mg1_4 +
                mg0_2 * (-20 + 3 * mg1_2) + mg0 * mg1 * (-20 + 3 * mg1_2)) * R0_2 * R1_2 +
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_4) * (x0_4 - x1_4))) +
                o1_a1 * ((1.0/300.0) * (-150 * cg0_4 * R0 * R1 * (R0_2 - 2 * mg0 * R0 * R1 - R1_2) * (x0 - x1) +
                150 * cg1_4 * R0 * R1 * (R0_2 - 2 * mg1 * R0 * R1 - R1_2) * (x0 - x1) +
                50 * cg0_3 * (R0_4 - 6 * mg0 * R0_3 * R1 + 2 * (-2 + 3 * mg0_2) * R0_2 * R1_2 +
                6 * mg0 * R0 * R1_3 + R1_4) * (x0 - x1) * (x0 + x1) -
                50 * cg1_3 * (R0_4 - 6 * mg1 * R0_3 * R1 + 2 * (-2 + 3 * mg1_2) * R0_2 * R1_2 +
                6 * mg1 * R0 * R1_3 + R1_4) * (x0 - x1) * (x0 + x1) +
                100 * cg0_2 * (mg0 * R0 + R1) * (-R0 + mg0 * R1) * (-R0_2 + 2 * mg0 * R0 * R1 +
                R1_2) * (x0_3 - x1_3) +
                100 * cg1_2 * (mg1 * R0 + R1) * (-R0 + mg1 * R1) * (-R0_2 + 2 * mg1 * R0 * R1 +
                R1_2) * (-x0_3 + x1_3) +
                75 * cg0 * (mg0 * R0 + R1) * (mg0 * R0 + R1) * (R0 - mg0 * R1) * (R0 - mg0 * R1) * (x0_4 - x1_4) -
                75 * cg1 * (mg1 * R0 + R1) * (mg1 * R0 + R1) * (R0 - mg1 * R1) * (R0 - mg1 * R1) * (x0_4 - x1_4) +
                2 * (mg0 - mg1) * (10 * (mg0_2 + mg0 * mg1 + mg1_2) * R0_4 -
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0_3 * R1 +
                2 * (15 + 3 * mg0_4 + 3 * mg0_3 * mg1 - 20 * mg1_2 + 3 * mg1_4 +
                mg0_2 * (-20 + 3 * mg1_2) + mg0 * mg1 * (-20 + 3 * mg1_2)) * R0_2 * R1_2 +
                15 * (mg0 + mg1) * (-2 + mg0_2 + mg1_2) * R0 * R1_3 +
                10 * (mg0_2 + mg0 * mg1 + mg1_2) * R1_4) * (x0_5 - x1_5)));

    }

};


class Species {

public://TODO: private later

    vector<int> ns;

    double charge, mass, density;
    double wp0, wc0;
    double dppara, dpperp;
    size_t npara, nperp;
    size_t npara_g, nperp_g;
    size_t npara_h, nperp_h;
    size_t n_taylor;

    arr2d f_p;
    arr1d ppara, pperp;

    arr1d ppara_h, pperp_h; //Half grid spacing
    arr2d df_dpperp_h, df_dppara_h;
    arr2d gamma, vpara_h, U0, U1, U2, U3, W0, W1;

    arr1d z;
    arr2d jns, jnds;
    arr1d das, a_mins;
    arr2d damaxs;
    arr2d kp0, kp1, kp2, kp3, kp4, kp5;
    TaylorSum series;

    Species(const boost::python::object &species, double B) {
        charge = p::extract<double>(species.attr("charge"));
        mass = p::extract<double>(species.attr("mass"));
        density = p::extract<double>(species.attr("density"));
        wc0 = B * abs(charge) / mass;
        wp0 = pow(density * charge * charge / (e0 * mass), 0.5);

        ppara = extract1d(species.attr("ppara")); //TODO: Extract just a scaler value
        pperp = extract1d(species.attr("pperp")); //TODO: Extract just a scaler value
        npara = ppara.size();
        nperp = pperp.size();
        f_p = extract2d(species.attr("f_p"), nperp, npara);

        dppara = ppara[1] - ppara[0];
        dpperp = pperp[1] - pperp[0];

        nperp_g = nperp/2 + 1;
        npara_g = npara/2 + 1;
        nperp_h = nperp_g*2 + 1;
        npara_h = npara_g*2 + 1;

        n_taylor = 1000;

        ppara_h = arr1d(npara_h, 0.0);
        pperp_h = arr1d(nperp_h, 0.0);

        arr2d f_p_h = arr2d(nperp_h, arr1d(npara_h, 0.0));
        df_dppara_h = arr2d(nperp_h, arr1d(npara_h, 0.0));
        df_dpperp_h = arr2d(nperp_h, arr1d(npara_h, 0.0));
        vpara_h = arr2d(nperp_h, arr1d(npara_h, 0.0));
        gamma = arr2d(nperp_h, arr1d(npara_h, 0.0));

        U0 = arr2d(nperp_h, arr1d(npara_h, 0.0));
        U1 = arr2d(nperp_h, arr1d(npara_h, 0.0));
        U2 = arr2d(nperp_h, arr1d(npara_h, 0.0));
        U3 = arr2d(nperp_h, arr1d(npara_h, 0.0));
        W0 = arr2d(nperp_h, arr1d(npara_h, 0.0));
        W1 = arr2d(nperp_h, arr1d(npara_h, 0.0));

        z = arr1d(nperp_h, 0.0);

        for (size_t i=0;i<nperp_h;i++){
            pperp_h[i] = (i + 0.5)*dpperp + pperp[0];
        }
        for (size_t j=0;j<npara_h;j++){
            ppara_h[j] = (j - 0.5)*dppara + ppara[0];
        }

        for (size_t i=0;i < nperp - 1; i++){
            for (size_t j=0;j < npara - 1; j++){
                double dppara_h = ppara[j+1] - ppara[j];
                double dpperp_h = pperp[i+1] - pperp[i];
                df_dpperp_h[i][j+1] = (f_p[i+1][j] - f_p[i][j] + f_p[i+1][j+1] - f_p[i][j+1])*0.5/dpperp_h;
                df_dppara_h[i][j+1] = (f_p[i][j+1] - f_p[i][j] + f_p[i+1][j+1] - f_p[i+1][j])*0.5/dppara_h;
                f_p_h[i][j+1] = 0.25*(f_p[i+1][j] + f_p[i][j] + f_p[i+1][j+1] + f_p[i][j+1]);
                //TODO: Make sure simpson is setup correctly ---> convergence order should be > O(1)
            }
        }


//        double sum = 0.0;
//        for (size_t i=0;i<nperp_g;i++){
//            for (size_t j=0;j<nperp_g;j++){
//                sum +=
//            }
//        }
        double sum = 0.0;
        for (size_t i=1;i < nperp_h -1 ; i++) {
            size_t cs1 = (1+i%2);
            for (size_t j = 1; j < npara_h -1; j++) {
                size_t cs = 4*cs1*(1+j%2);
                sum += cs*(f_p_h[i][j]*pperp_h[i]);
            }
        }
        sum = 2.0*M_PI*dppara*dpperp*sum/9.0;

        for (size_t i=0;i < nperp_h ; i++){
            for (size_t j=0;j<npara_h;j++){
                gamma[i][j] = sqrt(1.0 + pow(pperp_h[i]/(cl*mass), 2) + pow(ppara_h[j]/(cl*mass), 2));
                vpara_h[i][j] = ppara_h[j] / (gamma[i][j]*mass);
                df_dpperp_h[i][j]/=sum;
                df_dppara_h[i][j]/=sum;
                U0[i][j] = df_dpperp_h[i][j] / gamma[i][j];
                U1[i][j] = ppara_h[j]*df_dpperp_h[i][j]/gamma[i][j];
                U2[i][j] = (pperp_h[i]*df_dppara_h[i][j] - ppara_h[j]*df_dpperp_h[i][j])/gamma[i][j];
                U3[i][j] = ppara_h[j]*(pperp_h[i]*df_dppara_h[i][j] - ppara_h[j]*df_dpperp_h[i][j])/gamma[i][j];
                W0[i][j] = ppara_h[j]*pperp_h[i]*df_dppara_h[i][j]/gamma[i][j];
                W1[i][j] = wc0*ppara_h[j]*(ppara_h[j]*df_dpperp_h[i][j] - pperp_h[i]*df_dppara_h[i][j])/gamma[i][j];
            }
        }

        p::list pyns = p::extract<p::list>(species.attr("ns"));
        for (int i = 0; i < len(pyns); ++i) {
            int n{p::extract<int>(pyns[i])};
            ns.push_back(n);
        }
        jns = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        jnds = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp0 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp1 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp2 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp3 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp4 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
        kp5 = arr2d(ns.size(), arr1d(nperp_h, 0.0));
    }

    void push_kperp(const double kperp) {
        for (size_t i=0;i<nperp_h;i++){
            z[i] = kperp*pperp_h[i]/wc0;
        }
        for (size_t k=0;k<ns.size();k++){
            int n = ns[k];
            for (size_t i=0;i<nperp_h;i++) {
                jns[k][i] = bm::cyl_bessel_j(n, z[i]);
                jnds[k][i] = bm::cyl_bessel_j_prime(n, z[i]);
                kp0[k][i] = n*n*jns[k][i]*jns[k][i]*pperp_h[i]*pperp_h[i]/(z[i]*z[i]);
                kp1[k][i] = n*jns[k][i]*jnds[k][i]*pperp_h[i]*pperp_h[i]/z[i];
                kp2[k][i] = n*jns[k][i]*jns[k][i]*pperp_h[i]/z[i];
                kp3[k][i] = jnds[k][i]*jnds[k][i]*pperp_h[i]*pperp_h[i];
                kp4[k][i] = n*jns[k][i]*jnds[k][i]*pperp_h[i];
                kp5[k][i] = jns[k][i]*jns[k][i];
            }
        }
    }

    void push_kpara(const double kpara){
        series = TaylorSum(n_taylor, ns);
        das = arr1d(ns.size(), 0.0);
        a_mins = arr1d(ns.size(), 0.0);
        damaxs = arr2d(ns.size(), arr1d(n_taylor, 0.0));
        for (size_t k=0;k<ns.size();k++) {
            const int n = ns[k];
            double a_min{DBL_MAX}, a_max{DBL_MIN}, da;
            for (size_t i = 0; i < nperp_g + 1; i++) {
                for (size_t j = 0; j < npara_g + 1; j++) {
                    double a = -kpara*vpara_h[i*2][j*2] - n*gamma[i*2][j*2]*wc0;
                    if (a>a_max){
                        a_max = a;
                    }
                    if (a<a_min){
                        a_min = a;
                    }
                }
            }
            da = (a_max - a_min)/n_taylor;
            das[k] = da;
            a_mins[k] = a_min;
            series.setA(k, a_min, da);
            for (size_t i = 0; i < nperp_g; i++) {
                for (size_t j = 0; j < npara_g; j++) {
                    double a, a0, a1, b, c, d, z0, z1, z2, z3, damax;
                    size_t ai;
                    z0 = -kpara*vpara_h[i*2][j*2] - n*gamma[i*2][j*2]*wc0;
                    z1 = -kpara*vpara_h[i*2][j*2+2] - n*gamma[i*2][j*2+2]*wc0;
                    z2 = -kpara*vpara_h[i*2+2][j*2] - n*gamma[i*2+2][j*2]*wc0;
                    z3 = -kpara*vpara_h[i*2+2][j*2+2] - n*gamma[i*2+2][j*2+2]*wc0;
                    a = 0.25*(z0+z1+z2+z3);
                    ai = (size_t)((a - a_min)/da);
                    a0 = ai*da + a_min;
                    damax = max(max(fabs(z0 - a0), fabs(z1 - a0)), max(fabs(z2 - a0), fabs(z3 - a0)));
                    damaxs[k][ai] = max(damaxs[k][ai], damax);
                    a1 = z0 - a0;
                    b = z1 - z0; //TODO: check
                    c = z2 - z0; //TODO: check
                    d = z3 - z0 - b - c;
                    series.pushDenominator(k, i, j, ai, a1, b, c, d);
                    series.accumulate(series.sX00a, U0, kp0[k], 1.0);
                    series.accumulate(series.sX00b, U2, kp0[k], kpara);
                    series.accumulate(series.sX01a, U0, kp1[k], 1.0);
                    series.accumulate(series.sX01b, U2, kp1[k], kpara);
                    series.accumulate(series.sX02a, U1, kp2[k], 1.0);
                    series.accumulate(series.sX02b, U3, kp2[k], kpara);
                    series.accumulate(series.sX11a, U0, kp3[k], 1.0);
                    series.accumulate(series.sX11b, U2, kp3[k], kpara);
                    series.accumulate(series.sX12a, U1, kp4[k], 1.0);
                    series.accumulate(series.sX12b, U3, kp4[k], kpara);
                    series.accumulate(series.sX22a, W0, kp5[k], 1.0);
                    series.accumulate(series.sX22b, W1, kp5[k], n);
                }
            }
        }
    }

    array<array<cdouble, 3>, 3> push_omega(const double kpara, const double wr, const double wi) {
        array<array<cdouble, 3>, 3> X;
        cdouble X00{0.0, 0.0}, X01{0.0, 0.0}, X02{0.0, 0.0}, X11{0.0, 0.0}, X12{0.0, 0.0}, X22{0.0, 0.0};
        const cdouble w = wr + I*wi;
        const cdouble iw = 1.0/w;
        for (size_t k=0;k<ns.size();k++){
            for (size_t ai=0;ai<n_taylor;ai++){
                if (fabs(a_mins[k] + ai*das[k] + wr)<10*damaxs[k][ai]){
                    const int n = ns[k];
                    continue;
                    //cout<<series.mapping[k].size()<<endl;
                    for (pair<size_t, size_t> p:series.mapping[k][ai]){
                        double z0, z1, z2, z3;
                        size_t i{p.first}, j{p.second};
                        z0 = wr -kpara*vpara_h[i*2][j*2] - n*gamma[i*2][j*2]*wc0;
                        z1 = wr -kpara*vpara_h[i*2][j*2+2] - n*gamma[i*2][j*2+2]*wc0;
                        z2 = wr -kpara*vpara_h[i*2+2][j*2] - n*gamma[i*2+2][j*2]*wc0;
                        z3 = wr -kpara*vpara_h[i*2+2][j*2+2] - n*gamma[i*2+2][j*2+2]*wc0;

                        TriangleIntegrator integrator;
                        integrator.pushTriangles(wi, z0, z1, z2, z3);

                        X00 += integrator.evaluate(i, j, U0, kp0[k], 1.0);
                        X00 += integrator.evaluate(i, j, U2, kp0[k], kpara)*iw;
                        X01 += integrator.evaluate(i, j, U0, kp1[k], 1.0);
                        X01 += integrator.evaluate(i, j, U2, kp1[k], kpara)*iw;
                        X02 += integrator.evaluate(i, j, U1, kp2[k], 1.0);
                        X02 += integrator.evaluate(i, j, U3, kp2[k], kpara)*iw;
                        X11 += integrator.evaluate(i, j, U0, kp3[k], 1.0);
                        X11 += integrator.evaluate(i, j, U2, kp3[k], kpara)*iw;
                        X12 += integrator.evaluate(i, j, U1, kp4[k], 1.0);
                        X12 += integrator.evaluate(i, j, U3, kp4[k], kpara)*iw;
                        X22 += integrator.evaluate(i, j, W0, kp5[k], 1.0);
                        X22 += integrator.evaluate(i, j, W1, kp5[k], n)*iw;
                    }
                    continue;
                }

                series.pushW(w, k, ai);
                X00+=series.evaluate(series.sX00a);
                X00+=series.evaluate(series.sX00b)*iw;
                X01+=series.evaluate(series.sX01a);
                X01+=series.evaluate(series.sX01b)*iw;
                X02+=series.evaluate(series.sX02a);
                X02+=series.evaluate(series.sX02b)*iw;
                X11+=series.evaluate(series.sX11a);
                X11+=series.evaluate(series.sX11b)*iw;
                X12+=series.evaluate(series.sX12a);
                X12+=series.evaluate(series.sX12b)*iw;
                X22+=series.evaluate(series.sX22a);
                X22+=series.evaluate(series.sX22b)*iw;
            }
        }
        X[0][0] = X00*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[0][1] = I*X01*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[0][2] = X02*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[1][0] = -I*X01*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[1][1] = X11*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[1][2] = I*X12*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][0] = -X02*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][1] = -I*X12*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][2] = X22*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        //cout<<1.0 + X[0][0]<<", "<<X[0][1]<<", "<<1.0 + X[1][1]<<", "<<1.0 + X[2][2]<<endl;
        //cout<<X[0][0]<<", "<<X[0][1]<<", "<<X[1][1]<<", "<<X[2][2]<<endl;
        return X;
    }




//    cdouble square_integral(const size_t i, const size_t j, const arr2d &P, const arr1d &Q, const double R,
//                              const cdouble a0, const double a1, const double a2,
//                            const cdouble a3, const double a4, const double a5){
//        return triangle_integral(i, j, P, Q, R, a0, a1, a2, true) +
//                triangle_integral(i, j, P, Q, R, a3, a4, a5, false);
//    }

//    cdouble triangle_integral(const size_t i, const size_t j, const arr2d &P, const arr1d &Q, const double R,
//                              const cdouble a0, const double a1,
//                              const double a2, bool upper) {
//        double z0, z1, z2, z3, z4, z5, z6, z7, z8;
//        double c0, c1, c2, c3, c4, c5, c6, c7, c8;
//        double R0, R1;
//        array<double, 2> Q0, Q1, Q2, Q4, Q5, Q6, Q7;
//        cdouble a;
//        double theta = atan2(a2, a1);
//        //R0 = a1*sqrt(1.0 / (a1 * a1 + a2 * a2));
//        //R1 = a2*a1*sqrt(1.0 / (a1 * a1 *(a1 * a1 + a2 * a2)));
//        R0 = cos(theta);
//        R1 = sin(theta);
//        a = (R0 * a1 + R1 * a2) / a0;
//        //cout<< a1<<", "<<a2<<", "<<(-R1*a1 + R0*a2)<<", "<<(R0 * a1 + R1 * a2)<<endl;
//
//        z0 = P[i*2][j*2]*Q[i*2]*R;
//        z1 = P[i*2][j*2+1]*Q[i*2]*R;
//        z2 = P[i*2][j*2+2]*Q[i*2]*R;
//        z3 = P[i*2+1][j*2]*Q[i*2+1]*R;
//        z4 = P[i*2+1][j*2+1]*Q[i*2+1]*R;
//        z5 = P[i*2+1][j*2+2]*Q[i*2+1]*R;
//        z6 = P[i*2+2][j*2]*Q[i*2+2]*R;
//        z7 = P[i*2+2][j*2+1]*Q[i*2+2]*R;
//        z8 = P[i*2+2][j*2+2]*Q[i*2+2]*R;
//
//        c0 = z0;
//        c1 = -3*z0 + 4*z1 - z2;
//        c2 = 2*(z0 - 2*z1 + z2);
//        c3 = -3*z0 + 4*z3 - z6;
//        c4 = 9*z0 - 12*z1 + 3*z2 - 12*z3 + 16*z4 - 4*z5 + 3*z6 -4*z7 + z8;
//        c5 = -2*(3*z0 - 6*z1 + 3*z2 - 4*z3 + 8*z4 - 4*z5 + z6 - 2*z7 + z8);
//        c6 = 2*(z0 - 2*z3 + z6);
//        c7 = -2*(3*z0 - 4*z1 +z2 - 6*z3 + 8*z4 -2*z5 +3*z6 -4*z7 + z8);
//        c8 = 4*(z0 -2*z1 + z2 - 2*z3 + 4*z4 -2*z5 + z6 - 2*z7 + z8);
//
//        if (upper){
//            Q0[0] = 0.0*R0 + 0.0*R1;
//            Q0[1] = 0.0*-R1 + 0.0*R0;
//            Q1[0] = 1.0*R0 + 0.0*R1;
//            Q1[1] = 1.0*-R1 + 0.0*R0;
//            Q2[0] = 0.0*R0 + 1.0*R1;
//            Q2[1] = 0.0*-R1 + 1.0*R0;
//        }else{
//            Q0[0] = 1.0*R0 + 1.0*R1;
//            Q0[1] = 1.0*-R1 + 1.0*R0;
//            Q1[0] = 1.0*R0 + 0.0*R1;
//            Q1[1] = 1.0*-R1 + 0.0*R0;
//            Q2[0] = 0.0*R0 + 1.0*R1;
//            Q2[1] = 0.0*-R1 + 1.0*R0;
//        }
//        if ((Q1[0] < Q0[0] && Q0[0] < Q2[0]) || (Q2[0] < Q0[0] && Q0[0] < Q1[0])){
//            Q4 = Q2;
//            Q5 = Q0;
//            Q6 = intersect(Q1, Q2, Q0[0]);
//            Q7 = Q1;
//        }
//        else if ((Q0[0] < Q1[0] && Q1[0] < Q2[0]) || (Q2[0] < Q1[0] && Q1[0] < Q0[0])) {
//            Q4 = Q2;
//            Q5 = Q1;
//            Q6 = intersect(Q0, Q2, Q1[0]);
//            Q7 = Q0;
//        }
//        else{
//            Q4 = Q0;
//            Q5 = Q2;
//            Q6 = intersect(Q0, Q1, Q2[0]);
//            Q7 = Q1;
//        }
//        return (integrateHalfTriangle(Q4, Q5, Q6, R0, R1, c0, c1, c2, c3, c4, c5, c6, c7, c8, a)
//                + integrateHalfTriangle(Q7, Q5, Q6, R0, R1, c0, c1, c2, c3, c4, c5, c6, c7, c8, a))/a0;
//    }


//    cdouble integrateHalfTriangle(const array<double, 2> &QA, const array<double, 2> &QB, const array<double, 2> &QC,
//               const double R0, const double R1, double c0,
//               double c1, double c2, double c3, double c4, double c5, double c6,
//               double c7, double c8, const cdouble a1) {
//        double mg0, mg1, cg0, cg1;
//        double x0, x1;
//        double mAB = (QA[1] - QB[1]) / (QA[0] - QB[0]);
//        double cAB = QA[1] - mAB * QA[0];
//        double mAC = (QA[1] - QC[1]) / (QA[0] - QC[0]);
//        double cAC = QA[1] - mAC * QA[0];
//
//        if (QC[1] > QB[1]) {
//            mg0 = mAB;
//            cg0 = cAB;
//            mg1 = mAC;
//            cg1 = cAC;
//        } else {
//            mg1 = mAB;
//            cg1 = cAB;
//            mg0 = mAC;
//            cg0 = cAC;
//        }
//
//        if (QA[0] < QB[0]) {
//            x0 = QA[0];
//            x1 = QB[0];
//        }
//        else {
//            x0 = QB[0];
//            x1 = QA[0];
//        }
//
//        TriangleIntegrator integrator{};
//
//        if (fabs(mg0>1E6)||fabs(mg1)>1E6){
//            //cout<<QA[0]<<", "<<QA[1]<<", "<<QB[0]<<", "<<QB[1]<<", "<<QC[0]<<", "<<QC[1]<<endl;
//            return cdouble{0.0, 0.0};
//        }
//
//
//        integrator.pushTriangle(mg0, mg1, cg0, cg1, x0, x1, R0, R1, a1);
//        return integrator.evaluate(c0, c1, c2, c3, c4, c5, c6, c7, c8);
//
////        return aroundPole(c0, c1, c2, c3, c4, c5,
////                c6, c7, c8, x0, x1, cg0, cg1, mg0, mg1, R0, R1,
////                -R1, R0, a1);
//    }
/*
    static cdouble aroundPole(double c0, double c1, double c2, double c3, double c4, double c5,
                              double c6, double c7, double c8, double x0, double x1, double cg0,
                              double cg1, double mg0, double mg1, double R0, double R1,
                              double R2, double R3, cdouble a1){

        int I0_0, I0_1, I0_2, I0_3, I0_4, I0_5, I0_6, I0_7, I0_8, I0_9;
        int I0_10, I0_11, I0_12, I0_13, I0_14, I0_15, I0_16, I0_17, I0_18, I0_19;
        int I0_20, I0_21, I0_22, I0_23, I0_24, I0_25, I0_26, I0_27, I0_28, I0_29;
        int I0_30, I0_31, I0_32, I0_33, I0_34, I0_35, I0_36, I0_37, I0_38, I0_39;
        int I0_40, I0_41, I0_42, I0_43, I0_44, I0_45, I0_46;

        double R0_306;

        {
            I0_7 = -60;
            I0_28 = -15;
            I0_35 = -75;
            I0_44 = 150;
            I0_27 = -30;
            I0_25 = 9;
            I0_32 = 300;
            I0_37 = 200;
            I0_17 = -8;
            I0_20 = -6;
            I0_46 = 100;
            I0_13 = 10;
            I0_4 = 4;
            I0_33 = -5;
            I0_40 = -10;
            I0_12 = 8;
            I0_19 = -18;
            I0_30 = 3600;
            I0_42 = 600;
            I0_10 = -20;
            I0_8 = 30;
            I0_15 = 60;
            I0_5 = -4;
            R0_306 = 0.0002777777777777778;
            I0_34 = 900;
            I0_11 = 6;
            I0_21 = 48;
            I0_36 = 75;
            I0_16 = -12;
            I0_6 = 5;
            I0_3 = -2;
            I0_18 = 18;
            I0_43 = -600;
            I0_45 = -100;
            I0_23 = 16;
            I0_26 = -9;
            I0_31 = 1800;
            I0_2 = 2;
            I0_22 = 36;
            I0_14 = 15;
            I0_24 = 12;
            I0_38 = -200;
            I0_0 = 1;
            I0_1 = 3;
            I0_9 = -3;
            I0_41 = 120;
            I0_29 = 20;
            I0_39 = 80;
        }

        cdouble C00;
        cdouble C0_19, C0_26, C0_28, C0_29, C0_30, C0_31, C0_32, C0_33, C0_36, C0_37;
        cdouble C0_46, C0_48, C0_69, C0_70, C0_72, C0_81, C0_90, C0_102, C0_103, C0_104;
        cdouble C0_105, C0_106, C0_120, C0_127, C0_140, C0_141, C0_142, C0_144, C0_148;
        cdouble C0_149, C0_150, C0_151, C0_155, C0_156, C0_157, C0_172, C0_173, C0_174;
        cdouble C0_175, C0_176, C0_184, C0_191, C0_192, C0_193, C0_196, C0_201, C0_205;
        cdouble C0_206, C0_207, C0_208, C0_229, C0_231, C0_232, C0_243, C0_249, C0_251;
        cdouble C0_253, C0_254, C0_255, C0_325, C0_327, C0_329, C0_330, C0_331, C0_333;
        cdouble C0_260, C0_261, C0_262, C0_264, C0_265, C0_266, C0_270, C0_271, C0_272;
        cdouble C0_273, C0_275, C0_276, C0_278, C0_279, C0_281, C0_282, C0_285, C0_286;
        cdouble C0_230, C0_233, C0_244, C0_250, C0_252, C0_256, C0_258, C0_263, C0_267;
        cdouble C0_245, C0_220, C0_257, C0_280, C0_283, C0_293, C0_210, C0_211, C0_294;
        cdouble C0_295, C0_296, C0_299, C0_300, C0_301, C0_303, C0_304, C0_307, C0_309;
        cdouble C0_310, C0_312, C0_313, C0_311, C0_314, C0_315, C0_316, C0_317, C0_318;
        cdouble C0_319, C0_320, C0_321, C0_322, C0_323, C0_324, C0_326, C0_328, C0_297;
        cdouble C0_302, C0_305, C0_308, C0_332, C0_334, C0_335, C0_336, C0_337, C0_338;
        cdouble C0_339, C0_340, C0_341, C0_342, C0_343, C0_34, C0_35;

        double R0_0, R0_1, R0_2, R0_3, R0_4, R0_5, R0_6, R0_7, R0_8, R0_9;
        double R0_10, R0_11, R0_12, R0_13, R0_14, R0_15, R0_16, R0_17, R0_18;
        double R0_20, R0_21, R0_22, R0_23, R0_24, R0_25, R0_27;
        double R0_38, R0_39;
        double R0_40, R0_41, R0_42, R0_43, R0_44, R0_45, R0_47, R0_49;
        double R0_50, R0_51, R0_52, R0_53, R0_54, R0_55, R0_56, R0_57, R0_58, R0_59;
        double R0_60, R0_61, R0_62, R0_63, R0_64, R0_65, R0_66, R0_67, R0_68;
        double R0_71, R0_73, R0_74, R0_75, R0_76, R0_77, R0_78, R0_79;
        double R0_80, R0_82, R0_83, R0_84, R0_85, R0_86, R0_87, R0_88, R0_89;
        double R0_91, R0_92, R0_93, R0_94, R0_95, R0_96, R0_97, R0_98, R0_99;

        double R0_100, R0_101, R0_107, R0_108, R0_109;
        double R0_110, R0_111, R0_112, R0_113, R0_114, R0_115, R0_116, R0_117, R0_118, R0_119;
        double R0_121, R0_122, R0_123, R0_124, R0_125, R0_126, R0_128, R0_129;
        double R0_130, R0_131, R0_132, R0_133, R0_134, R0_135, R0_136, R0_137, R0_138, R0_139;
        double R0_143, R0_145, R0_146, R0_147;
        double R0_152, R0_153, R0_154, R0_158, R0_159;
        double R0_160, R0_161, R0_162, R0_163, R0_164, R0_165, R0_166, R0_167, R0_168, R0_169;
        double R0_170, R0_171, R0_177, R0_178, R0_179;
        double R0_180, R0_181, R0_182, R0_183, R0_185, R0_186, R0_187, R0_188, R0_189;
        double R0_190, R0_194, R0_195, R0_197, R0_198, R0_199;

        double R0_200, R0_202, R0_203, R0_204, R0_209;
        double R0_212, R0_213, R0_214, R0_215, R0_216, R0_217, R0_218, R0_219;
        double R0_221, R0_222, R0_223, R0_224, R0_225, R0_226, R0_227, R0_228;
        double R0_234, R0_235, R0_236, R0_237, R0_238, R0_239;
        double R0_240, R0_241, R0_242, R0_246, R0_247, R0_248;
        double R0_259;
        double R0_268, R0_269;
        double R0_274, R0_277;
        double R0_284, R0_287, R0_288, R0_289;
        double R0_290, R0_291, R0_292, R0_298;



        R0_0 = c0;
        R0_1 = c1;
        R0_2 = c2;
        R0_3 = c3;
        R0_4 = c4;
        R0_5 = c5;
        R0_6 = c6;
        R0_7 = c7;
        R0_8 = c8;
        R0_9 = x0;
        R0_10 = x1;
        R0_11 = cg0;
        R0_12 = cg1;
        R0_13 = mg0;
        R0_14 = mg1;
        R0_15 = R0;
        R0_16 = R1;
        R0_17 = R2;
        R0_18 = R3;
        C0_19 = a1;
        R0_20 = -R0_10;
        R0_21 = R0_9 + R0_20;
        R0_22 = -R0_14;
        R0_23 = R0_13 + R0_22;
        R0_24 = -R0_12;
        R0_25 = R0_11 + R0_24;
        C0_26 = C0_19 * R0_25;
        R0_27 = -R0_13;
        C0_28 = C0_26 + R0_27 + R0_14;
        C0_29 = C0_19 * R0_9;
        C0_30 = I0_0;
        C0_30 = C0_30 + C0_29;
        C0_31 = log(C0_30);
        C0_32 = C0_19 * R0_10;
        C0_33 = I0_0;
        C0_33 = C0_33 + C0_32;
        C0_34 = log(C0_33);
        C0_35 = -C0_34;
        C0_36 = C0_31 + C0_35;
        {
            int S0 = I0_1;
            cdouble S1 = C0_19;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_37 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_37 = S1 * C0_37;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_37 = 1. / C0_37;
            }
        }
        {
            int S0 = I0_1;
            double S1 = R0_10;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_38 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_38 = S1 * R0_38;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_38 = 1. / R0_38;
            }
        }
        R0_39 = I0_2;
        R0_39 = R0_39 * R0_11;
        R0_40 = I0_3;
        R0_40 = R0_40 * R0_12;
        R0_41 = R0_9 + R0_10;
        R0_42 = R0_23 * R0_41;
        R0_43 = R0_39 + R0_40 + R0_42;
        R0_44 = R0_18 * R0_18;
        R0_45 = R0_10 * R0_10;
        C0_46 = C0_19 * C0_19;
        {
            int S0 = I0_4;
            double S1 = R0_10;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_47 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_47 = S1 * R0_47;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_47 = 1. / R0_47;
            }
        }
        {
            int S0 = I0_4;
            cdouble S1 = C0_19;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_48 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_48 = S1 * C0_48;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_48 = 1. / C0_48;
            }
        }
        R0_49 = R0_27 + R0_14;
        R0_50 = R0_13 + R0_14;
        R0_51 = R0_17 * R0_17;
        R0_52 = I0_4;
        R0_52 = R0_52 * R0_11 * R0_13;
        R0_53 = I0_5;
        R0_53 = R0_53 * R0_12 * R0_14;
        R0_54 = R0_23 * R0_50 * R0_41;
        R0_55 = R0_52 + R0_53 + R0_54;
        R0_56 = R0_9 * R0_9;
        R0_57 = R0_11 * R0_11;
        R0_58 = R0_12 * R0_12;
        {
            int S0 = I0_1;
            double S1 = R0_9;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_59 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_59 = S1 * R0_59;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_59 = 1. / R0_59;
            }
        }
        R0_60 = -R0_38;
        R0_61 = R0_59 + R0_60;
        R0_62 = -R0_59;
        R0_63 = R0_62 + R0_38;
        {
            int S0 = I0_4;
            double S1 = R0_9;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_64 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_64 = S1 * R0_64;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_64 = 1. / R0_64;
            }
        }
        R0_65 = -R0_47;
        R0_66 = R0_64 + R0_65;
        {
            int S0 = I0_6;
            double S1 = R0_10;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_67 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_67 = S1 * R0_67;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_67 = 1. / R0_67;
            }
        }
        R0_68 = R0_11 + R0_12;
        C0_69 = C0_19 * R0_68;
        C0_70 = C0_69 + R0_27 + R0_22;
        R0_71 = I0_7;
        R0_71 = R0_71 * R0_23 * R0_50 * R0_21;
        C0_72 = I0_8;
        C0_72 = C0_72 * C0_19 * R0_21 * R0_55;
        R0_73 = I0_1;
        R0_73 = R0_73 * R0_57;
        R0_74 = I0_9;
        R0_74 = R0_74 * R0_58;
        R0_75 = I0_1;
        R0_75 = R0_75 * R0_11 * R0_13 * R0_41;
        R0_76 = I0_9;
        R0_76 = R0_76 * R0_12 * R0_14 * R0_41;
        R0_77 = R0_9 * R0_10;
        R0_78 = R0_56 + R0_77 + R0_45;
        R0_79 = R0_23 * R0_50 * R0_78;
        R0_80 = R0_73 + R0_74 + R0_75 + R0_76 + R0_79;
        C0_81 = I0_10;
        C0_81 = C0_81 * C0_46 * R0_21 * R0_80;
        R0_82 = I0_11;
        R0_82 = R0_82 * R0_57 * R0_21 * R0_41;
        R0_83 = -R0_56;
        R0_84 = R0_83 + R0_45;
        R0_85 = I0_11;
        R0_85 = R0_85 * R0_58 * R0_84;
        R0_86 = I0_12;
        R0_86 = R0_86 * R0_11 * R0_13 * R0_61;
        R0_87 = I0_12;
        R0_87 = R0_87 * R0_12 * R0_14 * R0_63;
        R0_88 = I0_1;
        R0_88 = R0_88 * R0_23 * R0_50 * R0_66;
        R0_89 = R0_82 + R0_85 + R0_86 + R0_87 + R0_88;
        C0_90 = I0_6;
        C0_90 = C0_90 * C0_37 * R0_89;
        R0_91 = I0_13;
        R0_91 = R0_91 * R0_58 * R0_61;
        R0_92 = I0_13;
        R0_92 = R0_92 * R0_57 * R0_63;
        R0_93 = I0_14;
        R0_93 = R0_93 * R0_12 * R0_14 * R0_66;
        R0_94 = -R0_64;
        R0_95 = R0_94 + R0_47;
        R0_96 = I0_14;
        R0_96 = R0_96 * R0_11 * R0_13 * R0_95;
        {
            int S0 = I0_6;
            double S1 = R0_9;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_97 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_97 = S1 * R0_97;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_97 = 1. / R0_97;
            }
        }
        R0_98 = -R0_97;
        R0_99 = R0_98 + R0_67;
        R0_100 = I0_11;
        R0_100 = R0_100 * R0_23 * R0_50 * R0_99;
        R0_101 = R0_91 + R0_92 + R0_93 + R0_96 + R0_100;
        C0_102 = I0_2;
        C0_102 = C0_102 * C0_48 * R0_101;
        C0_103 = R0_71 + C0_72 + C0_81 + C0_90 + C0_102;
        C0_104 = C0_19 * C0_103;
        C0_105 = I0_15;
        C0_105 = C0_105 * C0_70 * C0_28 * C0_36;
        C0_106 = C0_104 + C0_105;
        {
            int S0 = I0_4;
            double S1 = R0_13;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_107 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_107 = S1 * R0_107;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_107 = 1. / R0_107;
            }
        }
        {
            int S0 = I0_4;
            double S1 = R0_14;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_108 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_108 = S1 * R0_108;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_108 = 1. / R0_108;
            }
        }
        R0_109 = -R0_108;
        R0_110 = R0_107 + R0_109;
        {
            int S0 = I0_1;
            double S1 = R0_13;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_111 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_111 = S1 * R0_111;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_111 = 1. / R0_111;
            }
        }
        {
            int S0 = I0_1;
            double S1 = R0_14;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_112 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_112 = S1 * R0_112;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_112 = 1. / R0_112;
            }
        }
        R0_113 = R0_13 * R0_13;
        R0_114 = R0_14 * R0_14;
        R0_115 = I0_16;
        R0_115 = R0_115 * R0_110 * R0_21;
        R0_116 = I0_12;
        R0_116 = R0_116 * R0_11 * R0_111;
        R0_117 = I0_17;
        R0_117 = R0_117 * R0_12 * R0_112;
        R0_118 = R0_110 * R0_41;
        R0_119 = R0_116 + R0_117 + R0_118;
        C0_120 = I0_11;
        C0_120 = C0_120 * C0_19 * R0_21 * R0_119;
        R0_121 = I0_18;
        R0_121 = R0_121 * R0_57 * R0_113;
        R0_122 = I0_19;
        R0_122 = R0_122 * R0_58 * R0_114;
        R0_123 = I0_11;
        R0_123 = R0_123 * R0_11 * R0_111 * R0_41;
        R0_124 = I0_20;
        R0_124 = R0_124 * R0_12 * R0_112 * R0_41;
        R0_125 = R0_110 * R0_78;
        R0_126 = R0_121 + R0_122 + R0_123 + R0_124 + R0_125;
        C0_127 = I0_5;
        C0_127 = C0_127 * C0_46 * R0_21 * R0_126;
        {
            int S0 = I0_1;
            double S1 = R0_11;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_128 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_128 = S1 * R0_128;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_128 = 1. / R0_128;
            }
        }
        R0_129 = I0_21;
        R0_129 = R0_129 * R0_128 * R0_13 * R0_21;
        {
            int S0 = I0_1;
            double S1 = R0_12;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_130 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_130 = S1 * R0_130;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_130 = 1. / R0_130;
            }
        }
        R0_131 = -R0_9;
        R0_132 = R0_131 + R0_10;
        R0_133 = I0_21;
        R0_133 = R0_133 * R0_130 * R0_14 * R0_132;
        R0_134 = I0_22;
        R0_134 = R0_134 * R0_57 * R0_113 * R0_21 * R0_41;
        R0_135 = I0_22;
        R0_135 = R0_135 * R0_58 * R0_114 * R0_84;
        R0_136 = I0_23;
        R0_136 = R0_136 * R0_11 * R0_111 * R0_61;
        R0_137 = I0_23;
        R0_137 = R0_137 * R0_12 * R0_112 * R0_63;
        R0_138 = I0_1;
        R0_138 = R0_138 * R0_110 * R0_66;
        R0_139 = R0_129 + R0_133 + R0_134 + R0_135 + R0_136 + R0_137 + \
R0_138;
        C0_140 = C0_37 * R0_139;
        C0_141 = R0_115 + C0_120 + C0_127 + C0_140;
        C0_142 = C0_19 * C0_141;
        R0_143 = R0_57 + R0_58;
        C0_144 = C0_46 * R0_143;
        R0_145 = R0_11 * R0_13;
        R0_146 = R0_12 * R0_14;
        R0_147 = R0_145 + R0_146;
        C0_148 = I0_3;
        C0_148 = C0_148 * C0_19 * R0_147;
        C0_149 = C0_144 + R0_113 + R0_114 + C0_148;
        C0_150 = I0_24;
        C0_150 = C0_150 * C0_70 * C0_28 * C0_149 * C0_36;
        C0_151 = C0_142 + C0_150;
        R0_152 = R0_16 * R0_16;
        R0_153 = R0_15 * R0_15;
        R0_154 = I0_7;
        R0_154 = R0_154 * R0_110 * R0_21;
        C0_155 = I0_8;
        C0_155 = C0_155 * C0_19 * R0_21 * R0_119;
        C0_156 = I0_10;
        C0_156 = C0_156 * C0_46 * R0_21 * R0_126;
        C0_157 = I0_6;
        C0_157 = C0_157 * C0_37 * R0_139;
        {
            int S0 = I0_4;
            double S1 = R0_11;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_158 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_158 = S1 * R0_158;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_158 = 1. / R0_158;
            }
        }
        R0_159 = I0_6;
        R0_159 = R0_159 * R0_158 * R0_21;
        {
            int S0 = I0_4;
            double S1 = R0_12;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_160 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_160 = S1 * R0_160;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_160 = 1. / R0_160;
            }
        }
        R0_161 = I0_6;
        R0_161 = R0_161 * R0_160 * R0_132;
        R0_162 = I0_13;
        R0_162 = R0_162 * R0_128 * R0_13 * R0_21 * R0_41;
        R0_163 = I0_13;
        R0_163 = R0_163 * R0_130 * R0_14 * R0_84;
        R0_164 = I0_13;
        R0_164 = R0_164 * R0_57 * R0_113 * R0_61;
        R0_165 = I0_13;
        R0_165 = R0_165 * R0_58 * R0_114 * R0_63;
        R0_166 = I0_6;
        R0_166 = R0_166 * R0_11 * R0_111 * R0_66;
        R0_167 = I0_6;
        R0_167 = R0_167 * R0_12 * R0_112 * R0_95;
        R0_168 = -R0_67;
        R0_169 = R0_97 + R0_168;
        R0_170 = R0_110 * R0_169;
        R0_171 = R0_159 + R0_161 + R0_162 + R0_163 + R0_164 + R0_165 + R0_166 \
+ R0_167 + R0_170;
        C0_172 = I0_16;
        C0_172 = C0_172 * C0_48 * R0_171;
        C0_173 = R0_154 + C0_155 + C0_156 + C0_157 + C0_172;
        C0_174 = C0_19 * C0_173;
        C0_175 = I0_15;
        C0_175 = C0_175 * C0_70 * C0_28 * C0_149 * C0_36;
        C0_176 = C0_174 + C0_175;
        R0_177 = -R0_112;
        R0_178 = R0_111 + R0_177;
        R0_179 = I0_11;
        R0_179 = R0_179 * R0_178;
        R0_180 = I0_11;
        R0_180 = R0_180 * R0_11 * R0_113;
        R0_181 = I0_20;
        R0_181 = R0_181 * R0_12 * R0_114;
        R0_182 = R0_178 * R0_41;
        R0_183 = R0_180 + R0_181 + R0_182;
        C0_184 = I0_9;
        C0_184 = C0_184 * C0_19 * R0_183;
        R0_185 = I0_18;
        R0_185 = R0_185 * R0_57 * R0_13;
        R0_186 = I0_19;
        R0_186 = R0_186 * R0_58 * R0_14;
        R0_187 = I0_25;
        R0_187 = R0_187 * R0_11 * R0_113 * R0_41;
        R0_188 = I0_26;
        R0_188 = R0_188 * R0_12 * R0_114 * R0_41;
        R0_189 = I0_2;
        R0_189 = R0_189 * R0_178 * R0_78;
        R0_190 = R0_185 + R0_186 + R0_187 + R0_188 + R0_189;
        C0_191 = C0_46 * R0_190;
        C0_192 = R0_179 + C0_184 + C0_191;
        C0_193 = C0_19 * R0_21 * C0_192;
        R0_194 = -R0_130;
        R0_195 = R0_128 + R0_194;
        C0_196 = C0_37 * R0_195;
        R0_197 = -R0_111;
        R0_198 = I0_9;
        R0_198 = R0_198 * R0_57 * R0_13;
        R0_199 = I0_1;
        R0_199 = R0_199 * R0_58 * R0_14;
        R0_200 = R0_198 + R0_199;
        C0_201 = C0_46 * R0_200;
        R0_202 = R0_11 * R0_113;
        R0_203 = R0_12 * R0_114;
        R0_204 = -R0_203;
        R0_203 = R0_202 + R0_204;
        C0_205 = I0_1;
        C0_205 = C0_205 * C0_19 * R0_203;
        C0_206 = C0_196 + R0_197 + R0_112 + C0_201 + C0_205;
        C0_207 = I0_11;
        C0_207 = C0_207 * C0_206 * C0_36;
        C0_208 = C0_193 + C0_207;
        R0_209 = I0_15;
        R0_209 = R0_209 * R0_178 * R0_21;
        C0_210 = I0_27;
        C0_210 = C0_210 * C0_19 * R0_21 * R0_183;
        C0_211 = I0_13;
        C0_211 = C0_211 * C0_46 * R0_21 * R0_190;
        R0_212 = I0_4;
        R0_212 = R0_212 * R0_128 * R0_21;
        R0_213 = I0_4;
        R0_213 = R0_213 * R0_130 * R0_132;
        R0_214 = I0_11;
        R0_214 = R0_214 * R0_57 * R0_13 * R0_21 * R0_41;
        R0_215 = I0_11;
        R0_215 = R0_215 * R0_58 * R0_14 * R0_84;
        R0_216 = I0_4;
        R0_216 = R0_216 * R0_11 * R0_113 * R0_61;
        R0_217 = I0_4;
        R0_217 = R0_217 * R0_12 * R0_114 * R0_63;
        R0_218 = R0_178 * R0_66;
        R0_219 = R0_212 + R0_213 + R0_214 + R0_215 + R0_216 + R0_217 + \
R0_218;
        C0_220 = I0_28;
        C0_220 = C0_220 * C0_37 * R0_219;
        R0_221 = I0_13;
        R0_221 = R0_221 * R0_128 * R0_21 * R0_41;
        R0_222 = I0_13;
        R0_222 = R0_222 * R0_130 * R0_84;
        R0_223 = I0_29;
        R0_223 = R0_223 * R0_57 * R0_13 * R0_61;
        R0_224 = I0_29;
        R0_224 = R0_224 * R0_58 * R0_14 * R0_63;
        R0_225 = I0_14;
        R0_225 = R0_225 * R0_11 * R0_113 * R0_66;
        R0_226 = I0_14;
        R0_226 = R0_226 * R0_12 * R0_114 * R0_95;
        R0_227 = I0_4;
        R0_227 = R0_227 * R0_178 * R0_169;
        R0_228 = R0_221 + R0_222 + R0_223 + R0_224 + R0_225 + R0_226 + \
R0_227;
        C0_229 = I0_1;
        C0_229 = C0_229 * C0_48 * R0_228;
        C0_230 = R0_209 + C0_210 + C0_211 + C0_220 + C0_229;
        C0_231 = C0_19 * C0_230;
        C0_232 = I0_15;
        C0_232 = C0_232 * C0_206 * C0_36;
        C0_233 = C0_231 + C0_232;
        {
            int S0 = I0_6;
            double S1 = R0_13;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_234 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_234 = S1 * R0_234;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_234 = 1. / R0_234;
            }
        }
        {
            int S0 = I0_6;
            double S1 = R0_14;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            R0_235 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    R0_235 = S1 * R0_235;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                R0_235 = 1. / R0_235;
            }
        }
        R0_236 = -R0_235;
        R0_237 = R0_234 + R0_236;
        R0_238 = -R0_11;
        R0_239 = R0_238 + R0_12;
        R0_240 = I0_11;
        R0_240 = R0_240 * R0_23;
        R0_241 = I0_11;
        R0_241 = R0_241 * R0_12;
        R0_242 = I0_9;
        R0_242 = R0_242 * R0_23 * R0_41;
        C0_243 = C0_19 * R0_41;
        C0_244 = I0_3;
        C0_244 = C0_244 + C0_243;
        C0_245 = I0_1;
        C0_245 = C0_245 * R0_11 * C0_244;
        R0_246 = I0_9;
        R0_246 = R0_246 * R0_12 * R0_41;
        R0_247 = I0_2;
        R0_247 = R0_247 * R0_23 * R0_78;
        R0_248 = R0_246 + R0_247;
        C0_249 = C0_19 * R0_248;
        C0_250 = R0_241 + R0_242 + C0_245 + C0_249;
        C0_251 = C0_19 * C0_250;
        C0_252 = R0_240 + C0_251;
        C0_253 = C0_19 * R0_21 * C0_252;
        C0_254 = I0_11;
        C0_254 = C0_254 * C0_28 * C0_31;
        C0_255 = C0_19 * R0_239;
        C0_256 = C0_255 + R0_13 + R0_22;
        C0_257 = I0_11;
        C0_257 = C0_257 * C0_256 * C0_34;
        C0_258 = C0_253 + C0_254 + C0_257;
        R0_259 = I0_16;
        R0_259 = R0_259 * R0_23 * R0_50 * R0_21;
        C0_260 = I0_11;
        C0_260 = C0_260 * C0_19 * R0_21 * R0_55;
        C0_261 = I0_5;
        C0_261 = C0_261 * C0_46 * R0_21 * R0_80;
        C0_262 = C0_37 * R0_89;
        C0_263 = R0_259 + C0_260 + C0_261 + C0_262;
        C0_264 = C0_19 * C0_263;
        C0_265 = I0_24;
        C0_265 = C0_265 * C0_70 * C0_28 * C0_31;
        C0_266 = I0_16;
        C0_266 = C0_266 * C0_70 * C0_28 * C0_34;
        C0_267 = C0_264 + C0_265 + C0_266;
        R0_268 = I0_3;
        R0_268 = R0_268 * R0_13;
        R0_269 = I0_2;
        R0_269 = R0_269 * R0_14;
        C0_270 = C0_19 * R0_43;
        C0_271 = R0_268 + R0_269 + C0_270;
        C0_272 = -C0_31;
        C0_273 = C0_272 + C0_34;
        R0_274 = I0_3;
        R0_274 = R0_274 * R0_23 * R0_50;
        C0_275 = C0_19 * R0_55;
        C0_276 = R0_274 + C0_275;
        R0_277 = I0_11;
        R0_277 = R0_277 * R0_23 * R0_50;
        C0_278 = I0_9;
        C0_278 = C0_278 * C0_19 * R0_55;
        C0_279 = I0_2;
        C0_279 = C0_279 * C0_46 * R0_80;
        C0_280 = R0_277 + C0_278 + C0_279;
        C0_281 = C0_19 * R0_21 * C0_280;
        C0_282 = I0_11;
        C0_282 = C0_282 * C0_70 * C0_28 * C0_273;
        C0_283 = C0_281 + C0_282;
        R0_284 = I0_24;
        R0_284 = R0_284 * R0_23 * R0_50 * R0_21;
        C0_285 = I0_20;
        C0_285 = C0_285 * C0_19 * R0_21 * R0_55;
        C0_286 = I0_4;
        C0_286 = C0_286 * C0_46 * R0_21 * R0_80;
        R0_287 = I0_11;
        R0_287 = R0_287 * R0_58 * R0_21 * R0_41;
        R0_288 = I0_11;
        R0_288 = R0_288 * R0_57 * R0_84;
        R0_289 = I0_12;
        R0_289 = R0_289 * R0_12 * R0_14 * R0_61;
        R0_290 = I0_12;
        R0_290 = R0_290 * R0_11 * R0_13 * R0_63;
        R0_291 = I0_9;
        R0_291 = R0_291 * R0_23 * R0_50 * R0_66;
        R0_292 = R0_287 + R0_288 + R0_289 + R0_290 + R0_291;
        C0_293 = C0_37 * R0_292;
        C0_294 = R0_284 + C0_285 + C0_286 + C0_293;
        C0_295 = C0_19 * C0_294;
        C0_296 = I0_24;
        C0_296 = C0_296 * C0_70 * C0_28 * C0_273;
        C0_297 = C0_295 + C0_296;
        R0_298 = I0_16;
        R0_298 = R0_298 * R0_178 * R0_21;
        C0_299 = I0_11;
        C0_299 = C0_299 * C0_19 * R0_21 * R0_183;
        C0_300 = I0_3;
        C0_300 = C0_300 * C0_46 * R0_21 * R0_190;
        C0_301 = I0_1;
        C0_301 = C0_301 * C0_37 * R0_219;
        C0_302 = R0_298 + C0_299 + C0_300 + C0_301;
        C0_303 = C0_19 * C0_302;
        C0_304 = I0_24;
        C0_304 = C0_304 * C0_206 * C0_273;
        C0_305 = C0_303 + C0_304;
        {
            int S0 = I0_11;
            cdouble S1 = C0_19;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_307 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_307 = S1 * C0_307;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_307 = 1. / C0_307;
            }
        }
        C0_308 = 1. / C0_307;
        C0_307 = C0_19 * R0_23 * R0_21;
        C0_309 = C0_28 * C0_36;
        C0_307 = C0_307 + C0_309;
        C0_309 = I0_30;
        C0_309 = C0_309 * C0_48 * R0_0 * C0_307;
        C0_307 = C0_19 * R0_21 * C0_271;
        C0_310 = -C0_307;
        C0_307 = I0_2;
        C0_307 = C0_307 * C0_28 * C0_36;
        C0_310 = C0_310 + C0_307;
        C0_307 = I0_31;
        C0_307 = C0_307 * C0_37 * R0_3 * R0_17 * C0_310;
        C0_310 = I0_24;
        C0_310 = C0_310 * R0_23 * R0_21;
        C0_311 = I0_1;
        C0_311 = C0_311 * R0_11;
        C0_312 = I0_9;
        C0_312 = C0_312 * R0_12;
        C0_313 = I0_2;
        C0_313 = C0_313 * R0_13 * R0_9;
        C0_314 = I0_3;
        C0_314 = C0_314 * R0_14 * R0_9;
        C0_311 = C0_311 + C0_312 + C0_313 + C0_314;
        C0_312 = R0_56 * C0_311;
        C0_311 = I0_1;
        C0_311 = C0_311 * R0_239 * R0_45;
        C0_313 = I0_2;
        C0_313 = C0_313 * R0_49 * R0_38;
        C0_312 = C0_312 + C0_311 + C0_313;
        C0_311 = I0_2;
        C0_311 = C0_311 * C0_46 * C0_312;
        C0_312 = I0_5;
        C0_312 = C0_312 * R0_11;
        C0_313 = I0_4;
        C0_313 = C0_313 * R0_12;
        C0_314 = I0_9;
        C0_314 = C0_314 * R0_13 * R0_9;
        C0_315 = I0_1;
        C0_315 = C0_315 * R0_14 * R0_9;
        C0_312 = C0_312 + C0_313 + C0_314 + C0_315;
        C0_313 = R0_59 * C0_312;
        C0_312 = I0_4;
        C0_312 = C0_312 * R0_25 * R0_38;
        C0_314 = I0_1;
        C0_314 = C0_314 * R0_23 * R0_47;
        C0_313 = C0_313 + C0_312 + C0_314;
        C0_312 = C0_37 * C0_313;
        C0_313 = I0_20;
        C0_313 = C0_313 * C0_19 * R0_21 * R0_43;
        C0_310 = C0_310 + C0_311 + C0_312 + C0_313;
        C0_311 = C0_19 * C0_310;
        C0_310 = I0_24;
        C0_310 = C0_310 * C0_28 * C0_36;
        C0_311 = C0_311 + C0_310;
        C0_310 = I0_32;
        C0_310 = C0_310 * C0_19 * R0_5 * R0_17 * R0_44 * C0_311;
        C0_311 = I0_15;
        C0_311 = C0_311 * R0_23;
        C0_312 = I0_9;
        C0_312 = C0_312 * C0_19 * R0_9;
        C0_313 = I0_4;
        C0_313 = C0_313 + C0_312;
        C0_312 = C0_19 * R0_9 * C0_313;
        C0_313 = I0_20;
        C0_313 = C0_313 + C0_312;
        C0_312 = C0_19 * R0_9 * C0_313;
        C0_313 = I0_24;
        C0_313 = C0_313 + C0_312;
        C0_312 = I0_6;
        C0_312 = C0_312 * R0_12 * C0_313;
        C0_313 = I0_1;
        C0_313 = C0_313 * C0_19 * R0_9;
        C0_314 = I0_5;
        C0_314 = C0_314 + C0_313;
        C0_313 = C0_19 * R0_9 * C0_314;
        C0_314 = I0_11;
        C0_314 = C0_314 + C0_313;
        C0_313 = C0_19 * R0_9 * C0_314;
        C0_314 = I0_16;
        C0_314 = C0_314 + C0_313;
        C0_313 = I0_6;
        C0_313 = C0_313 * R0_11 * C0_314;
        C0_314 = I0_4;
        C0_314 = C0_314 * C0_19 * R0_9;
        C0_315 = I0_33;
        C0_315 = C0_315 + C0_314;
        C0_314 = I0_1;
        C0_314 = C0_314 * C0_19 * R0_9 * C0_315;
        C0_315 = I0_29;
        C0_315 = C0_315 + C0_314;
        C0_314 = C0_19 * R0_9 * C0_315;
        C0_315 = I0_27;
        C0_315 = C0_315 + C0_314;
        C0_314 = R0_23 * R0_9 * C0_315;
        C0_312 = C0_312 + C0_313 + C0_314;
        C0_313 = C0_19 * C0_312;
        C0_311 = C0_311 + C0_313;
        C0_313 = R0_9 * C0_311;
        C0_311 = I0_15;
        C0_311 = C0_311 * C0_28 * R0_10;
        C0_312 = I0_27;
        C0_312 = C0_312 * C0_19 * C0_28 * R0_45;
        C0_314 = I0_29;
        C0_314 = C0_314 * C0_46 * C0_28 * R0_38;
        C0_315 = I0_28;
        C0_315 = C0_315 * C0_37 * C0_28 * R0_47;
        C0_316 = I0_24;
        C0_316 = C0_316 * C0_48 * R0_49 * R0_67;
        C0_313 = C0_313 + C0_311 + C0_312 + C0_314 + C0_315 + C0_316;
        C0_311 = C0_19 * C0_313;
        C0_313 = I0_15;
        C0_313 = C0_313 * C0_28 * C0_36;
        C0_311 = C0_311 + C0_313;
        C0_313 = I0_15;
        C0_313 = C0_313 * R0_8 * R0_51 * R0_44 * C0_311;
        C0_311 = C0_19 * R0_21 * C0_276;
        C0_312 = I0_2;
        C0_312 = C0_312 * C0_70 * C0_28 * C0_36;
        C0_311 = C0_311 + C0_312;
        C0_312 = I0_34;
        C0_312 = C0_312 * C0_37 * R0_3 * R0_15 * C0_311;
        C0_311 = I0_15;
        C0_311 = C0_311 * R0_8 * R0_16 * R0_51 * R0_18 * C0_106;
        C0_314 = I0_15;
        C0_314 = C0_314 * R0_8 * R0_15 * R0_17 * R0_44 * C0_106;
        C0_315 = I0_35;
        C0_315 = C0_315 * C0_19 * R0_7 * R0_153 * R0_16 * C0_151;
        C0_316 = I0_36;
        C0_316 = C0_316 * C0_19 * R0_5 * R0_15 * R0_152 * C0_151;
        C0_317 = I0_8;
        C0_317 = C0_317 * R0_8 * R0_15 * R0_152 * R0_17 * C0_176;
        C0_318 = I0_8;
        C0_318 = C0_318 * R0_8 * R0_153 * R0_16 * R0_18 * C0_176;
        C0_319 = I0_37;
        C0_319 = C0_319 * C0_46 * R0_6 * R0_153 * C0_208;
        C0_320 = I0_38;
        C0_320 = C0_320 * C0_46 * R0_4 * R0_15 * R0_16 * C0_208;
        C0_321 = I0_37;
        C0_321 = C0_321 * C0_46 * R0_2 * R0_152 * C0_208;
        C0_322 = I0_29;
        C0_322 = C0_322 * R0_8 * R0_152 * R0_51 * C0_233;
        C0_323 = I0_39;
        C0_323 = C0_323 * R0_8 * R0_15 * R0_16 * R0_17 * R0_18 * C0_233;
        C0_324 = I0_29;
        C0_324 = C0_324 * R0_8 * R0_153 * R0_44 * C0_233;
        C0_325 = I0_15;
        C0_325 = C0_325 * R0_237 * R0_21;
        C0_326 = I0_13;
        C0_326 = C0_326 * R0_11 * R0_107;
        C0_327 = I0_40;
        C0_327 = C0_327 * R0_12 * R0_108;
        C0_328 = R0_237 * R0_41;
        C0_326 = C0_326 + C0_327 + C0_328;
        C0_327 = I0_27;
        C0_327 = C0_327 * C0_19 * R0_21 * C0_326;
        C0_326 = I0_15;
        C0_326 = C0_326 * R0_57 * R0_111;
        C0_328 = I0_7;
        C0_328 = C0_328 * R0_58 * R0_112;
        C0_329 = I0_14;
        C0_329 = C0_329 * R0_11 * R0_107 * R0_41;
        C0_330 = I0_28;
        C0_330 = C0_330 * R0_12 * R0_108 * R0_41;
        C0_331 = I0_2;
        C0_331 = C0_331 * R0_237 * R0_78;
        C0_326 = C0_326 + C0_328 + C0_329 + C0_330 + C0_331;
        C0_328 = I0_13;
        C0_328 = C0_328 * C0_46 * R0_21 * C0_326;
        C0_326 = I0_41;
        C0_326 = C0_326 * R0_128 * R0_113 * R0_21;
        C0_329 = I0_41;
        C0_329 = C0_329 * R0_130 * R0_114 * R0_132;
        C0_330 = I0_15;
        C0_330 = C0_330 * R0_57 * R0_111 * R0_21 * R0_41;
        C0_331 = I0_15;
        C0_331 = C0_331 * R0_58 * R0_112 * R0_84;
        C0_332 = I0_29;
        C0_332 = C0_332 * R0_11 * R0_107 * R0_61;
        C0_333 = I0_29;
        C0_333 = C0_333 * R0_12 * R0_108 * R0_63;
        C0_334 = I0_1;
        C0_334 = C0_334 * R0_237 * R0_66;
        C0_326 = C0_326 + C0_329 + C0_330 + C0_331 + C0_332 + C0_333 + \
C0_334;
        C0_329 = I0_33;
        C0_329 = C0_329 * C0_37 * C0_326;
        C0_326 = I0_32;
        C0_326 = C0_326 * R0_158 * R0_13 * R0_21;
        C0_330 = I0_32;
        C0_330 = C0_330 * R0_160 * R0_14 * R0_132;
        C0_331 = I0_32;
        C0_331 = C0_331 * R0_128 * R0_113 * R0_21 * R0_41;
        C0_332 = I0_32;
        C0_332 = C0_332 * R0_130 * R0_114 * R0_84;
        C0_333 = I0_37;
        C0_333 = C0_333 * R0_57 * R0_111 * R0_61;
        C0_334 = I0_37;
        C0_334 = C0_334 * R0_58 * R0_112 * R0_63;
        C0_335 = I0_36;
        C0_335 = C0_335 * R0_11 * R0_107 * R0_66;
        C0_336 = I0_36;
        C0_336 = C0_336 * R0_12 * R0_108 * R0_95;
        C0_337 = I0_24;
        C0_337 = C0_337 * R0_237 * R0_169;
        C0_326 = C0_326 + C0_330 + C0_331 + C0_332 + C0_333 + C0_334 + C0_335 \
+ C0_336 + C0_337;
        C0_330 = C0_48 * C0_326;
        C0_325 = C0_325 + C0_327 + C0_328 + C0_329 + C0_330;
        C0_327 = C0_19 * C0_325;
        {
            int S0 = I0_6;
            cdouble S1 = C0_19;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_325 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_325 = S1 * C0_325;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_325 = 1. / C0_325;
            }
        }
        {
            int S0 = I0_6;
            double S1 = R0_11;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_328 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_328 = S1 * C0_328;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_328 = 1. / C0_328;
            }
        }
        {
            int S0 = I0_6;
            double S1 = R0_12;
            bool S2 = 0;
            if( S0 < 0)
            {
                S2 = 1;
                S0 = -S0;
            }
            C0_329 = 1;
            while( S0)
            {
                if( S0 & 1)
                {
                    C0_329 = S1 * C0_329;
                }
                S1 = S1 * S1;
                S0 = S0 >> 1;
            }
            if( S2)
            {
                C0_329 = 1. / C0_329;
            }
        }
        C0_330 = -C0_329;
        C0_328 = C0_328 + C0_330;
        C0_325 = C0_325 * C0_328;
        C0_328 = -R0_234;
        C0_330 = I0_33;
        C0_330 = C0_330 * R0_158 * R0_13;
        C0_329 = I0_6;
        C0_329 = C0_329 * R0_160 * R0_14;
        C0_330 = C0_330 + C0_329;
        C0_329 = C0_48 * C0_330;
        C0_330 = R0_128 * R0_113;
        C0_326 = R0_130 * R0_114;
        C0_331 = -C0_326;
        C0_330 = C0_330 + C0_331;
        C0_331 = I0_13;
        C0_331 = C0_331 * C0_37 * C0_330;
        C0_330 = R0_57 * R0_111;
        C0_326 = -C0_330;
        C0_330 = R0_58 * R0_112;
        C0_326 = C0_326 + C0_330;
        C0_330 = I0_13;
        C0_330 = C0_330 * C0_46 * C0_326;
        C0_326 = R0_11 * R0_107;
        C0_332 = R0_12 * R0_108;
        C0_333 = -C0_332;
        C0_326 = C0_326 + C0_333;
        C0_333 = I0_6;
        C0_333 = C0_333 * C0_19 * C0_326;
        C0_325 = C0_325 + C0_328 + R0_235 + C0_329 + C0_331 + C0_330 + \
C0_333;
        C0_328 = I0_15;
        C0_328 = C0_328 * C0_325 * C0_36;
        C0_327 = C0_327 + C0_328;
        C0_328 = I0_24;
        C0_328 = C0_328 * R0_8 * R0_153 * R0_152 * C0_327;
        C0_327 = I0_42;
        C0_327 = C0_327 * C0_46 * R0_6 * R0_51 * C0_258;
        C0_325 = I0_43;
        C0_325 = C0_325 * C0_46 * R0_4 * R0_17 * R0_18 * C0_258;
        C0_329 = I0_42;
        C0_329 = C0_329 * C0_46 * R0_2 * R0_44 * C0_258;
        C0_331 = I0_32;
        C0_331 = C0_331 * C0_19 * R0_5 * R0_16 * R0_17 * R0_18 * C0_267;
        C0_330 = I0_44;
        C0_330 = C0_330 * C0_19 * R0_5 * R0_15 * R0_44 * C0_267;
        C0_333 = C0_19 * R0_21 * C0_271;
        C0_326 = I0_2;
        C0_326 = C0_326 * C0_28 * C0_273;
        C0_333 = C0_333 + C0_326;
        C0_326 = I0_31;
        C0_326 = C0_326 * C0_37 * R0_1 * R0_18 * C0_333;
        C0_333 = I0_16;
        C0_333 = C0_333 * R0_23 * R0_21;
        C0_332 = I0_9;
        C0_332 = C0_332 * R0_11;
        C0_334 = I0_1;
        C0_334 = C0_334 * R0_12;
        C0_335 = I0_3;
        C0_335 = C0_335 * R0_13 * R0_9;
        C0_336 = I0_2;
        C0_336 = C0_336 * R0_14 * R0_9;
        C0_332 = C0_332 + C0_334 + C0_335 + C0_336;
        C0_334 = R0_56 * C0_332;
        C0_332 = I0_1;
        C0_332 = C0_332 * R0_25 * R0_45;
        C0_335 = I0_2;
        C0_335 = C0_335 * R0_23 * R0_38;
        C0_334 = C0_334 + C0_332 + C0_335;
        C0_332 = I0_2;
        C0_332 = C0_332 * C0_46 * C0_334;
        C0_334 = I0_4;
        C0_334 = C0_334 * R0_11;
        C0_335 = I0_5;
        C0_335 = C0_335 * R0_12;
        C0_336 = I0_1;
        C0_336 = C0_336 * R0_13 * R0_9;
        C0_337 = I0_9;
        C0_337 = C0_337 * R0_14 * R0_9;
        C0_334 = C0_334 + C0_335 + C0_336 + C0_337;
        C0_335 = R0_59 * C0_334;
        C0_334 = I0_4;
        C0_334 = C0_334 * R0_239 * R0_38;
        C0_336 = I0_1;
        C0_336 = C0_336 * R0_49 * R0_47;
        C0_335 = C0_335 + C0_334 + C0_336;
        C0_334 = C0_37 * C0_335;
        C0_335 = I0_11;
        C0_335 = C0_335 * C0_19 * R0_21 * R0_43;
        C0_333 = C0_333 + C0_332 + C0_334 + C0_335;
        C0_332 = C0_19 * C0_333;
        C0_333 = I0_24;
        C0_333 = C0_333 * C0_28 * C0_273;
        C0_332 = C0_332 + C0_333;
        C0_333 = I0_32;
        C0_333 = C0_333 * C0_19 * R0_7 * R0_51 * R0_18 * C0_332;
        C0_332 = C0_19 * R0_21 * C0_276;
        C0_334 = -C0_332;
        C0_332 = I0_2;
        C0_332 = C0_332 * C0_70 * C0_28 * C0_273;
        C0_334 = C0_334 + C0_332;
        C0_332 = I0_34;
        C0_332 = C0_332 * C0_37 * R0_1 * R0_16 * C0_334;
        C0_334 = I0_43;
        C0_334 = C0_334 * C0_46 * R0_6 * R0_15 * R0_17 * C0_283;
        C0_335 = I0_32;
        C0_335 = C0_335 * C0_46 * R0_4 * R0_16 * R0_17 * C0_283;
        C0_336 = I0_32;
        C0_336 = C0_336 * C0_46 * R0_4 * R0_15 * R0_18 * C0_283;
        C0_337 = I0_43;
        C0_337 = C0_337 * C0_46 * R0_2 * R0_16 * R0_18 * C0_283;
        C0_338 = I0_44;
        C0_338 = C0_338 * C0_19 * R0_7 * R0_16 * R0_51 * C0_297;
        C0_339 = I0_32;
        C0_339 = C0_339 * C0_19 * R0_7 * R0_15 * R0_17 * R0_18 * C0_297;
        C0_340 = I0_37;
        C0_340 = C0_340 * C0_19 * R0_7 * R0_15 * R0_16 * R0_17 * C0_305;
        C0_341 = I0_45;
        C0_341 = C0_341 * C0_19 * R0_5 * R0_152 * R0_17 * C0_305;
        C0_342 = I0_46;
        C0_342 = C0_342 * C0_19 * R0_7 * R0_153 * R0_18 * C0_305;
        C0_343 = I0_38;
        C0_343 = C0_343 * C0_19 * R0_5 * R0_15 * R0_16 * R0_18 * C0_305;
        C0_309 = C0_309 + C0_307 + C0_310 + C0_313 + C0_312 + C0_311 + C0_314 \
+ C0_315 + C0_316 + C0_317 + C0_318 + C0_319 + C0_320 + C0_321 + \
C0_322 + C0_323 + C0_324 + C0_328 + C0_327 + C0_325 + C0_329 + C0_331 \
+ C0_330 + C0_326 + C0_333 + C0_332 + C0_334 + C0_335 + C0_336 + \
C0_337 + C0_338 + C0_339 + C0_340 + C0_341 + C0_342 + C0_343;
        C0_307 = R0_306 * C0_308 * C0_309;

        return C0_307;
    }
*/

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
        for (Species &spec: species) {
            spec.push_kpara(kpara);
        }
    }

    cdouble evaluate(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> X;
        for (Species &spec: species) {
            X+=spec.push_omega(kpara, wr, wi);
        }
        X[0][0] += 1.0;
        X[1][1] += 1.0;
        X[2][2] += 1.0;
        cdouble w = wr + I*wi;
        cdouble nx = cl*kperp/w;
        cdouble nz = cl*kpara/w;

        //X[0][0] -= nz*nz;
        //X[0][2] += nx*nz;
        //X[1][1] -= (nx*nx + nz*nz);
        //X[2][0] += nx*nz;
        //X[2][2] -= nx*nx;

        cdouble detM = 0.0;
        detM += nz*nz*nx*nx*X[2][2];
        detM += nz*nz*nz*nz*X[2][2];
        detM += X[0][0]*nz*nz*nx*nx;
        detM += X[0][0]*nx*nx*nx*nx;
        detM +=-nz*nz*X[1][1]*X[2][2];
        detM +=-X[0][0]*nx*nx*X[2][2];
        detM +=-X[0][0]*nz*nz*X[2][2];
        detM += -nx*nx*X[0][0]*X[1][1];
        detM += X[0][1]*X[1][0]*nx*nx;
        detM += nx*nz*X[0][1]*X[1][2];
        detM += X[2][1]*nz*nz*X[1][2];

        detM += X[0][0]*X[1][1]*X[2][2];
        detM +=-X[0][1]*X[1][0]*X[2][2];
        detM +=-X[1][1]*X[2][0]*X[0][2];
        detM +=-X[2][1]*X[0][0]*X[1][2];


        return detM;
    }

    np::ndarray marginalize(const np::ndarray &arr) {
        int nd = arr.get_nd();
        Py_intptr_t const *shape = arr.get_shape();
        Py_intptr_t const *stride = arr.get_strides();
        np::ndarray result = np::zeros(nd, shape, arr.get_dtype());
        if (nd == 1) {
            for (size_t i = 0; i < (size_t) shape[0]; i++) {
                cdouble w = *reinterpret_cast<cdouble const *>(arr.get_data() + i * stride[0]);
                cdouble res = evaluate(w.real(), w.imag());
                *reinterpret_cast<cdouble *>(result.get_data() + i * stride[0]) = res;
            }
        } else if (nd == 2) {
            for (size_t i = 0; i < (size_t) shape[0]; i++) {
                for (size_t j = 0; j < (size_t) shape[1]; j++) {
                    cdouble w = *reinterpret_cast<cdouble const *>(arr.get_data() + i * stride[0] + j * stride[1]);
                    cdouble res = evaluate(w.real(), w.imag());
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
            .def("evaluate", &Solver::evaluate)
            .def("marginalize", &Solver::marginalize);

}

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


class TaylorSum2 {

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
    cdouble c1, c2, c3, c4, c5;

    Element sX00a, sX01a, sX02a, sX11a, sX12a, sX22a;
    Element sX00b, sX01b, sX02b, sX11b, sX12b, sX22b;

    vector<vector<vector<pair<size_t, size_t>>>> mapping;

    struct Triangle{
        double p0[5], p1[5], p2[5], p3[5], p4[5], p5[5], p6[5], p7[5], p8[5];
        array<double, 2> QA, QB, QC;
        double R0, R1, y0, x0, x1, m0, m1, mx;
        bool active;
    };

    Triangle triA, triB, triC, triD;

    TaylorSum2(){
    }

    TaylorSum2(size_t n_a, vector<int> &ns) {
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
                                const double z0, const double z1, const double z2, const double z3){
        this->ni = ni;
        this->i = i;
        this->j = j;
        this->ai = ai;

        setupHalfTriangles(z1-z0, z2-z0, false);
        setupHalfTriangles(z3-z2, z3-z1, true);
        double a1 = z0;
        double a2 = z1 + z2 - z3;
        double da1 = a1 - a0[ni][ai];
        double da2 = a2 - a0[ni][ai];
        computeTriangleIntegral(triA, ni, ai, da1);
        computeTriangleIntegral(triB, ni, ai, da1);
        computeTriangleIntegral(triC, ni, ai, da2);
        computeTriangleIntegral(triD, ni, ai, da2);
        mapping[ni][ai].push_back(pair<size_t, size_t>(i, j));
    }

    static array<double, 2> intersect(const array<double, 2> &P1, const array<double, 2> &P2, const double x){
        double m = (P1[1] - P2[1])/(P1[0] - P2[0]);
        double c = P1[1] - m*P1[0];
        return array<double, 2>{x, m*x + c};
    };

    void setupTriangle(Triangle &t, const double R0, const double R1) {
        t.R0 = R0;
        t.R1 = R1;
        t.y0 = t.QA[1];
        t.x0 = t.QA[0];
        t.x1 = t.QB[0] - t.x0;
        double mAB = (t.QA[1] - t.QB[1]) / (t.QA[0] - t.QB[0]);
        double mAC = (t.QA[1] - t.QC[1]) / (t.QA[0] - t.QC[0]);
        t.m0 = (mAB > mAC) ? mAC : mAB;
        t.m1 = (mAB > mAC) ? mAB : mAC;
        t.active = true;
    }

    void zeroTriangle(Triangle &t){
        t.active = false;
    }

    void setupHalfTriangles(const double mx, const double my, const bool upper){
        array<double, 2> Q0, Q1, Q2, Q4, Q5, Q6, Q7;
        double theta = atan2(my, mx);
        double R0 = cos(theta);
        double R1 = sin(theta);
        //a = (R0 * mx + R1 * my) / a0;
        double mq = (R0 * mx + R1 * my);
        Triangle &tri1 = upper?triD:triB;
        Triangle &tri2 = upper?triC:triA;
        tri1.mx = mq;
        tri2.mx = mq;
        if (upper){
            //Triangles always clockwise.
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
        if (fabs(Q0[0] - Q1[0])<0.0000001){
            tri1.QA = Q2;
            tri1.QB = Q0;
            tri1.QC = Q1;
            setupTriangle(tri1, R0, R1);
            zeroTriangle(tri2);
            return;
        }
        if (fabs(Q0[0] - Q2[0])<0.0000001){
            tri1.QA = Q1;
            tri1.QB = Q0;
            tri1.QC = Q2;
            setupTriangle(tri1, R0, R1);
            zeroTriangle(tri2);
            return;
        }
        if (fabs(Q1[0] - Q2[0])<0.0000001){
            tri1.QA = Q0;
            tri1.QB = Q1;
            tri1.QC = Q2;
            setupTriangle(tri1, R0, R1);
            zeroTriangle(tri2);
            return;
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

        tri1.QA = Q4;
        tri1.QB = Q5;
        tri1.QC = Q6;
        tri2.QA = Q7;
        tri2.QB = Q5;
        tri2.QC = Q6;

        setupTriangle(tri1, R0, R1);
        setupTriangle(tri2, R0, R1);
    }

    void computeTriangleIntegral(Triangle& t, const size_t ni, const size_t ai, const double da) {

        if (!t.active){
            for (size_t l=0;l<5;l++) {
                t.p0[l] = 0.0;
                t.p1[l] = 0.0;
                t.p2[l] = 0.0;
                t.p3[l] = 0.0;
                t.p4[l] = 0.0;
                t.p5[l] = 0.0;
                t.p6[l] = 0.0;
                t.p7[l] = 0.0;
                t.p8[l] = 0.0;
            }
            return;
        }

        double x0 = t.x0;
        double x1 = t.x1;
        double y0 = t.y0;
        double R0 = t.R0;
        double R1 = t.R1;
        double m0 = t.m0;
        double m1 = t.m1;
        double mx = t.mx;
        double x0_2 = x0 * x0;
        double x0_3 = x0 * x0_2;
        double x0_4 = x0 * x0_3;
        double x1_2 = x1 * x1;
        double x1_3 = x1 * x1_2;
        double x1_4 = x1 * x1_3;
        double x1_5 = x1 * x1_4;
        double x1_6 = x1 * x1_5;
        double x1_7 = x1 * x1_6;
        double x1_8 = x1 * x1_7;
        double x1_9 = x1 * x1_8;
        double x1_10 = x1 * x1_9;
        double y0_2 = y0 * y0;
        double y0_3 = y0 * y0_2;
        double y0_4 = y0 * y0_3;
        double m0_2 = m0*m0;
        double m0_3 = m0_2*m0;
        double m0_4 = m0_3*m0;
        double m0_5 = m0_4*m0;
        double m1_2 = m1*m1;
        double m1_3 = m1_2*m1;
        double m1_4 = m1_3*m1;
        double m1_5 = m1_4*m1;
        double mx_2 = mx*mx;
        double mx_3 = mx*mx_2;
        double mx_4 = mx*mx_3;
        double da_2 = da * da;
        double da_3 = da*da_2;
        double da_4 = da*da_3;
        double R0_2 = R0*R0;
        double R0_3 = R0_2*R0;
        double R0_4 = R0_3*R0;
        double R1_2 = R1*R1;
        double R1_3 = R1_2*R1;
        double R1_4 = R1_3*R1;



        double O1[5], O2[5], O3[5], O4[5], O5[5];

        O1[0] = -(x1_2*0.5);
        O1[1] = (3 * da * x1_2 + 3 * mx * x0 * x1_2 - 2 * mx * x1_3)*(1./6.);
        O1[2] = (-6 * da_2 * x1_2 - 12 * da * mx * x0 * x1_2 - 6 * mx_2 * x0_2 * x1_2 +
                8 * da * mx * x1_3 + 8 * mx_2 * x0 * x1_3 - 3 * mx_2 * x1_4)*(1./12.);
        O1[3] = (1./20.) * (10 * da_3 * x1_2 + 30 * da_2 * mx * x0 * x1_2 +
                30 * da * mx_2 * x0_2 * x1_2 + 10 * mx_3 * x0_3 * x1_2 -
                20 * da_2 * mx * x1_3 - 40 * da * mx_2 * x0 * x1_3 -
                20 * mx_3 * x0_2 * x1_3 + 15 * da * mx_2 * x1_4 + 15 * mx_3 * x0 * x1_4 -
                4 * mx_3 * x1_5);
        O1[4] = -(1./30.) * x1_2 * (15 * da_4 + 20 * da_3 * mx * (3 * x0 - 2 * x1) +
                15 * da_2 * mx_2 * (6 * x0_2 - 8 * x0 * x1 + 3 * x1_2) +
                6 * da * mx_3 * (10 * x0_3 - 20 * x0_2 * x1 + 15 * x0 * x1_2 - 4 * x1_3) +
                mx_4 * (15 * x0_4 - 40 * x0_3 * x1 + 45 * x0_2 * x1_2 - 24 * x0 * x1_3 + 5 * x1_4));

        O2[0] = -(x1_3*(1./3.));
        O2[1] = (4 * da * x1_3 + 4 * mx * x0 * x1_3 - 3 * mx * x1_4)*(1./12.);
        O2[2] = (-10 * da_2 * x1_3 - 20 * da * mx * x0 * x1_3 - 10 * mx_2 * x0_2 * x1_3 +
                15 * da * mx * x1_4 + 15 * mx_2 * x0 * x1_4 - 6 * mx_2 * x1_5)*(1./30.);
        O2[3] = (1./60.) * (20 * da_3 * x1_3 + 60 * da_2 * mx * x0 * x1_3 +
                60 * da * mx_2 * x0_2 * x1_3 + 20 * mx_3 * x0_3 * x1_3 -
                45 * da_2 * mx * x1_4 - 90 * da * mx_2 * x0 * x1_4 -
                45 * mx_3 * x0_2 * x1_4 + 36 * da * mx_2 * x1_5 + 36 * mx_3 * x0 * x1_5 -
                10 * mx_3 * x1_6);
        O2[4] = (1./105.) * (-35 * da_4 * x1_3 - 140 * da_3 * mx * x0 * x1_3 -
                210 * da_2 * mx_2 * x0_2 * x1_3 - 140 * da * mx_3 * x0_3 * x1_3 -
                35 * mx_4 * x0_4 * x1_3 + 105 * da_3 * mx * x1_4 + 315 * da_2 * mx_2 * x0 * x1_4 +
                315 * da * mx_3 * x0_2 * x1_4 + 105 * mx_4 * x0_3 * x1_4 -
                126 * da_2 * mx_2 * x1_5 - 252 * da * mx_3 * x0 * x1_5 - 126 * mx_4 * x0_2 * x1_5 +
                70 * da * mx_3 * x1_6 + 70 * mx_4 * x0 * x1_6 - 15 * mx_4 * x1_7);

        O3[0] = -(x1_4*(1./4.));
        O3[1] = (5 * da * x1_4 + 5 * mx * x0 * x1_4 - 4 * mx * x1_5)*(1./20.);
        O3[2] = (-15 * da_2 * x1_4 - 30 * da * mx * x0 * x1_4 - 15 * mx_2 * x0_2 * x1_4 +
                24 * da * mx * x1_5 + 24 * mx_2 * x0 * x1_5 - 10 * mx_2 * x1_6)*(1./60.);
        O3[3] = (1./140.) * (35 * da_3 * x1_4 + 105 * da_2 * mx * x0 * x1_4 +
                105 * da * mx_2 * x0_2 * x1_4 + 35 * mx_3 * x0_3 * x1_4 - 84 * da_2 * mx * x1_5 -
                168 * da * mx_2 * x0 * x1_5 - 84 * mx_3 * x0_2 * x1_5 + 70 * da * mx_2 * x1_6 +
                70 * mx_3 * x0 * x1_6 - 20 * mx_3 * x1_7);
        O3[4] = (1./280.) * (-70 * da_4 * x1_4 - 280 * da_3 * mx * x0 * x1_4 -
                420 * da_2 * mx_2 * x0_2 * x1_4 - 280 * da * mx_3 * x0_3 * x1_4 -
                70 * mx_4 * x0_4 * x1_4 + 224 * da_3 * mx * x1_5 + 672 * da_2 * mx_2 * x0 * x1_5 +
                672 * da * mx_3 * x0_2 * x1_5 + 224 * mx_4 * x0_3 * x1_5 - 280 * da_2 * mx_2 * x1_6 -
                560 * da * mx_3 * x0 * x1_6 - 280 * mx_4 * x0_2 * x1_6 + 160 * da * mx_3 * x1_7 +
                160 * mx_4 * x0 * x1_7 - 35 * mx_4 * x1_8);

        O4[0] = -(x1_5*(1./5.));
        O4[1] = (6 * da * x1_5 + 6 * mx * x0 * x1_5 - 5 * mx * x1_6)*(1./30.);
        O4[2] = (-21 * da_2 * x1_5 - 42 * da * mx * x0 * x1_5 - 21 * mx_2 * x0_2 * x1_5 +
                35 * da * mx * x1_6 + 35 * mx_2 * x0 * x1_6 - 15 * mx_2 * x1_7)*(1./105.);
        O4[3] = (1./280.) * (56 * da_3 * x1_5 + 168 * da_2 * mx * x0 * x1_5 + 168 * da * mx_2 * x0_2 * x1_5 +
                56 * mx_3 * x0_3 * x1_5 - 140 * da_2 * mx * x1_6 - 280 * da * mx_2 * x0 * x1_6 -
                140 * mx_3 * x0_2 * x1_6 + 120 * da * mx_2 * x1_7 + 120 * mx_3 * x0 * x1_7 -
                35 * mx_3 * x1_8);
        O4[4] = (1./630.) * (-126 * da_4 * x1_5 - 504 * da_3 * mx * x0 * x1_5 - 756 * da_2 * mx_2 * x0_2 * x1_5 -
                504 * da * mx_3 * x0_3 * x1_5 - 126 * mx_4 * x0_4 * x1_5 + 420 * da_3 * mx * x1_6 +
                1260 * da_2 * mx_2 * x0 * x1_6 + 1260 * da * mx_3 * x0_2 * x1_6 + 420 * mx_4 * x0_3 * x1_6 -
                540 * da_2 * mx_2 * x1_7 - 1080 * da * mx_3 * x0 * x1_7 - 540 * mx_4 * x0_2 * x1_7 +
                315 * da * mx_3 * x1_8 + 315 * mx_4 * x0 * x1_8 - 70 * mx_4 * x1_9);

        O5[0] = -(x1_6*(1./6.));
        O5[1] = (7 * da * x1_6 + 7 * mx * x0 * x1_6 - 6 * mx * x1_7)*(1./42.);
        O5[2] = (-28 * da_2 * x1_6 - 56 * da * mx * x0 * x1_6 - 28 * mx_2 * x0_2 * x1_6 +
                48 * da * mx * x1_7 + 48 * mx_2 * x0 * x1_7 - 21 * mx_2 * x1_8)*(1./168.);
        O5[3] = (1./504.) * (84 * da_3 * x1_6 + 252 * da_2 * mx * x0 * x1_6 + 252 * da * mx_2 * x0_2 * x1_6 +
                84 * mx_3 * x0_3 * x1_6 - 216 * da_2 * mx * x1_7 - 432 * da * mx_2 * x0 * x1_7 -
                216 * mx_3 * x0_2 * x1_7 + 189 * da * mx_2 * x1_8 + 189 * mx_3 * x0 * x1_8 -
                56 * mx_3 * x1_9);
        O5[4] = (1./1260.) * (-210 * da_4 * x1_6 - 840 * da_3 * mx * x0 * x1_6 -
                1260 * da_2 * mx_2 * x0_2 * x1_6 - 840 * da * mx_3 * x0_3 * x1_6 -
                210 * mx_4 * x0_4 * x1_6 + 720 * da_3 * mx * x1_7 + 2160 * da_2 * mx_2 * x0 * x1_7 +
                2160 * da * mx_3 * x0_2 * x1_7 + 720 * mx_4 * x0_3 * x1_7 - 945 * da_2 * mx_2 * x1_8 -
                1890 * da * mx_3 * x0 * x1_8 - 945 * mx_4 * x0_2 * x1_8 + 560 * da * mx_3 * x1_9 +
                560 * mx_4 * x0 * x1_9 - 126 * mx_4 * x1_10);


        for (size_t l=0;l<5;l++) {
            double Ix4y0 = (m0 - m1) * O5[l];
            double Ix3y1 = 0.5 * (m0_2 - m1_2) * O5[l];
            double Ix2y2 = (1. / 3.) * (m0_3 - m1_3) * O5[l];
            double Ix1y3 = 0.25 * (m0_4 - m1_4) * O5[l];
            double Ix0y4 = 0.2 * (m0_5 - m1_5) * O5[l];
            double Ix3y0 = (-m0 + m1) * O4[l];
            double Ix2y1 = -0.5 * (m0_2 - m1_2) * O4[l];
            double Ix1y2 = (1.0 / 3.0) * (-m0_3 + m1_3) * O4[l];
            double Ix0y3 = 0.25 * (-m0_4 + m1_4) * O4[l];
            double Ix2y0 = (m0 - m1) * O3[l];
            double Ix1y1 = 0.5 * (m0_2 - m1_2) * O3[l];
            double Ix0y2 = (1.0 / 3.0) * (m0_3 - m1_3) * O3[l];
            double Ix1y0 = (-m0 + m1) * O2[l];
            double Ix0y1 = 0.5 * (-m0_2 + m1_2) * O2[l];
            double Ix0y0 = (m0 - m1) * O1[l];

            t.p0[l] = Ix0y0;
            t.p1[l] = (R0*Ix1y0
                       - R1*Ix0y1
                       + (R0*x0 - R1*y0)*Ix0y0);
            t.p2[l] = (R0_2*Ix2y0
                       - 2*R0*R1*Ix1y1
                       + (2.*R0_2*x0 - 2.*R0*R1*y0)*Ix1y0
                       + R1_2*Ix0y2
                       + (-2 * R0 * R1 * x0 + 2.*R1_2 * y0)*Ix0y1
                       + (R0_2*x0_2 - 2.*R0*R1*x0*y0 + R1_2*y0_2)*Ix0y0);
            t.p3[l] = (R1*Ix1y0
                       + R0*Ix0y1
                       + (R1*x0 + R0*y0)*Ix0y0);
            t.p4[l] = (R0*R1*Ix2y0
                       + (R0_2 - R1_2)*Ix1y1
                       + (2.*R0*R1*x0 + R0_2*y0 - R1_2*y0)*Ix1y0
                       - R0*R1*Ix0y2
                       + (R0_2*x0 - R1_2*x0 - 2.*R0*R1*y0)*Ix0y1
                       + (R0*R1*x0_2 + R0_2*x0*y0 - R1_2*x0*y0 - R0*R1*y0_2)*Ix0y0);
            t.p5[l] = (R0_2*R1*Ix3y0
                       + (R0_3 - 2*R0*R1_2)*Ix2y1
                       + (3 * R0_2 * R1 * x0 + R0_3 * y0 - 2 * R0 * R1_2 * y0)*Ix2y0
                       + (-2 * R0_2 * R1 + R1_3)*Ix1y2
                       + (2 * R0_3 * x0 - 4 * R0 * R1_2 * x0 - 4 * R0_2 * R1 * y0 + 2 * R1_3 * y0)*Ix1y1
                       + (3 * R0_2 * R1 * x0_2 + 2 * R0_3 * x0 * y0 - 4 * R0 * R1_2 * x0 * y0 - 2 * R0_2 * R1 * y0_2 + R1_3 * y0_2) * Ix1y0
                       + (R0 * R1_2) * Ix0y3
                       + (-2 * R0_2 * R1 * x0 + R1_3 * x0 + 3 * R0 * R1_2 * y0)*Ix0y2
                       + (R0_3 * x0_2 - 2 * R0 * R1_2 * x0_2 - 4 * R0_2 * R1 * x0 * y0 + 2 * R1_3 * x0 * y0 + 3 * R0 * R1_2 * y0_2) * Ix0y1
                       + (R0_2 * R1 * x0_3 + R0_3 * x0_2 * y0 - 2 * R0 * R1_2 * x0_2 * y0 - 2 * R0_2 * R1 * x0 * y0_2 + R1_3 * x0 * y0_2 + R0 * R1_2 * y0_3)*Ix0y0);
            t.p6[l] = (R1_2 * Ix2y0
                       + (2 * R0 * R1)*Ix1y1
                       + (2 * R1_2 * x0 + 2 * R0 * R1 * y0) * Ix1y0
                       + R0_2*Ix0y2
                       + (2 * R0 * R1 * x0 + 2 * R0_2 * y0)*Ix0y1
                       + (R1_2 * x0_2 + 2 * R0 * R1 * x0 * y0 + R0_2 * y0_2)*Ix0y0);
            t.p7[l] = ((R0 * R1_2) * Ix3y0
                       + (2 * R0_2 * R1 - R1_3)*Ix2y1
                       + (3 * R0 * R1_2 * x0 + 2 * R0_2 * R1 * y0 - R1_3 * y0)*Ix2y0
                       + (R0_3 - 2 * R0 * R1_2)*Ix1y2
                       + (4 * R0_2 * R1 * x0 - 2 * R1_3 * x0 + 2 * R0_3 * y0 - 4 * R0 * R1_2 * y0) * Ix1y1
                       + (3 * R0 * R1_2 * x0_2 + 4 * R0_2 * R1 * x0 * y0 - 2 * R1_3 * x0 * y0 + R0_3 * y0_2 - 2 * R0 * R1_2 * y0_2) * Ix1y0
                       + (-R0_2 * R1)*Ix0y3
                       + (R0_3 * x0 - 2 * R0 * R1_2 * x0 - 3 * R0_2 * R1 * y0)*Ix0y2
                       + (2 * R0_2 * R1 * x0_2 - R1_3 * x0_2 + 2 * R0_3 * x0 * y0 - 4 * R0 * R1_2 * x0 * y0 - 3 * R0_2 * R1 * y0_2)*Ix0y1
                       + (R0 * R1_2 * x0_3 + 2 * R0_2 * R1 * x0_2 * y0 - R1_3 * x0_2 * y0 + R0_3 * x0 * y0_2 - 2 * R0 * R1_2 * x0 * y0_2 - R0_2 * R1 * y0_3)*Ix0y0);
            t.p8[l] = ((R0_2 * R1_2)*Ix4y0
                       + (2 * R0_3 * R1 - 2 * R0 * R1_3)*Ix3y1
                       + (4 * R0_2 * R1_2 * x0 + 2 * R0_3 * R1 * y0 - 2 * R0 * R1_3 * y0)*Ix3y0
                       + (R0_4 - 4 * R0_2 * R1_2 + R1_4)*Ix2y2
                       + (6 * R0_3 * R1 * x0 - 6 * R0 * R1_3 * x0 + 2 * R0_4 * y0 - 8 * R0_2 * R1_2 * y0 + 2 * R1_4 * y0)*Ix2y1
                       + (6 * R0_2 * R1_2 * x0_2 + 6 * R0_3 * R1 * x0 * y0 - 6 * R0 * R1_3 * x0 * y0 + R0_4 * y0_2 - 4 * R0_2 * R1_2 * y0_2 + R1_4 * y0_2)*Ix2y0
                       + (-2 * R0_3 * R1 + 2 * R0 * R1_3)*Ix1y3
                       + (2 * R0_4 * x0 - 8 * R0_2 * R1_2 * x0 + 2 * R1_4 * x0 - 6 * R0_3 * R1 * y0 + 6 * R0 * R1_3 * y0)*Ix1y2
                       + (6 * R0_3 * R1 * x0_2 - 6 * R0 * R1_3 * x0_2 + 4 * R0_4 * x0 * y0 - 16 * R0_2 * R1_2 * x0 * y0 + 4 * R1_4 * x0 * y0 - 6 * R0_3 * R1 * y0_2 + 6 * R0 * R1_3 * y0_2)*Ix1y1
                       + (4 * R0_2 * R1_2 * x0_3 + 6 * R0_3 * R1 * x0_2 * y0 - 6 * R0 * R1_3 * x0_2 * y0 + 2 * R0_4 * x0 * y0_2 - 8 * R0_2 * R1_2 * x0 * y0_2 + 2 * R1_4 * x0 * y0_2 - 2 * R0_3 * R1 * y0_3 + 2 * R0 * R1_3 * y0_3)*Ix1y0
                       + (R0_2 * R1_2)*Ix0y4
                       + (-2 * R0_3 * R1 * x0 + 2 * R0 * R1_3 * x0 + 4 * R0_2 * R1_2 * y0)*Ix0y3
                       + (R0_4 * x0_2 - 4 * R0_2 * R1_2 * x0_2 + R1_4 * x0_2 - 6 * R0_3 * R1 * x0 * y0 + 6 * R0 * R1_3 * x0 * y0 + 6 * R0_2 * R1_2 * y0_2)*Ix0y2
                       + (2 * R0_3 * R1 * x0_3 - 2 * R0 * R1_3 * x0_3 + 2 * R0_4 * x0_2 * y0 - 8 * R0_2 * R1_2 * x0_2 * y0 + 2 * R1_4 * x0_2 * y0 - 6 * R0_3 * R1 * x0 * y0_2 + 6 * R0 * R1_3 * x0 * y0_2 + 4 * R0_2 * R1_2 * y0_3)*Ix0y1
                       + (R0_2 * R1_2 * x0_4 + 2 * R0_3 * R1 * x0_3 * y0 - 2 * R0 * R1_3 * x0_3 * y0 + R0_4 * x0_2 * y0_2 - 4 * R0_2 * R1_2 * x0_2 * y0_2 + R1_4 * x0_2 * y0_2 - 2 * R0_3 * R1 * x0 * y0_3 + 2 * R0 * R1_3 * x0 * y0_3 + R0_2 * R1_2 * y0_4)*Ix0y0);
        }
    }


    void accumulate(Element &X, const arr2d &p, const arr1d &q, const double r){
        double z0 = p[i*2][j*2]*q[i*2]*r;
        double z1 = p[i*2][j*2+1]*q[i*2]*r;
        double z2 = p[i*2][j*2+2]*q[i*2]*r;
        double z3 = p[i*2+1][j*2]*q[i*2+1]*r;
        double z4 = p[i*2+1][j*2+1]*q[i*2+1]*r;
        double z5 = p[i*2+1][j*2+2]*q[i*2+1]*r;
        double z6 = p[i*2+2][j*2]*q[i*2+2]*r;
        double z7 = p[i*2+2][j*2+1]*q[i*2+2]*r;
        double z8 = p[i*2+2][j*2+2]*q[i*2+2]*r;

        double c0 = z0;
        double c1 = -3*z0 + 4*z1 - z2;
        double c2 = 2*(z0 - 2*z1 + z2);
        double c3 = -3*z0 + 4*z3 - z6;
        double c4 = 9*z0 - 12*z1 + 3*z2 - 12*z3 + 16*z4 - 4*z5 + 3*z6 -4*z7 + z8;
        double c5 = -2*(3*z0 - 6*z1 + 3*z2 - 4*z3 + 8*z4 - 4*z5 + z6 - 2*z7 + z8);
        double c6 = 2*(z0 - 2*z3 + z6);
        double c7 = -2*(3*z0 - 4*z1 +z2 - 6*z3 + 8*z4 -2*z5 +3*z6 -4*z7 + z8);
        double c8 = 4*(z0 -2*z1 + z2 - 2*z3 + 4*z4 -2*z5 + z6 - 2*z7 + z8);

        for (size_t l=0;l<5;l++){//TODO: Fix this sloppy code
            double *psum;
            switch(l){
                case 0:
                    psum = &X.sum1[ni][ai];
                    break;
                case 1:
                    psum = &X.sum2[ni][ai];
                    break;
                case 2:
                    psum = &X.sum3[ni][ai];
                    break;
                case 3:
                    psum = &X.sum4[ni][ai];
                    break;
                case 4:
                    psum = &X.sum5[ni][ai];
                    break;
                default:
                    psum = nullptr;
            }
            double &sum = *psum;
            sum+=c0*(triA.p0[l] + triB.p0[l] + triC.p0[l] + triD.p0[l]);
            sum+=c1*(triA.p1[l] + triB.p1[l] + triC.p1[l] + triD.p1[l]);
            sum+=c2*(triA.p2[l] + triB.p2[l] + triC.p2[l] + triD.p2[l]);
            sum+=c3*(triA.p3[l] + triB.p3[l] + triC.p3[l] + triD.p3[l]);
            sum+=c4*(triA.p4[l] + triB.p4[l] + triC.p4[l] + triD.p4[l]);
            sum+=c5*(triA.p5[l] + triB.p5[l] + triC.p5[l] + triD.p5[l]);
            sum+=c6*(triA.p6[l] + triB.p6[l] + triC.p6[l] + triD.p6[l]);
            sum+=c7*(triA.p7[l] + triB.p7[l] + triC.p7[l] + triD.p7[l]);
            sum+=c8*(triA.p8[l] + triB.p8[l] + triC.p8[l] + triD.p8[l]);
        }
    }

    inline void pushW(cdouble w, const size_t ni, const size_t ai){
        this->ni = ni;
        this->ai = ai;
        c1 = 1.0/(w + a0[ni][ai]);
        c2 = c1*c1;
        c3 = c2*c1;
        c4 = c3*c1;
        c5 = c4*c1;
    }

    inline cdouble evaluate(Element &X){
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

    void setupHalfTriangles(const cdouble a0, const double b1, const double c1, const bool upper, bool debug = false){
        array<double, 2> Q0, Q1, Q2, Q4, Q5, Q6, Q7;
        cdouble a;
        double theta = atan2(c1, b1);
        double R0 = cos(theta);
        double R1 = sin(theta);
        a = (R0 * b1 + R1 * c1) / a0;
        Triangle &tri1 = upper?triD:triB;
        Triangle &tri2 = upper?triC:triA;
        if (upper){
            //Triangles always clockwise.
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
        if (fabs(Q0[0] - Q1[0])<0.00001){
            tri1.QA = Q2;
            tri1.QB = Q0;
            tri1.QC = Q1;
            setupTriangle(tri1, R0, R1, a, a0);
            zeroTriangle(tri2);
            return;
        }
        if (fabs(Q0[0] - Q2[0])<0.00001){
            tri1.QA = Q1;
            tri1.QB = Q0;
            tri1.QC = Q2;
            setupTriangle(tri1, R0, R1, a, a0);
            zeroTriangle(tri2);
            return;
        }
        if (fabs(Q1[0] - Q2[0])<0.00001){
            tri1.QA = Q0;
            tri1.QB = Q1;
            tri1.QC = Q2;
            setupTriangle(tri1, R0, R1, a, a0);
            zeroTriangle(tri2);
            return;
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

        tri1.QA = Q4;
        tri1.QB = Q5;
        tri1.QC = Q6;
        tri2.QA = Q7;
        tri2.QB = Q5;
        tri2.QC = Q6;

        setupTriangle(tri1, R0, R1, a, a0);
        setupTriangle(tri2, R0, R1, a, a0);

    }

    void zeroTriangle(Triangle &t){
        t.p0 = t.p1 = t.p2 = t.p3 = t.p4 = t.p5 = t.p6 = t.p7 = t.p8 = 0.0;
    }

    void setupTriangle(Triangle &t, const double R0, const double R1, const cdouble a1, const cdouble a0){
        double y0 = t.QA[1];
        double x0 = t.QA[0];
        double x1 = t.QB[0] - x0;
        double mAB = (t.QA[1] - t.QB[1]) / (t.QA[0] - t.QB[0]);
        double mAC = (t.QA[1] - t.QC[1]) / (t.QA[0] - t.QC[0]);
        double m0 = (mAB>mAC)?mAC:mAB;
        double m1 = (mAB>mAC)?mAB:mAC;

        computeTriangleIntegral(m0, m1, x0, y0, x1, R0, R1, a1, t);
        cdouble a0i = (1.0)/a0;
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

    void computeTriangleIntegral(const double m0, const double m1, const double x0, const double y0,
                                  const double x1, const double R0, const double R1, const cdouble a1, Triangle& t) {

        //cdouble logfac = log(1. + (a1 * x1)/(1. + a1 * x0));

        //Integral is of form 1/(1 + a1*x0 + a1*x1)
        //Divide through by (1 + a1*x0)
        //Integral now of form 1/(1 + a2*x1)
        //Each triangle would need to contain R0, R1, m0, m1

        cdouble Inorm = 1.0/(1.0 + a1*x0);
        cdouble a2 = a1*Inorm;
        cdouble O0, O1, O2, O3, O4, O5;
        double narr[]{1./1., 1./2., 1./3., 1./4., 1./5., 1./5., 1./7., 1./8., 1./9., 1./10., 1./11., 1./12., 1./13., 1./14., 1./15., 1./16.};
        if ((x1*x1*(a2.real()*a2.real() + a2.imag()*a2.imag())) < 0.1){
            //Logs are expanded for numerical precision purposes
            cdouble logfacsum{0.0, 0.0};
            cdouble powarr[16];
            powarr[0] = -x1*a2;
            for (size_t i=1;i<16;i++){
                powarr[i] = powarr[i-1]*(-x1*a2);
            }
            for (size_t i = 16; i > 5; i--) {
                logfacsum -= powarr[i-1]*narr[i-1];
            }
            O5 = logfacsum;
            O4 = O5 - powarr[4]*narr[4];
            O3 = O4 - powarr[3]*narr[3];
            O2 = O3 - powarr[2]*narr[2];
            O1 = O2 - powarr[1]*narr[1];
            O0 = O1 - powarr[0]*narr[0];
        }else{
            //(1 + ax) * (1 + ax) + b*b*x*x
            //1 + 2*a*x + a*a*x*x
            O0 = 0.5*log1p(x1*(2.*a2.real() + x1*(a2.real()*a2.real() + a2.imag()*a2.imag()))) + I*atan2(x1*a2.imag(), 1.0+x1*a2.real());
            cdouble powarr[5];
            powarr[0] = -x1*a2;
            for (size_t i=1;i<5;i++){
                powarr[i] = powarr[i-1]*(-x1*a2);
            }
            O1 = O0 + powarr[0]*narr[0];
            O2 = O1 + powarr[1]*narr[1];
            O3 = O2 + powarr[2]*narr[2];
            O4 = O3 + powarr[3]*narr[3];
            O5 = O4 + powarr[4]*narr[4];
        }

        cdouble oa2 = 1.0/a2;
        cdouble oa2_2 = Inorm*oa2*oa2;
        cdouble oa2_3 = oa2_2*oa2;
        cdouble oa2_4 = oa2_3*oa2;
        cdouble oa2_5 = oa2_4*oa2;
        cdouble oa2_6 = oa2_5*oa2;

        O5 *= oa2_6;
        O4 *= oa2_5;
        O3 *= oa2_4;
        O2 *= oa2_3;
        O1 *= oa2_2;

        double x0_2 = x0 * x0;
        double x0_3 = x0 * x0_2;
        double x0_4 = x0 * x0_3;
        double y0_2 = y0 * y0;
        double y0_3 = y0 * y0_2;
        double y0_4 = y0 * y0_3;
        double m0_2 = m0*m0;
        double m0_3 = m0_2*m0;
        double m0_4 = m0_3*m0;
        double m0_5 = m0_4*m0;
        double m1_2 = m1*m1;
        double m1_3 = m1_2*m1;
        double m1_4 = m1_3*m1;
        double m1_5 = m1_4*m1;
        double R0_2 = R0*R0;
        double R0_3 = R0_2*R0;
        double R0_4 = R0_3*R0;
        double R1_2 = R1*R1;
        double R1_3 = R1_2*R1;
        double R1_4 = R1_3*R1;

        cdouble Ix4y0 = ((m0 - m1) * O5);
        cdouble Ix3y1 = (0.5*(m0_2 - m1_2) * O5);
        cdouble Ix3y0 = ((-m0 + m1) * O4);
        cdouble Ix2y2 = ((m0_3 - m1_3) * O5)*(1./3.);
        cdouble Ix2y1 = (-0.5*(m0_2 - m1_2) * O4);
        cdouble Ix2y0 = ((m0 - m1) * O3);
        cdouble Ix1y3 = (0.25*(m0_4 - m1_4) * O5);
        cdouble Ix1y2 = ((1.0/3.0) * (-m0_3 + m1_3) * O4);
        cdouble Ix1y1 = (0.5*(m0_2 - m1_2) * O3);
        cdouble Ix1y0 = ((-m0 + m1) * O2);
        cdouble Ix0y4 = (0.2*(m0_5 - m1_5) * O5);
        cdouble Ix0y3 = (0.25*(-m0_4 + m1_4) * O4);
        cdouble Ix0y2 = ((1.0/3.0)*(m0_3 - m1_3) * O3);
        cdouble Ix0y1 = (0.5*(-m0_2 + m1_2) * O2);
        cdouble Ix0y0 = ((m0 - m1) * O1);

        //Work out taylor expansions for Ix0y0, ... for omega

        //60 variables --> precomputation possible but using too much memory
        t.p0 = Ix0y0;
        t.p1 = R0*Ix1y0
               - R1*Ix0y1
               + (R0*x0 - R1*y0)*Ix0y0;
        t.p2 = R0*R0*Ix2y0
               - 2*R0*R1*Ix1y1
               + (2.*R0_2*x0 - 2.*R0*R1*y0)*Ix1y0
               + R1_2*Ix0y2
               + (-2 * R0 * R1 * x0 + 2.*R1*R1 * y0)*Ix0y1
               + (R0_2*x0*x0 - 2.*R0*R1*x0*y0 + R1_2*y0*y0)*Ix0y0;
        t.p3 = R1*Ix1y0
               + R0*Ix0y1
               + (R1*x0 + R0*y0)*Ix0y0;
        t.p4 = R0*R1*Ix2y0
               + (R0_2 - R1_2)*Ix1y1
               + (2.*R0*R1*x0 + R0_2*y0 - R1_2*y0)*Ix1y0
               - R0*R1*Ix0y2
               + (R0_2*x0 - R1_2*x0 - 2.*R0*R1*y0)*Ix0y1
               + (R0*R1*x0_2 + R0_2*x0*y0 - R1_2*x0*y0 - R0*R1*y0_2)*Ix0y0;
        t.p5 = R0_2*R1*Ix3y0
               + (R0_3 - 2*R0*R1_2)*Ix2y1
               + (3 * R0_2 * R1 * x0 + R0_3 * y0 - 2 * R0 * R1_2 * y0)*Ix2y0
               + (-2 * R0_2 * R1 + R1_3)*Ix1y2
               + (2 * R0_3 * x0 - 4 * R0 * R1_2 * x0 - 4 * R0_2 * R1 * y0 + 2 * R1_3 * y0)*Ix1y1
               + (3 * R0_2 * R1 * x0_2 + 2 * R0_3 * x0 * y0 - 4 * R0 * R1_2 * x0 * y0 - 2 * R0_2 * R1 * y0_2 + R1_3 * y0_2) * Ix1y0
               + (R0 * R1_2) * Ix0y3
               + (-2 * R0_2 * R1 * x0 + R1_3 * x0 + 3 * R0 * R1_2 * y0)*Ix0y2
               + (R0_3 * x0_2 - 2 * R0 * R1_2 * x0_2 - 4 * R0_2 * R1 * x0 * y0 + 2 * R1_3 * x0 * y0 + 3 * R0 * R1_2 * y0_2) * Ix0y1
               + (R0_2 * R1 * x0_3 + R0_3 * x0_2 * y0 - 2 * R0 * R1_2 * x0_2 * y0 - 2 * R0_2 * R1 * x0 * y0_2 + R1_3 * x0 * y0_2 + R0 * R1_2 * y0_3)*Ix0y0;
        t.p6 = R1_2 * Ix2y0
               + (2 * R0 * R1)*Ix1y1
               + (2 * R1_2 * x0 + 2 * R0 * R1 * y0) * Ix1y0
               + R0_2*Ix0y2
               + (2 * R0 * R1 * x0 + 2 * R0_2 * y0)*Ix0y1
               + (R1_2 * x0_2 + 2 * R0 * R1 * x0 * y0 + R0_2 * y0_2)*Ix0y0;
        t.p7 = (R0 * R1_2) * Ix3y0
               + (2 * R0_2 * R1 - R1_3)*Ix2y1
               + (3 * R0 * R1_2 * x0 + 2 * R0_2 * R1 * y0 - R1_3 * y0)*Ix2y0
               + (R0_3 - 2 * R0 * R1_2)*Ix1y2
               + (4 * R0_2 * R1 * x0 - 2 * R1_3 * x0 + 2 * R0_3 * y0 - 4 * R0 * R1_2 * y0) * Ix1y1
               + (3 * R0 * R1_2 * x0_2 + 4 * R0_2 * R1 * x0 * y0 - 2 * R1_3 * x0 * y0 + R0_3 * y0_2 - 2 * R0 * R1_2 * y0_2) * Ix1y0
               + (-R0_2 * R1)*Ix0y3
               + (R0_3 * x0 - 2 * R0 * R1_2 * x0 - 3 * R0_2 * R1 * y0)*Ix0y2
               + (2 * R0_2 * R1 * x0_2 - R1_3 * x0_2 + 2 * R0_3 * x0 * y0 - 4 * R0 * R1_2 * x0 * y0 - 3 * R0_2 * R1 * y0_2)*Ix0y1
               + (R0 * R1_2 * x0_3 + 2 * R0_2 * R1 * x0_2 * y0 - R1_3 * x0_2 * y0 + R0_3 * x0 * y0_2 - 2 * R0 * R1_2 * x0 * y0_2 - R0_2 * R1 * y0_3)*Ix0y0;
        t.p8 = (R0_2 * R1_2)*Ix4y0
               + (2 * R0_3 * R1 - 2 * R0 * R1_3)*Ix3y1
               + (4 * R0_2 * R1_2 * x0 + 2 * R0_3 * R1 * y0 - 2 * R0 * R1_3 * y0)*Ix3y0
               + (R0_4 - 4 * R0_2 * R1_2 + R1_4)*Ix2y2
               + (6 * R0_3 * R1 * x0 - 6 * R0 * R1_3 * x0 + 2 * R0_4 * y0 - 8 * R0_2 * R1_2 * y0 + 2 * R1_4 * y0)*Ix2y1
               + (6 * R0_2 * R1_2 * x0_2 + 6 * R0_3 * R1 * x0 * y0 - 6 * R0 * R1_3 * x0 * y0 + R0_4 * y0_2 - 4 * R0_2 * R1_2 * y0_2 + R1_4 * y0_2)*Ix2y0
               + (-2 * R0_3 * R1 + 2 * R0 * R1_3)*Ix1y3
               + (2 * R0_4 * x0 - 8 * R0_2 * R1_2 * x0 + 2 * R1_4 * x0 - 6 * R0_3 * R1 * y0 + 6 * R0 * R1_3 * y0)*Ix1y2
               + (6 * R0_3 * R1 * x0_2 - 6 * R0 * R1_3 * x0_2 + 4 * R0_4 * x0 * y0 - 16 * R0_2 * R1_2 * x0 * y0 + 4 * R1_4 * x0 * y0 - 6 * R0_3 * R1 * y0_2 + 6 * R0 * R1_3 * y0_2)*Ix1y1
               + (4 * R0_2 * R1_2 * x0_3 + 6 * R0_3 * R1 * x0_2 * y0 - 6 * R0 * R1_3 * x0_2 * y0 + 2 * R0_4 * x0 * y0_2 - 8 * R0_2 * R1_2 * x0 * y0_2 + 2 * R1_4 * x0 * y0_2 - 2 * R0_3 * R1 * y0_3 + 2 * R0 * R1_3 * y0_3)*Ix1y0
               + (R0_2 * R1_2)*Ix0y4
               + (-2 * R0_3 * R1 * x0 + 2 * R0 * R1_3 * x0 + 4 * R0_2 * R1_2 * y0)*Ix0y3
               + (R0_4 * x0_2 - 4 * R0_2 * R1_2 * x0_2 + R1_4 * x0_2 - 6 * R0_3 * R1 * x0 * y0 + 6 * R0 * R1_3 * x0 * y0 + 6 * R0_2 * R1_2 * y0_2)*Ix0y2
               + (2 * R0_3 * R1 * x0_3 - 2 * R0 * R1_3 * x0_3 + 2 * R0_4 * x0_2 * y0 - 8 * R0_2 * R1_2 * x0_2 * y0 + 2 * R1_4 * x0_2 * y0 - 6 * R0_3 * R1 * x0 * y0_2 + 6 * R0 * R1_3 * x0 * y0_2 + 4 * R0_2 * R1_2 * y0_3)*Ix0y1
               + (R0_2 * R1_2 * x0_4 + 2 * R0_3 * R1 * x0_3 * y0 - 2 * R0 * R1_3 * x0_3 * y0 + R0_4 * x0_2 * y0_2 - 4 * R0_2 * R1_2 * x0_2 * y0_2 + R1_4 * x0_2 * y0_2 - 2 * R0_3 * R1 * x0 * y0_3 + 2 * R0 * R1_3 * x0 * y0_3 + R0_2 * R1_2 * y0_4)*Ix0y0;
    }



};


class RelativisticSpecies {

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
    arr2d igamma, vpara_h, U0, U1, U2, U3, W0, W1;

    arr1d z;
    arr2d jns, jnds;
    arr1d das, a_mins;
    arr2d damaxs;
    arr2d kp0, kp1, kp2, kp3, kp4, kp5;
    TaylorSum2 series;

    RelativisticSpecies(const boost::python::object &species, double B) {
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
        igamma = arr2d(nperp_h, arr1d(npara_h, 0.0));

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
                igamma[i][j] = 1.0/sqrt(1.0 + pow(pperp_h[i]/(cl*mass), 2) + pow(ppara_h[j]/(cl*mass), 2));
                vpara_h[i][j] = ppara_h[j] * igamma[i][j] * (1.0/mass);
                df_dpperp_h[i][j]/=sum;
                df_dppara_h[i][j]/=sum;
                U0[i][j] = df_dpperp_h[i][j] * igamma[i][j];
                U1[i][j] = ppara_h[j]*df_dpperp_h[i][j] * igamma[i][j];
                U2[i][j] = (pperp_h[i]*df_dppara_h[i][j] - ppara_h[j]*df_dpperp_h[i][j])*igamma[i][j]*igamma[i][j]*(1.0/mass);
                U3[i][j] = ppara_h[j]*(pperp_h[i]*df_dppara_h[i][j] - ppara_h[j]*df_dpperp_h[i][j])*igamma[i][j]*igamma[i][j]*(1.0/mass);
                W0[i][j] = ppara_h[j]*pperp_h[i]*df_dppara_h[i][j]*igamma[i][j];
                W1[i][j] = wc0*ppara_h[j]*(ppara_h[j]*df_dpperp_h[i][j] - pperp_h[i]*df_dppara_h[i][j])*(igamma[i][j]*igamma[i][j]);
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
            z[i] = kperp*pperp_h[i]/(wc0*mass);
        }
        for (size_t k=0;k<ns.size();k++){
            int n = ns[k];
            for (size_t i=0;i<nperp_h;i++) {
                jns[k][i] = bm::cyl_bessel_j(n, z[i]);
                jnds[k][i] = bm::cyl_bessel_j_prime(n, z[i]);
                kp0[k][i] = n*n*jns[k][i]*jns[k][i]*(wc0*mass*wc0*mass/(kperp*kperp));
                kp1[k][i] = n*jns[k][i]*jnds[k][i]*pperp_h[i]*(wc0*mass/(kperp));
                kp2[k][i] = n*jns[k][i]*jns[k][i]*(wc0*mass/(kperp));
                kp3[k][i] = jnds[k][i]*jnds[k][i]*pperp_h[i]*pperp_h[i];
                kp4[k][i] = jns[k][i]*jnds[k][i]*pperp_h[i];
                kp5[k][i] = jns[k][i]*jns[k][i];
            }
        }
    }

    void push_kpara(const double kpara){
        series = TaylorSum2(n_taylor+1, ns);
        das = arr1d(ns.size(), 0.0);
        a_mins = arr1d(ns.size(), 0.0);
        damaxs = arr2d(ns.size(), arr1d(n_taylor+1, 0.0));
        for (size_t k=0;k<ns.size();k++) {
            const int n = ns[k];
            double a_min{DBL_MAX}, a_max{DBL_MIN}, da;
            for (size_t i = 0; i < nperp_g + 1; i++) {
                for (size_t j = 0; j < npara_g + 1; j++) {
                    double a = -kpara*vpara_h[i*2][j*2] - n*wc0*igamma[i*2][j*2];
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
                    double a, a0, z0, z1, z2, z3, damax;
                    size_t ai;
                    z0 = -kpara*vpara_h[i*2][j*2] - n*wc0*igamma[i*2][j*2];
                    z1 = -kpara*vpara_h[i*2][j*2+2] - n*wc0*igamma[i*2][j*2+2];
                    z2 = -kpara*vpara_h[i*2+2][j*2] - n*wc0*igamma[i*2+2][j*2];
                    z3 = -kpara*vpara_h[i*2+2][j*2+2] - n*wc0*igamma[i*2+2][j*2+2];
                    a = 0.25*(z0+z1+z2+z3);
                    ai = (size_t)((a - a_min)/da + 0.5);
                    a0 = ai*da + a_min;
                    damax = max(max(fabs(z0 - a0), fabs(z1 - a0)), max(fabs(z2 - a0), fabs(z3 - a0)));
                    damaxs[k][ai] = max(damaxs[k][ai], damax);
                    //a1 = a0 - z0;
                    //b = z1 - z0;
                    //c = z2 - z0;
                    //d = z3 - z0 - b - c;
                    //series.pushDenominator(k, i, j, ai, a1, b, c, d);
                    series.pushDenominator(k, i, j, ai, z0, z1, z2, z3);
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
                    //cout<<series.mapping[k].size()<<endl;
                    for (pair<size_t, size_t> p:series.mapping[k][ai]){
                        double z0, z1, z2, z3;
                        size_t i{p.first}, j{p.second};
                        z0 = wr -kpara*vpara_h[i*2][j*2] - n*wc0*igamma[i*2][j*2];
                        z1 = wr -kpara*vpara_h[i*2][j*2+2] - n*wc0*igamma[i*2][j*2+2];
                        z2 = wr -kpara*vpara_h[i*2+2][j*2] - n*wc0*igamma[i*2+2][j*2];
                        z3 = wr -kpara*vpara_h[i*2+2][j*2+2] - n*wc0*igamma[i*2+2][j*2+2];

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
        X[1][2] = 0.0*-I*X12*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][0] = X02*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][1] = 0.0*I*X12*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        X[2][2] = X22*2.0*M_PI*wp0*wp0*iw*dppara*dpperp*4.0;
        //cout<<1.0 + X[0][0]<<", "<<X[0][1]<<", "<<1.0 + X[1][1]<<", "<<1.0 + X[2][2]<<endl;
        //cout<<X[0][0]<<", "<<X[0][1]<<", "<<X[1][1]<<", "<<X[2][2]<<endl;
        return X;
    }
};

class RelativisticSolver {

    double kperp, kpara;

    vector<RelativisticSpecies> species;

public:

    RelativisticSolver(const p::list &list, const double B) {
        Py_Initialize();
        np::initialize();
        for (int i = 0; i < len(list); i++) {
            species.push_back(RelativisticSpecies(list[i], B));
        }
    }

    void push_kperp(const double kperp) {
        this->kperp = kperp;
        for (RelativisticSpecies &spec: species) {
            spec.push_kperp(kperp);
        }
    }

    void push_kpara(const double kpara) {
        this->kpara = kpara;
        for (RelativisticSpecies &spec: species) {
            spec.push_kpara(kpara);
        }
    }

    cdouble evaluate(const double wr, const double wi) {
        array<array<cdouble, 3>, 3> X;
        for (RelativisticSpecies &spec: species) {
            X+=spec.push_omega(kpara, wr, wi);
        }
        X[0][0] += 1.0;
        X[1][1] += 1.0;
        X[2][2] += 1.0;
        cdouble w = wr + I*wi;
        cdouble nx = cl*kperp/w;
        cdouble nz = cl*kpara/w;

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

        return detM*pow(wr, 4) / pow((kpara * kpara + kperp * kperp) * cl * cl, 2);
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





BOOST_PYTHON_MODULE (libRelativisticSolver) {
    p::class_<RelativisticSolver>("RelativisticSolver", p::init<const boost::python::list, double>())
            .def("push_kperp", &RelativisticSolver::push_kperp)
            .def("push_kpara", &RelativisticSolver::push_kpara)
            .def("evaluate", &RelativisticSolver::evaluate)
            .def("marginalize", &RelativisticSolver::marginalize);

}
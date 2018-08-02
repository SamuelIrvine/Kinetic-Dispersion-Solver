//
// Created by h on 20/07/18.
//

#ifndef KINETIC_DISPERSION_SOLVER_RELATIVISTICSPECIES_H
#define KINETIC_DISPERSION_SOLVER_RELATIVISTICSPECIES_H

#include "common.h"
#include "arrayutils.h"

struct Triangle{
    double p0[5], p1[5], p2[5], p3[5], p4[5], p5[5], p6[5], p7[5], p8[5];
    array<double, 2> QA, QB, QC;
    double R0, R1, y0, x0, x1, m0, m1, mx;
    bool active;
};

struct Element{
    arr2d sum1, sum2, sum3, sum4, sum5;

    Element();

    Element(size_t n_s, size_t n_a);
};

class TaylorSum {

public:

    Element sX00a, sX01a, sX02a, sX11a, sX12a, sX22a;
    Element sX00b, sX01b, sX02b, sX11b, sX12b, sX22b;

    vector<vector<vector<pair<size_t, size_t>>>> mapping;

    TaylorSum();
    TaylorSum(size_t n_a, vector<int> &ns);
    void setA(const size_t ni, const double a_min, const double da);
    void pushDenominator(const size_t ni, const size_t i, const size_t j, const size_t ai, const double z0, const double z1,
                         const double z2, const double z3);
    void accumulate(Element &X, const arr2d &p, const arr1d &q, const double r);
    void pushW(cdouble w, const size_t ni, const size_t ai);
    cdouble evaluate(Element &X);


private:

    arr2d a0;
    size_t i, j, ni, ai;
    cdouble c1, c2, c3, c4, c5;
    Triangle triA, triB, triC, triD;

    void setupHalfTriangles(const double mx, const double my, const bool upper);
};

class RelativisticSpecies {

public:

    RelativisticSpecies(const boost::python::object &species, double B);

    void push_kperp(const double kperp);

    void push_kpara(const double kpara);

    array<array<cdouble, 3>, 3> push_omega(const double kpara, const double wr, const double wi);

private:

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
    TaylorSum series;
};







#endif //KINETIC_DISPERSION_SOLVER_RELATIVISTICSPECIES_H

//
// Created by h on 20/07/18.
//

#ifndef KINETIC_DISPERSION_SOLVER_SPECIES_H
#define KINETIC_DISPERSION_SOLVER_SPECIES_H

#include "common.h"
#include "arrayutils.h"

class Species {

public:

    Species(const p::object &species, double B);

    void push_kperp(const double kperp);

    array<array<cdouble, 3>, 3> push_omega(const double kpara, const double kperp, const double wr, const double wi);

private:

    arr2d df_dvpara_h, df_dvperp_h;
    arr1d dvperp_h;
    arr1d vpara_h, vperp_h;
    arr1d vpara_hm, vpara_hp, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8;
    std::vector<int> ns;
    std::vector<arr1d> jns, jnps;
    std::vector<array<arr1d, 6>> qs, q_ms, q_cs;
    double charge, mass, density;
    double wp, wc;
    size_t npara, nperp;

};

#endif //KINETIC_DISPERSION_SOLVER_SPECIES_H

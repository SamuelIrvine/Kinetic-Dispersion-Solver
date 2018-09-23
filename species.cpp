//
// Created by h on 20/07/18.
//

#include "species.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>

Species::Species(const boost::python::object &species, double B) {
    charge = p::extract<double>(species.attr("charge"));
    mass = p::extract<double>(species.attr("mass"));
    density = p::extract<double>(species.attr("density"));
    wc = B * abs(charge) / mass;
    wp = pow(density * charge * charge / (e0 * mass), 0.5);

    vpara_h = arrayutils::extract1d(species.attr("vpara_h"));
    vperp_h = arrayutils::extract1d(species.attr("vperp_h"));

    npara = vpara_h.size();
    nperp = vperp_h.size();

    df_dvpara_h = arrayutils::extract2d(species.attr("df_dvpara_h"), npara, nperp);
    df_dvperp_h = arrayutils::extract2d(species.attr("df_dvperp_h"), npara, nperp);

    p::list pyns = p::extract<p::list>(species.attr("ns"));
    for (int i = 0; i < len(pyns); ++i) {
        int n{p::extract<int>(pyns[i])};
        ns.push_back(n);
        jns.push_back(arr1d(nperp));
        jnps.push_back(arr1d(nperp));
        qs.push_back({arr1d(npara), arr1d(npara), arr1d(npara), arr1d(npara), arr1d(npara), arr1d(npara)});
        q_ms.push_back({arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1),
                        arr1d(npara - 1)});
        q_cs.push_back({arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1), arr1d(npara - 1),
                        arr1d(npara - 1)});
    }

    dvperp_h = arr1d(nperp - 1);
    vpara_hp = arr1d(npara - 1);
    vpara_hm = arr1d(npara - 1);
    dv1 = arr1d(npara - 1);
    dv2 = arr1d(npara - 1);
    dv3 = arr1d(npara - 1);
    dv4 = arr1d(npara - 1);
    dv5 = arr1d(npara - 1);
    dv6 = arr1d(npara - 1);
    dv7 = arr1d(npara - 1);
    dv8 = arr1d(npara - 1);

    for (size_t i = 0; i < nperp - 1; i++) {
        dvperp_h[i] = vperp_h[i + 1] - vperp_h[i];
    }

    for (size_t i = 0; i < npara - 1; i++) {
        vpara_hm[i] = vpara_h[i];
        vpara_hp[i] = vpara_h[i + 1];
        dv1[i] = pow(vpara_hp[i], 1) - pow(vpara_hm[i], 1);
        dv2[i] = pow(vpara_hp[i], 2) - pow(vpara_hm[i], 2);
        dv3[i] = pow(vpara_hp[i], 3) - pow(vpara_hm[i], 3);
        dv4[i] = pow(vpara_hp[i], 4) - pow(vpara_hm[i], 4);
        dv5[i] = pow(vpara_hp[i], 5) - pow(vpara_hm[i], 5);
        dv6[i] = pow(vpara_hp[i], 6) - pow(vpara_hm[i], 6);
        dv7[i] = pow(vpara_hp[i], 7) - pow(vpara_hm[i], 7);
        dv8[i] = pow(vpara_hp[i], 8) - pow(vpara_hm[i], 8);
    }

}

void Species::push_kperp(const double kperp) {
    for (size_t ni = 0; ni < ns.size(); ni++) {
        int n{ns[ni]};
        double jns0 = bm::cyl_bessel_j(n, -(kperp / wc) * vperp_h[0]);
        double jnps0 = bm::cyl_bessel_j(n, -(kperp / wc) * vperp_h[0]);
        for (size_t j = 0; j < nperp; j++) {
            jns[ni][j] = bm::cyl_bessel_j(n, (kperp / wc) * vperp_h[j]);
            jnps[ni][j] = bm::cyl_bessel_j_prime(n, (kperp / wc) * vperp_h[j]);
        }
        for (size_t i = 0; i < npara; i++) {
            double qh[6]{};
            qh[0] = 0.5 * jns0 * jns0 * df_dvperp_h[i][0] * dvperp_h[0];
            qh[1] = 0.5 * jns0 * jns0 * vperp_h[0] * df_dvpara_h[i][0] * dvperp_h[0];
            qh[2] = 0.5 * jns0 * jnps0 * vperp_h[0] * df_dvperp_h[i][0] * dvperp_h[0];
            qh[3] = 0.5 * jns0 * jnps0 * vperp_h[0] * vperp_h[0] * df_dvpara_h[i][0] * dvperp_h[0];
            qh[4] = 0.5 * jnps0 * jnps0 * vperp_h[0] * vperp_h[0] * df_dvperp_h[i][0] * dvperp_h[0];
            qh[5] = 0.5 * jnps0 * jnps0 * vperp_h[0] * vperp_h[0] * vperp_h[0] * df_dvpara_h[i][0] * dvperp_h[0];
            for (size_t j = 0; j < nperp - 1; j++) {
                qh[0] += jns[ni][j] * jns[ni][j] * df_dvperp_h[i][j] * dvperp_h[j];
                qh[1] += jns[ni][j] * jns[ni][j] * vperp_h[j] * df_dvpara_h[i][j] * dvperp_h[j];
                qh[2] += jns[ni][j] * jnps[ni][j] * vperp_h[j] * df_dvperp_h[i][j] * dvperp_h[j];
                qh[3] += jns[ni][j] * jnps[ni][j] * vperp_h[j] * vperp_h[j] * df_dvpara_h[i][j] * dvperp_h[j];
                qh[4] += jnps[ni][j] * jnps[ni][j] * vperp_h[j] * vperp_h[j] * df_dvperp_h[i][j] * dvperp_h[j];
                qh[5] += jnps[ni][j] * jnps[ni][j] * vperp_h[j] * vperp_h[j] * vperp_h[j] * df_dvpara_h[i][j] *
                         dvperp_h[j];
            }
            for (size_t k = 0; k < 6; k++) {
                qs[ni][k][i] = qh[k];
            }
        }
        for (size_t i = 0; i < npara - 1; i++) {
            for (size_t k = 0; k < 6; k++) {
                q_ms[ni][k][i] = (qs[ni][k][i + 1] - qs[ni][k][i]) / dv1[i];
                q_cs[ni][k][i] = qs[ni][k][i] - q_ms[ni][k][i] * vpara_hm[i];
            }
        }
    }
}

array<array<cdouble, 3>, 3> Species::push_omega(const double kpara, const double kperp, const double wr, const double wi) {
    array<array<cdouble, 3>, 3> XP{};
    for (size_t ni = 0; ni < ns.size(); ni++) {
        int n{ns[ni]};
        double a{wr};
        double b{wi};
        double b2{b * b};
        double d{n * wc};
        cdouble apb{a, b};
        cdouble apbmd{a - d, b};
        cdouble apbmd2 = apbmd*apbmd;
        cdouble apbmd3 = apbmd2*apbmd;
        cdouble iapb{1.0 / apb};
        cdouble iapbmd{1.0 / apbmd};

        array<cdouble, 6> s_L0_q_m{};
        array<cdouble, 6> s_L0_q_c{};
        array<cdouble, 6> s_L1_q_m{};
        array<cdouble, 6> s_L1_q_c{};
        array<cdouble, 6> s_L2_q_m{};
        array<cdouble, 6> s_L2_q_c{};
        array<cdouble, 6> s_L3_q_m{};

        for (size_t i = 0; i < npara - 1; i++) {
            //cdouble clogfac{log((apb - d - vpara_hp[i]*kpara)/(apb - d - vpara_hm[i]*kpara))};
            //cdouble clogfac{log((apb - d - vpara_hm[i]*kpara - dv1[i]*kpara)/(apb - d - vpara_hm[i]*kpara))};
            //cdouble clogfac{log(1.0 - (dv1[i]*kpara)/(apb - d - vpara_hm[i]*kpara))};

            //Wish to taylor expand in kpara.
            //cdouble cx = (-dv1[i]*kpara)/(apbmd - vpara_hm[i]*kpara);
            cdouble cx = kpara*iapbmd;
            cdouble L0, L1, L2, L3;
            const double narr[]{1./1., 1./2., 1./3., 1./4., 1./5., 1./5., 1./7., 1./8.};
            if ((vpara_hm[i]+vpara_hp[i])*(vpara_hm[i]+vpara_hp[i])*(cx.real()*cx.real() + cx.imag()*cx.imag()) < -0.000001){
                cdouble logfacsum{0.0, 0.0};
                cdouble powarr[8];
                const cdouble dvarr[]{dv1[i], dv2[i], dv3[i], dv4[i], dv5[i], dv6[i],dv7[i], dv8[i]};
                powarr[0] = cx;
                for (size_t i = 1; i < 8;i++){
                    powarr[i] = powarr[i-1]*(cx);
                }
                for (size_t i = 8; i > 3; i--) {
                    logfacsum -= powarr[i-1]*(narr[i-1]*dvarr[i-1]);
                }
                L3 = logfacsum;
                L2 = L3 - powarr[2]*narr[2]*dvarr[2];
                L1 = L2 - powarr[1]*narr[1]*dvarr[1];
                L0 = L1 - powarr[0]*narr[0]*dvarr[0];
//                if (wi<0.0){
//                    cdouble di = I*L0.imag();
//                    L0 -= di;
//                    L1 -= di;
//                    L2 -= di;
//                    L3 -= di;
//                }
            }else{
                double advkm = a - d - vpara_hm[i] * kpara;
                double rlogfac = 0.5 * log1p(dv1[i] * kpara * (dv1[i] * kpara - 2.0 * advkm) / (advkm * advkm + b2));
                //double ilogfac = atan2(abs(b)*dv1[i]*kpara, (advkm + dv1[i]*kpara)*advkm + b2);
                double ilogfac = atan2(b*dv1[i]*kpara, (advkm + dv1[i]*kpara)*advkm + b2);
                L0 = cdouble(rlogfac, ilogfac);
                cdouble powarr[3];
                const cdouble dvarr[]{dv1[i], dv2[i], dv3[i]};
                powarr[0] = cx;
                for (size_t i = 1; i < 3;i++){
                    powarr[i] = powarr[i-1]*(cx);
                }
                L1 = L0 + powarr[0]*narr[0]*dvarr[0];
                L2 = L1 + powarr[1]*narr[1]*dvarr[1];
                L3 = L2 + powarr[2]*narr[2]*dvarr[2];
                if (b<0.0) {
                    L0 = L0 - 2 * L0.imag();
                    L1 = L1 - 2 * L1.imag();
                }

            }
            //if (abs(apbmd.real()) < abs(dv1[i]*kpara)){
                //analytic continuation
                //add 2*I*pi*F
            //}


            for (size_t k = 0; k < 6; k++) {
                s_L0_q_m[k] += L0 * q_ms[ni][k][i];
                s_L0_q_c[k] += L0 * q_cs[ni][k][i];
                s_L1_q_m[k] += L1 * q_ms[ni][k][i];
            }

            for (size_t k = 0; k < 5; k++) {
                s_L1_q_c[k] += L1 * q_cs[ni][k][i];
                s_L2_q_m[k] += L2 * q_ms[ni][k][i];
            }

            s_L2_q_c[0] += L2 * q_cs[ni][0][i];
            s_L2_q_c[2] += L2 * q_cs[ni][2][i];
            s_L3_q_m[0] += L3 * q_ms[ni][0][i];
            s_L3_q_m[2] += L3 * q_ms[ni][2][i];
        }

        double kpara2 = kpara*kpara;
        double kpara3 = kpara2*kpara;
        double kpara4 = kpara3*kpara;
        double kperp2 = kperp*kperp;

        cdouble c00 = M_PI * 2.0 * wp * wp * wc * wc * n * n * iapb * iapb;
        XP[0][0] -= c00*apb*(kpara2*apbmd*s_L1_q_m[0] + kpara3*s_L0_q_c[0]);
        XP[0][0] -= c00*(kpara3*apbmd*s_L1_q_m[1] + kpara4*s_L0_q_c[1]);
        XP[0][0] += c00*(kpara2*apbmd2*s_L2_q_m[0] + kpara3*apbmd*s_L1_q_c[0]);

        cdouble c01 = M_PI * 2.0 * wp * wp * n * wc * I * iapb * iapb * kperp;
        XP[0][1] -= c01*apb*(kpara2*apbmd*s_L1_q_m[2] + kpara3*s_L0_q_c[2]);
        XP[0][1] -= c01*(kpara3*apbmd*s_L1_q_m[3] + kpara4*s_L0_q_c[3]);
        XP[0][1] += c01*(kpara2*apbmd2*s_L2_q_m[2] + kpara3*apbmd*s_L1_q_c[2]);

        cdouble c02 = M_PI * 2.0 * wp * wp * n * wc * iapb * iapb * kperp;
        XP[0][2] -= c02*apb*(kpara*apbmd2*s_L2_q_m[0] + kpara2*apbmd*s_L1_q_c[0]);
        XP[0][2] -= c02*(kpara2*apbmd2*s_L2_q_m[1] + kpara3*apbmd*s_L1_q_c[1]);
        XP[0][2] += c02*(kpara*apbmd3*s_L3_q_m[0] + kpara2*apbmd2*s_L2_q_c[0]);

        cdouble c11 = M_PI * 2.0 * wp * wp * iapb * iapb * kperp2;
        XP[1][1] -= c11*apb*(kpara2*apbmd*s_L1_q_m[4] + kpara3*s_L0_q_c[4]);
        XP[1][1] -= c11*(kpara3*apbmd*s_L1_q_m[5] + kpara4*s_L0_q_c[5]);
        XP[1][1] += c11*(kpara2*apbmd2*s_L2_q_m[4] + kpara3*apbmd*s_L1_q_c[4]);

        cdouble c12 = M_PI * 2.0 * wp * wp * -I * iapb * iapb * kperp2;
        XP[1][2] -= c12*apb*(kpara*apbmd2*s_L2_q_m[2] + kpara2*apbmd*s_L1_q_c[2]);
        XP[1][2] -= c12*(kpara2*apbmd2*s_L2_q_m[3] + kpara3*apbmd*s_L1_q_c[3]);
        XP[1][2] += c12*(kpara*apbmd3*s_L3_q_m[2] + kpara2*apbmd2*s_L2_q_c[2]);

        cdouble c22 = M_PI * 2.0 * wp * wp * iapb * iapb * kperp2;
        XP[2][2] -= c22*apbmd*(kpara*apbmd2*s_L2_q_m[1] + kpara2*apbmd*s_L1_q_c[1]);
        XP[2][2] -= c22*d*(apbmd3*s_L2_q_m[0] + kpara*apbmd2*s_L2_q_c[0]);
    }

    XP[1][0] = -XP[0][1];
    XP[2][0] = XP[0][2];
    XP[2][1] = -XP[1][2];

    return XP;
}

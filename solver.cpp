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
//const double mu0 = cl*cl*e0;

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


template<size_t M, size_t N, typename T, typename X, typename Y>
static T polyDet(const array<array<array<array<T, N>, M>, 3>, 3> &p, const X &x, const Y &y) {
    //array<array<T, N*3-2>, M*3-2> sum{};
    T sum{};
    for (size_t i = 0; i < 3; i++) {
        //sum+=polyMul(polyMul(p[1][(i+1)%3], p[2][(i+2)%3]), p[0][i]);
        //sum-=polyMul(polyMul(p[1][(i+2)%3], p[2][(i+1)%3]), p[0][i]);

        sum += polyEval(p[1][(i + 1) % 3], x, y) * polyEval(p[2][(i + 2) % 3], x, y) * polyEval(p[0][i], x, y);
        sum -= polyEval(p[1][(i + 2) % 3], x, y) * polyEval(p[2][(i + 1) % 3], x, y) * polyEval(p[0][i], x, y);
    }
    return sum;
    //return polyEval(sum, x, y);
};

class Species {

public://TODO: private later

    arr2d df_dvpara_h, df_dvperp_h;
    arr1d dvperp_h;
    arr1d vpara_h, vperp_h;
    arr1d vpara_hm, vpara_hp, dv1, dv2, dv3;
    vector<int> ns;
    vector<arr1d> jns, jnps;
    vector<array<arr1d, 6>> qs, q_ms, q_cs;
    vector<array<double, 6>> s_dv1_q_ms, s_dv2_q_ms, s_dv3_q_ms;
    vector<array<double, 6>> s_dv1_q_cs, s_dv2_q_cs;
    double charge, mass, density;
    double wp, wc;
    size_t npara, nperp;

    Species(const boost::python::object &species, double B) {
        charge = p::extract<double>(species.attr("charge"));
        mass = p::extract<double>(species.attr("mass"));
        density = p::extract<double>(species.attr("density"));
        wc = B * abs(charge) / mass;
        wp = pow(density * charge * charge / (e0 * mass), 0.5);

        vpara_h = extract1d(species.attr("vpara_h"));
        vperp_h = extract1d(species.attr("vperp_h"));

        npara = vpara_h.size();
        nperp = vperp_h.size();

        df_dvpara_h = extract2d(species.attr("df_dvpara_h"), npara, nperp);
        df_dvperp_h = extract2d(species.attr("df_dvperp_h"), npara, nperp);

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
            s_dv1_q_ms.push_back({});
            s_dv2_q_ms.push_back({});
            s_dv3_q_ms.push_back({});
            s_dv1_q_cs.push_back({});
            s_dv2_q_cs.push_back({});
        }

        dvperp_h = arr1d(nperp - 1);
        vpara_hp = arr1d(npara - 1);
        vpara_hm = arr1d(npara - 1);
        dv1 = arr1d(npara - 1);
        dv2 = arr1d(npara - 1);
        dv3 = arr1d(npara - 1);

        for (size_t i = 0; i < nperp - 1; i++) {
            dvperp_h[i] = vperp_h[i + 1] - vperp_h[i];
        }

        for (size_t i = 0; i < npara - 1; i++) {
            vpara_hm[i] = vpara_h[i];
            vpara_hp[i] = vpara_h[i + 1];
            dv1[i] = pow(vpara_hp[i], 1) - pow(vpara_hm[i], 1);
            dv2[i] = pow(vpara_hp[i], 2) - pow(vpara_hm[i], 2);
            dv3[i] = pow(vpara_hp[i], 3) - pow(vpara_hm[i], 3);
        }

    }

    void push_kperp(const double kperp) {
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
            for (size_t k = 0; k < 6; k++) {
                s_dv1_q_ms[ni][k] = 0.0;
                s_dv2_q_ms[ni][k] = 0.0;
                s_dv3_q_ms[ni][k] = 0.0;
                s_dv1_q_cs[ni][k] = 0.0;
                s_dv2_q_cs[ni][k] = 0.0;
            }
            for (size_t i = 0; i < npara - 1; i++) {
                for (size_t k = 0; k < 6; k++) {
                    q_ms[ni][k][i] = (qs[ni][k][i + 1] - qs[ni][k][i]) / dv1[i];
                    q_cs[ni][k][i] = qs[ni][k][i] - q_ms[ni][k][i] * vpara_hm[i];
                    s_dv1_q_ms[ni][k] += q_ms[ni][k][i] * dv1[i];
                    s_dv2_q_ms[ni][k] += q_ms[ni][k][i] * dv2[i];
                    s_dv3_q_ms[ni][k] += q_ms[ni][k][i] * dv3[i];
                    s_dv1_q_cs[ni][k] += q_cs[ni][k][i] * dv1[i];
                    s_dv2_q_cs[ni][k] += q_cs[ni][k][i] * dv2[i];
                }
            }
            //Now add up distributions

        }
    }

    array<array<array<array<cdouble, 7>, 5>, 3>, 3> push_omega(const double kpara, const double wr, const double wi) {
        array<array<array<array<cdouble, 7>, 5>, 3>, 3> XP{};
        for (size_t ni = 0; ni < ns.size(); ni++) {
            int n{ns[ni]};
            double a{wr};
            double b{wi};
            double b2{b * b};
            double d{n * wc};
            cdouble apb{a, b};
            cdouble apbmd{a - d, b};
            cdouble iapb{1.0 / apb};
            cdouble iapbmd{1.0 / apbmd};

            array<cdouble, 6> s_log_q_m{};
            array<cdouble, 6> s_log_q_c{};

            const double *s_dv1_q_m{s_dv1_q_ms[ni].data()};
            const double *s_dv1_q_c{s_dv1_q_cs[ni].data()};
            const double *s_dv2_q_m{s_dv2_q_ms[ni].data()};
            const double *s_dv2_q_c{s_dv2_q_cs[ni].data()};
            const double *s_dv3_q_m{s_dv3_q_ms[ni].data()};

            for (size_t i = 0; i < npara - 1; i++) {
                //cdouble clogfac{log((apb - d - vpara_hp[i]*kpara)/(apb - d - vpara_hm[i]*kpara))};
                //(apb - d - vpara_hp[i]*kpara)/(apb - d - vpara_hm[i]*kpara)
                //(apb - d - vpara_hm[i]*kpara - dv1[i]*kpara)/(apb - d - vpara_hm[i]*kpara)
                //1 - dv1[i]*kpara/(apb - d - vpara_hm[i]*kpara)
                //1 - dv1[i]*kpara*dv1[i]*kpara/(apb - d - vpara_hm[i]*kpara)*(apb - d - vpara_hm[i]*kpara)
                double advkm = a - d - vpara_hm[i] * kpara;
                //double rlogfac = 0.5 * log1p(dv1[i] * kpara * (dv1[i] * kpara - 2.0 * advkm) / (advkm * advkm + b2));
                double rlogfac = 0.5 * log1p(dv1[i] * kpara * dv1[i] * kpara / (advkm * advkm + b2));
                cdouble temp = (apbmd - vpara_hp[i] * kpara) / (apbmd - vpara_hm[i] * kpara);

                //Division by complex number...?
                //(e + if) = (a + ib)/(c + id)
                //(e + if)*(c + id) = a + ib
                //e*c - f*d + ifc + ide = a + ib
                //e*c - f*d = a
                //f*c + d*e = b
                //a, b, c, d are known, e, f are not
                //e = (a + f*d)/c
                //f = (b - d*e)/c
                //e = (a + d*(b - de)/c)/c
                //e*c*c + d*d*e = a*c + d*b
                //e = (a*c + b*d)/(d*d + c*c)
                //f = (b - d*(a + f*d)/c)/c
                //f*c*c + d*d*f = b*c - d*a
                //f = (b*c - d*a)/(d*d + c*c)
                //taylor expand arctan2(f, e)
                //Gives: ~ (b*c - d*a)/(a*c + b*d)
                //a = a - d - vpara_hp[i] * kpara
                //b = b
                //c = a - d - vpara_hm[i] * kpara
                //d = b
                // b * dv1[i] * kpara/((a - d - (vpara_hm[i] + dv1[i]) * kpara)*(a - d - vpara_hm[i] * kpara) + b2)
                //(a - d - (vpara_hm[i] + dv1[i]) * kpara)*(a - d - vpara_hm[i] * kpara)
                //(advkm)*(advkm - 2*dv1[i]*kpara)
                double ilogfac = atan2(temp.imag(), temp.real());
                //double ilogfac = temp.imag()/temp.real();
                cdouble clogfac{rlogfac, ilogfac};

                for (size_t k = 0; k < 6; k++) {
                    s_log_q_m[k] += clogfac * q_ms[ni][k][i];
                    s_log_q_c[k] += clogfac * q_cs[ni][k][i];
                }
            }

            cdouble c00 = M_PI * 2.0 * wp * wp * wc * wc * n * n * iapb * iapb;
            XP[0][0][0][2] -= c00 * apbmd * d * s_log_q_m[0];
            XP[0][0][0][3] -= c00 * (d * (s_dv1_q_m[0] + s_log_q_c[0]) + apbmd * s_log_q_m[1]);
            XP[0][0][0][4] -= c00 * (-s_dv1_q_c[0] + s_dv1_q_m[1] - 0.5 * s_dv2_q_m[0] + s_log_q_c[1]);

            cdouble c01 = M_PI * 2.0 * wp * wp * n * wc * cdouble{0.0, 1.0} * iapb * iapb;
            XP[0][1][1][2] -= c01 * apbmd * d * s_log_q_m[2];
            XP[0][1][1][3] -= c01 * (d * (s_dv1_q_m[2] + s_log_q_c[2]) + apbmd * s_log_q_m[3]);
            XP[0][1][1][4] -= c01 * (-s_dv1_q_c[2] + s_dv1_q_m[3] - 0.5 * s_dv2_q_m[2] + s_log_q_c[3]);

            cdouble c02 = M_PI * 2.0 * wp * wp * n * wc * iapb * iapb;
            XP[0][2][1][1] -= c02 * d * apbmd * apbmd * s_log_q_m[0];
            XP[0][2][1][2] -= c02 * apbmd * (d * s_dv1_q_m[0] + d * s_log_q_c[0] + apbmd * s_log_q_m[1]);
            XP[0][2][1][3] -=
                    c02 * (d * s_dv1_q_c[0] + apbmd * s_dv1_q_m[1] + 0.5 * d * s_dv2_q_m[0] + apbmd * s_log_q_c[1]);
            XP[0][2][1][4] -=
                    c02 * (s_dv1_q_c[1] - 0.5 * s_dv2_q_c[0] + 0.5 * s_dv2_q_m[1] - (1.0 / 3.0) * s_dv3_q_m[0]);

            cdouble c10 = M_PI * 2.0 * wp * wp * n * wc * cdouble{0.0, -1.0} * iapb * iapb;
            XP[1][0][1][2] -= c10 * apbmd * d * s_log_q_m[2];
            XP[1][0][1][3] -= c10 * (d * (s_dv1_q_m[2] + s_log_q_c[2]) + apbmd * s_log_q_m[3]);
            XP[1][0][1][4] -= c10 * (-s_dv1_q_c[2] + s_dv1_q_m[3] - 0.5 * s_dv2_q_m[2] + s_log_q_c[3]);

            cdouble c11 = M_PI * 2.0 * wp * wp * iapb * iapb;
            XP[1][1][2][2] -= c11 * apbmd * d * s_log_q_m[4];
            XP[1][1][2][3] -= c11 * (d * (s_dv1_q_m[4] + s_log_q_c[4]) + apbmd * s_log_q_m[5]);
            XP[1][1][2][4] -= c11 * (-s_dv1_q_c[4] + s_dv1_q_m[5] - 0.5 * s_dv2_q_m[4] + s_log_q_c[5]);

            cdouble c12 = M_PI * 2.0 * wp * wp * cdouble{0.0, -1.0} * iapb * iapb;
            XP[1][2][2][1] -= c12 * d * apbmd * apbmd * s_log_q_m[2];
            XP[1][2][2][2] -= c12 * apbmd * (d * s_dv1_q_m[2] + d * s_log_q_c[2] + apbmd * s_log_q_m[3]);
            XP[1][2][2][3] -=
                    c12 * (d * s_dv1_q_c[2] + apbmd * s_dv1_q_m[3] + 0.5 * d * s_dv2_q_m[2] + apbmd * s_log_q_c[3]);
            XP[1][2][2][4] -=
                    c12 * (s_dv1_q_c[3] - 0.5 * s_dv2_q_c[2] + 0.5 * s_dv2_q_m[3] - (1.0 / 3.0) * s_dv3_q_m[2]);

            cdouble c20 = M_PI * 2.0 * wp * wp * n * wc * iapb * iapb;
            XP[2][0][1][1] -= c20 * d * apbmd * apbmd * s_log_q_m[0];
            XP[2][0][1][2] -= c20 * apbmd * (d * s_dv1_q_m[0] + d * s_log_q_c[0] + apbmd * s_log_q_m[1]);
            XP[2][0][1][3] -=
                    c20 * (d * s_dv1_q_c[0] + apbmd * s_dv1_q_m[1] + 0.5 * d * s_dv2_q_m[0] + apbmd * s_log_q_c[1]);
            XP[2][0][1][4] -=
                    c20 * (s_dv1_q_c[1] - 0.5 * s_dv2_q_c[0] + 0.5 * s_dv2_q_m[1] - (1.0 / 3.0) * s_dv3_q_m[0]);

            cdouble c21 = M_PI * 2.0 * wp * wp * cdouble{0.0, 1.0} * iapb * iapb;
            XP[2][1][2][1] -= c21 * d * apbmd * apbmd * s_log_q_m[2];
            XP[2][1][2][2] -= c21 * apbmd * (d * s_dv1_q_m[2] + d * s_log_q_c[2] + apbmd * s_log_q_m[3]);
            XP[2][1][2][3] -=
                    c21 * (d * s_dv1_q_c[2] + apbmd * s_dv1_q_m[3] + 0.5 * d * s_dv2_q_m[2] + apbmd * s_log_q_c[3]);
            XP[2][1][2][4] -=
                    c21 * (s_dv1_q_c[3] - 0.5 * s_dv2_q_c[2] + 0.5 * s_dv2_q_m[3] - (1.0 / 3.0) * s_dv3_q_m[2]);

            cdouble c22 = M_PI * 2.0 * wp * wp * iapb * iapb;
            XP[2][2][2][0] -= c22 * apbmd * apbmd * apbmd * d * s_log_q_m[0];
            XP[2][2][2][1] -= c22 * (apbmd * apbmd * (apbmd * s_log_q_m[1] + d * (s_dv1_q_m[0] + s_log_q_c[0])));
            XP[2][2][2][2] -=
                    c22 * (apbmd * (apbmd * (s_dv1_q_m[1] + s_log_q_c[1]) + d * s_dv1_q_c[0] + 0.5 * d * s_dv2_q_m[0]));
            XP[2][2][2][3] -= c22 * (apbmd * (s_dv1_q_c[1] + 0.5 * s_dv2_q_m[1]) + 0.5 * d * s_dv2_q_c[0] +
                                     (1.0 / 3.0) * d * s_dv3_q_m[0]);
        }

        return XP;
    }
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

    cdouble evaluate(const double wr, const double wi) {
        array<array<array<array<cdouble, 7>, 5>, 3>, 3> M{};
        for (Species &spec: species) {
            M += spec.push_omega(kpara, wr, wi);
        }

        M[0][0][2][4] += 1.0;
        M[1][1][2][4] += 1.0;
        M[2][2][2][4] += 1.0;

        //cout<<polyEval(M[0][0], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[0][1], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[1][0], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[1][1], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[0][2], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[2][0], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[1][2], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[2][1], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;
        //cout<<polyEval(M[2][2], kpara, kperp)/(pow(kpara, 4)*pow(kperp, 2))<<endl;

        cdouble c2_w2 = pow(cl / cdouble{wr, wi}, 2);

        M[0][0][2][6] -= c2_w2;
        M[0][2][3][5] += c2_w2;
        M[1][1][2][6] -= c2_w2;
        M[1][1][4][4] -= c2_w2;
        M[2][0][3][5] += c2_w2;
        M[2][2][4][4] -= c2_w2;

        cdouble det = polyDet(M, kpara, kperp) / (pow(kpara, 12) * pow(kperp, 6));
        return det * pow(wr, 4) / pow((kpara * kpara + kperp * kperp) * cl * cl, 2);
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




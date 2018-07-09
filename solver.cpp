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

//template<size_t M1, size_t M2, size_t N1, size_t N2, typename T>
//static array<array<T, N1 + N2 - 1>, M1 + M2 - 1>
//polyMul(const array<array<T, N1>, M1> &a1, const array<array<T, N2>, M2> &a2) {
//    array<array<T, N1 + N2 - 1>, M1 + M2 - 1> p{};
//    for (size_t i = 0; i < M1; i++) {
//        for (size_t j = 0; j < N1; j++) {
//            for (size_t k = 0; k < M2; k++) {
//                for (size_t l = 0; l < N2; l++) {
//                    p[i + k][j + l] += a1[i][j] * a2[k][l];
//                }
//            }
//        }
//    }
//    return p;
//};

//template<size_t M, size_t N, typename T>
//static T polyEval(const array<array<T, N>, M> &p, const double &x, const double &y) {
//    T sum{};
//    double py{1.0};
//    for (size_t i = 0; i < M; i++) {
//        double px{1.0};
//        for (size_t j = 0; j < N; j++) {
//            sum += px * py * p[i][j];
//            px *= x;
//        }
//        py *= y;
//    }
//    return sum;
//};


//template<size_t M, size_t N, typename T, typename X, typename Y>
//static T polyDet(const array<array<array<array<T, N>, M>, 3>, 3> &p, const X &x, const Y &y) {
//    //array<array<T, N*3-2>, M*3-2> sum{};
//    T sum{};
//    for (size_t i = 0; i < 3; i++) {
//        //sum+=polyMul(polyMul(p[1][(i+1)%3], p[2][(i+2)%3]), p[0][i]);
//        //sum-=polyMul(polyMul(p[1][(i+2)%3], p[2][(i+1)%3]), p[0][i]);
//
//        sum += polyEval(p[1][(i + 1) % 3], x, y) * polyEval(p[2][(i + 2) % 3], x, y) * polyEval(p[0][i], x, y);
//        sum -= polyEval(p[1][(i + 2) % 3], x, y) * polyEval(p[2][(i + 1) % 3], x, y) * polyEval(p[0][i], x, y);
//    }
//    return sum;
//    //return polyEval(sum, x, y);
//};

template<typename T>
static T det(const array<array<T, 3>, 3> &p) {
    T sum{};
    for (size_t i = 0; i < 3; i++) {
        sum += p[1][(i + 1) % 3] * p[2][(i + 2) % 3] * p[0][i];
        sum -= p[1][(i + 2) % 3] * p[2][(i + 1) % 3] * p[0][i];
    }
    return sum;
};

class Species {

public://TODO: private later

    arr2d df_dvpara_h, df_dvperp_h;
    arr1d dvperp_h;
    arr1d vpara_h, vperp_h;
    arr1d vpara_hm, vpara_hp, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8;
    vector<int> ns;
    vector<arr1d> jns, jnps;
    vector<array<arr1d, 6>> qs, q_ms, q_cs;
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
            for (size_t i = 0; i < npara - 1; i++) {
                for (size_t k = 0; k < 6; k++) {
                    q_ms[ni][k][i] = (qs[ni][k][i + 1] - qs[ni][k][i]) / dv1[i];
                    q_cs[ni][k][i] = qs[ni][k][i] - q_ms[ni][k][i] * vpara_hm[i];
                }
            }
        }
    }

    array<array<cdouble, 3>, 3> push_omega(const double kpara, const double kperp, const double wr, const double wi) {
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
                if ((vpara_hm[i]+vpara_hp[i])*(vpara_hm[i]+vpara_hp[i])*(cx.real()*cx.real() + cx.imag()*cx.imag()) < 0.000001){
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
                    if (wi<0.0){
                        cdouble di = I*L0.imag();
                        L0 -= di;
                        L1 -= di;
                        L2 -= di;
                        L3 -= di;
                    }
                }else{
                    double advkm = a - d - vpara_hm[i] * kpara;
                    double rlogfac = 0.5 * log1p(dv1[i] * kpara * (dv1[i] * kpara - 2.0 * advkm) / (advkm * advkm + b2));
                    double ilogfac = atan2(abs(b)*dv1[i]*kpara, (advkm + dv1[i]*kpara)*advkm + b2);
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
                }

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

//        cout<<XP[0][0]/(kpara*kpara*kpara*kpara*kperp*kperp)<<endl;
//        cout<<XP[0][1]/(kpara*kpara*kpara*kpara*kperp*kperp)<<endl;
//        cout<<XP[1][1]/(kpara*kpara*kpara*kpara*kperp*kperp)<<endl;
//        cout<<XP[2][2]/(kpara*kpara*kpara*kpara*kperp*kperp)<<endl;

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
        double wa = sqrt(wr*wr + wi*wi);

        M[0][0] -= c2_w2*kperp2*kpara6;
        M[0][2] += c2_w2*kperp3*kpara5;
        M[1][1] -= c2_w2*kperp2*kpara6;
        M[1][1] -= c2_w2*kperp4*kpara4;
        M[2][0] += c2_w2*kperp3*kpara5;
        M[2][2] -= c2_w2*kperp4*kpara4;

        cdouble detM = det(M) / (pow(kpara, 12) * pow(kperp, 6));
        return detM * pow(wa, 4) / pow((kpara * kpara + kperp * kperp) * cl * cl, 2);
    }

    np::ndarray evaluateM(const double wr, const double wi) { /*Unfinished! This is a stub!*/
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

        p::tuple shape = p::make_tuple(3, 3);
        np::ndarray npM = np::zeros(shape, np::dtype::get_builtin<cdouble>());

        for (size_t i=0;i<3;i++){
            for (size_t j=0;j<3;j++){
                npM[i][j] = M[i][j]/(kperp2*kpara4);
            }
        }

        return npM;
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
            .def("evaluateM", &Solver::evaluateM)
            .def("marginalize", &Solver::marginalize);

}




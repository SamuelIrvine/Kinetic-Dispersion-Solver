import numpy as np
#import matplotlib.pyplot as plt
import libSolver
import time
from scipy.special import jv
from scipy.special import jvp
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.integrate import simps


class Species:

    def __init__(self, charge, mass, density, vpara, vperp, fv, ns):
        """Constructs a species object with the required fields. """
        assert(vpara.shape[0] == fv.shape[0])
        assert(vperp.shape[0] == fv.shape[1])
        assert(len(ns) != 0)

        self.charge = charge
        self.mass = mass
        self.density = density
        self.ns = list(ns)

        self.fv_h = 0.5*fv[1:, 1:] + 0.5*fv[:-1, :-1]
        self.vpara_h = 0.5*vpara[1:] + 0.5*vpara[:-1]
        self.vperp_h = 0.5*vperp[1:] + 0.5*vperp[:-1]
        self.dvpara_h = vpara[1:] - vpara[:-1]
        self.dvperp_h = vperp[1:] - vperp[:-1]
        normfv = 1.0/np.sum(2.0*np.pi*np.outer(np.ones(len(vpara)-1), self.vperp_h)*self.fv_h*np.outer(self.dvpara_h, self.dvperp_h))
        self.fv_h *= normfv
        self.df_dvpara_h = normfv*(0.5*fv[1:, 1:] - 0.5*fv[:-1, 1:] + 0.5*fv[1:, :-1] - 0.5*fv[:-1, :-1])/np.outer(self.dvpara_h, np.ones(len(vperp)-1))
        self.df_dvperp_h = normfv*(0.5*fv[1:, 1:] - 0.5*fv[1:, :-1] + 0.5*fv[:-1, 1:] - 0.5*fv[:-1, :-1])/np.outer(np.ones(len(vpara)-1), self.dvperp_h)


class Solver:

    def __init__(self, B, species):
        """Constructs a dispersion solver from a static magnetic field and a collection of Species objects. """
        self.lib = libSolver.Solver(list(species), B)
        self.kpara = None
        self.kperp = None

    def set_k(self, k):
        kpara = k[0]
        kperp = k[1]
        if not kpara == self.kpara:
            self.lib.push_kpara(kpara)
            self.kpara = kpara
        if not kperp == self.kperp:
            self.lib.push_kperp(kperp)
            self.kperp = kperp

    def marginalize(self, ww, k):
        """Computes an array of insolutions given a fixed value of k and array of complex frequencies ww. """
        self.set_k(k)
        insolution = self.lib.marginalize(ww)
        return insolution

    def identify(self, ww, k, insolution):
        """Identifies local minima by marginalization of insolution for a complex array of frequencies and fixed value of k. """
        insolution = np.abs(insolution)
        if len(np.shape(insolution)) == 1:
            minimas = np.where((insolution[1:-1] < insolution[2:])
                               & (insolution[1:-1] < insolution[:-2]))
            bounds = []
            for minima in zip(*minimas):
                bounds.append(((ww[zip(minima - np.array([1]))].real, ww[zip(minima + np.array([1]))].real),
                               (None, None)))
            return zip(ww[1:-1][minimas], insolution[1:-1][minimas], bounds)
        else:
            inspad = np.zeros([insolution.shape[0]+2, insolution.shape[1]+2])
            inspad[1:-1, 1:-1] = insolution
            inspad[1:-1, 0] = insolution[:, 0]*1.001
            inspad[1:-1, -1] = insolution[:, -1]*1.001
            inspad[0, 1:-1] = insolution[0, :]*1.001
            inspad[-1, 1:-1] = insolution[-1, :]*1.001
            wwpad = np.zeros([ww.shape[0]+2, ww.shape[1]+2], dtype='complex')
            wwpad[1:-1, 1:-1] = ww
            wwpad[1:-1, 0] = 2*ww[:, 0] - ww[:, 1]
            wwpad[1:-1, -1] = 2*ww[:, -1] - ww[:, -2]
            wwpad[0, 1:-1] = 2*ww[0, :] - ww[1, :]
            wwpad[-1, 1:-1] = 2*ww[-1, :] - ww[-2, :]
            minimas = np.where((inspad[1:-1, 1:-1] < inspad[2:, 1:-1])
                               & (inspad[1:-1, 1:-1] < inspad[:-2, 1:-1])
                               & (inspad[1:-1, 1:-1] < inspad[1:-1, 2:])
                               & (inspad[1:-1, 1:-1] < inspad[1:-1, :-2])
                               & (inspad[1:-1, 1:-1] < inspad[2:, 2:])
                               & (inspad[1:-1, 1:-1] < inspad[2:, :-2])
                               & (inspad[1:-1, 1:-1] < inspad[:-2, 2:])
                               & (inspad[1:-1, 1:-1] < inspad[:-2, :-2]))
            bounds = []
            for minima in zip(*minimas):
                bounds.append(((wwpad[0:-2, 1:-1][zip(minima)].real, wwpad[2:, 1:-1][zip(minima)].real),
                               (wwpad[1:-1, 0:-2][zip(minima)].imag, wwpad[1:-1, 2:][zip(minima)].imag)))
            return zip(ww[minimas], insolution[minimas], bounds)


    def solve(self, wguess, k, bounds=((None, None), (None, None))):
        """Attempts to compute numerical convergence given a single frequency guess and fixed value of k. """
        self.set_k(k)
        wr = wguess.real
        wi = wguess.imag
        wrbounds = bounds[0]
        wibounds = bounds[1]
        dwr = abs(wrbounds[1] - wrbounds[0])[0]
        if (wibounds[1] == None or wibounds[0] == None):
            wibounds = (np.array([wi*0.95, 0.0]), np.array([wi*1.05, 0.0]))
        dwi = abs(wibounds[1] - wibounds[0])[0]
        eps = min(dwr, dwi)*0.0001
        def f(x):
            res = self.lib.evaluateDetM(20*(x[0] - 1.0)*dwr + wr, 20*(x[1] - 1.0)*dwi + wi)
            return np.abs(res)
        res = minimize(f, (1.0, 1.0), method='Nelder-Mead', options={'fatol': 1E-6, 'xatol': 1E-6, 'disp': False, 'maxiter':200})
        return 20*(res['x'][0] - 1.0)*dwr + wr + 1.0j*(20*(res['x'][1] - 1.0)*dwi + wi), res['fun']

    def roots(self, ww, k, insolution=None):
        """Finds roots for a given marginalization of complex frequencies and value of k. """
        if insolution == None:
            insolution = self.marginalize(ww, k)
        candidates = self.identify(ww, k, insolution)
        roots = []
        for candidate, insol, bounds in candidates:
            root, insol = self.solve(candidate, k, bounds)
            if insol>0.01:
                continue
            #if np.abs(root.imag)>np.abs(root.real)*0.1:
            #	continue
            roots.append((root, insol))
        return roots

    def growth(self, ww, k):
        roots = self.roots(ww, k)
        gamma = [0.0, 0.0, 1E200]
        for root, insol in roots:
            if root.imag<gamma[0]:
                continue
            gamma = (root.imag, root, insol)
        return gamma

    def polarization(self, w, k):
        wr = w.real
        wi = w.imag
        self.set_k(k)
        M = self.lib.evaluateM(wr, wi)
        eigvals, Es = np.linalg.eig(M)
        Es = Es.T[np.argsort(np.abs(eigvals))]
        e1 = Es[0] #solution direction
        e1 = e1/np.sum(np.abs(e1)**2)**0.5
        ku = np.array((k[1], 0.0, k[0]))/(k[0]**2 + k[1]**2)**0.5 #propagation direction
        dop = np.arccos(np.abs(np.abs(e1).dot(ku))) #Degree of polarization
        theta = np.arctan2(k[1], k[0])
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        Ry = np.matrix(((ctheta, 0.0, stheta),
                        (0.0, 1.0, 0.0),
                        (-stheta, 0.0, ctheta)))
        e1r = Ry.dot(e1) #Now in basis where we can determine polarization
        px = e1r[0, 0]
        py = e1r[0, 1]
        phi = np.arctan2(np.abs(py), np.abs(px)) #orientation of polarization
        alphax = np.arctan2(px.imag, px.real)
        alphay = np.arctan2(py.imag, py.real)
        beta = alphay - alphax
        if beta > np.pi:
            beta = beta - 2*np.pi
        if beta < -np.pi:
            beta = beta + 2*np.pi
        if beta < -np.pi/2:
            beta -= 2*beta + np.pi
        if beta > np.pi/2:
            beta -= 2*beta - np.pi
        return dop, beta, phi

    def getTensorElements(self, w, k):
        wr = w.real
        wi = w.imag
        self.set_k(k)
        return self.lib.evaluateM(wr, wi)

    def getDielectricTensorElements(self, w, k):
        wr = w.real
        wi = w.imag
        self.set_k(k)
        return self.lib.evaluateEps(wr, wi)



    #return 2 variables, P, beta



if __name__ == '__main__':
    kb = 1.38E-23
    me = 9.11E-31
    T = 11500*1000
    B = 2.0
    e = 1.6E-19
    e0 = 8.85E-12
    density = 2.5E19
    vth = (kb*T/me)**0.5

    wpe = (density*e**2/(e0*me))**0.5
    wce = e*B/me

    #Outline distribution bounds
    vparamin = -5*vth
    vparamax = 25*vth
    vperpmin = 0.0*vth
    vperpmax = 5*vth

    #Define grids and staggered grids
    npara = 1000
    nperp = 1000
    vpara = np.linspace(vparamin, vparamax, npara, dtype='float64')
    vperp = np.linspace(vperpmin, vperpmax, nperp, dtype='float64')

    #Define distributions
    xi = 0.00512 #Tail fraction
    vm = 18.5*vth #Tail max velocity
    Fvperp = np.exp((-0.5*vperp**2/vth**2)) #Note: we have defined temperature this way.
    Fvpara = np.exp((-0.5*vpara**2/vth**2))
    Fvpara = Fvpara/np.sum(Fvpara) #Normalize to 1 as intermediate step
    Fvtail = np.where((vpara>0.0) & (vpara<vm), 1.0, 0.0) #Define bounds of flat tail
    Fvtail = xi*Fvtail/np.sum(Fvtail) #Normalize tail
    Fvpara = np.maximum(Fvpara, Fvtail) #Combine tail with background maxwellian
    Fv = np.outer(Fvpara, Fvperp)

    s1 = Species(-1*e, me, density, vpara, vperp, Fv, np.arange(-2, 3))

    solver = Solver(B, [s1])


    w = wpe*1.2 + wpe*0.0011j
    k = [0.12*wpe/vth, 0.11*wpe/vth]

    #solver.lib.push_kpara(k[0])
    #solver.lib.push_kperp(k[1])
    #solver.lib.evaluate(w.real, w.imag)

    #a = (wpe**2/((w-wce)*(w+wce)))
    #b = (1.0j*wce*wpe**2/(w*(w+wce)*(w-wce)))
    #c = wpe**2/w**2

    #print 1.0 - a
    #print b
    #print 1.0 - c

    #print (1.0 - c)*((1.0 - a)*(1.0 - a) - (b)*(-b))

    soln = solver.solve(w, k)
    print soln
    print soln[0]/wpe

    wrs = np.linspace(0.001*wpe, 2*wpe, 100)
    wis = np.linspace(0.00001*wpe, 0.001*wpe, 100)*1.0j
    wws = np.outer(wrs, np.ones(len(wis))) + np.outer(np.ones(len(wrs)), wis)

    #ww = np.linspace(0.001*wpe, 2*wpe, 1000) + 0.00000001j*wpe
    t0 = time.time()
    mms = solver.marginalize(wws, k)
    t1 = time.time()
    print 'Duration: '+str(t1-t0)
    print np.array(solver.growth(wws, k))/wpe
#plt.imshow(np.log(mms))
#plt.show()
#print 0.5*((wce**2+4*wpe**2)**0.5 + wce)/wpe
#print 0.5*((wce**2+4*wpe**2)**0.5 - wce)/wpe
#print 'Done!'
#roots = solver.roots(ww, k)
#for root in roots:
#	print root[0]/wpe


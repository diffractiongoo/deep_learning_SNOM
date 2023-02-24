from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D
import numpy as np
import numpy.matlib
import h5py
import matlab.engine
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import time

## start the matlab engine ##
eng = matlab.engine.start_matlab()


## function generates the eps ##
def eps(x, eps_inf, w_lo, gamma_lo, w_to, gamma_to):
    res = eps_inf * (w_lo ** 2 - x ** 2 - 1j * gamma_lo * x) / (w_to ** 2 - x ** 2 - 1j * gamma_to * x)
    res = np.sqrt(res) * np.sqrt(res)
    return res


## function for demodulation ##
def Demodulation(x, chi, Dz, Nh, ztip):
    Nw = chi.shape[1]
    chiNh = np.zeros((1, Nw), dtype=complex)
    for ii in range(Nw):
        chi_spline = interp1d(ztip, chi[:, ii], kind='cubic')
        res = chi_spline(ztip[0] + Dz * (1 - np.cos(x))) * np.cos(Nh * x)
        res_im = np.imag(res)
        res_re = np.real(res)
        chiNh[0, ii] = np.trapz(res_re, x=x) / np.pi + 1j * np.trapz(res_im, x=x) / np.pi
    return chiNh


## generates curve using Betak, Rk and parameters ##
def curve_generation(w, p):
    eps_inf, w_lo, gamma_lo, w_to, gamma_to, L1 = p[0], p[1], p[2], p[3], p[4], p[5]  ## parameters for calling eps function
    a = 1  ## define probe geometry
    zmin = 0.02 * a  ## min and max tip positions
    zmax = 10 * a
    Nz = 200  ##  number of grid points for tip sample separation
    N = 200  ## dimension of the eigenproblem
    Nz1 = int(np.ceil((Nz - 1) / 2))
    ztip = np.concatenate((np.linspace(zmin ** (1 / 3), (0.5 - 0.5 / Nz1) ** (1 / 3), Nz1, endpoint=True) ** 3,
                           np.linspace(0.5, zmax, Nz - Nz1, endpoint=True)), axis=None)  ## get the probe-sample distance array

    Dz = 5.0 / 3.0 * a  ## demodulation amplitude

    [betak, rk] = eng.FindBetaR(matlab.double([L1]), nargout=2)  ## use Matlab to get betak and rk corresponding to the predicted shape
    betak = np.array(betak)
    rk = np.array(rk)

    eps_samp = eps(w, eps_inf, w_lo, gamma_lo, w_to, gamma_to)  ## call eps function to get sample eps
    beta_samp = (eps_samp - 1) / (eps_samp + 1)  ## corresponding beta
    eps_ref = 11.7 + 0.1 * 1j  ## same procedure for reference eps and beta. The reference material is Si
    beta_ref = (eps_ref - 1) / (eps_ref + 1)
    chi_samp = np.zeros((Nz, Nw), dtype=complex)  ## create sample chi array
    chi_ref = np.zeros((Nz, 1), dtype=complex)  ## create reference chi array
    Nh = 3  ## harmonic order

    for h in range(Nz):  ## store values in the chi arrays
        for k in range(N):
            chi_samp[h, :] = chi_samp[h, :] + rk[k, h] / (betak[k, h] - beta_samp)
            chi_ref[h, :] = chi_ref[h, :] + rk[k, h] / (betak[k, h] - beta_ref)

    w_mat = 2 * np.pi * np.matlib.repmat(w, 200, 1)  ## adding radiative corrections
    chi_ref = np.matlib.repmat(chi_ref, 1, 351)
    chi_samp = ((3 * (10 ** -6)) ** 3 * chi_samp) / (1 - 1j * 2 / 3 * (w_mat ** 3) * chi_samp * (3 * (10 ** -6)) ** 3)
    chi_ref = ((3 * (10 ** -6)) ** 3 * chi_ref) / (1 - 1j * 2 / 3 * (w_mat ** 3) * chi_ref * (3 * (10 ** -6)) ** 3)

    phi = np.linspace(0, np.pi, 500, endpoint=True)
    chi3_samp = Demodulation(phi, chi_samp, Dz, Nh, ztip)  ## performing demodulation
    chi3_ref = Demodulation(phi, chi_ref, Dz, Nh, ztip)

    rPFF_samp = (2 ** -0.5 * eps_samp - (eps_samp - 0.5) ** 0.5) / (2 ** -0.5 * eps_samp + (eps_samp - 0.5) ** 0.5)  ## get the Far-Field factor and s3_bar
    FFF_samp = (1 + rPFF_samp) ** 2
    rPFF_ref = (2 ** -0.5 * eps_ref - (eps_ref - 0.5) ** 0.5) / (2 ** -0.5 * eps_ref + (eps_ref - 0.5) ** 0.5)
    FFF_ref = (1 + rPFF_ref) ** 2

    s3_samp = chi3_samp * FFF_samp
    s3_ref = chi3_ref * FFF_ref
    s3_bar = s3_samp / s3_ref

    return np.hstack([np.real(s3_bar), np.imag(s3_bar)])


## function for least square ##
def func(pp, w, y):
    wmax = 900.0
    wmin = 550.0
    eps_min = 1.0
    eps_max = 5.0
    gamma_min = 1.0
    gamma_max = 50.0
    Lmin = 5
    Lmax = 25
    eps_inf = eps_min + pp[0]*(eps_max-eps_min)
    w_lo = wmin + pp[1]*(wmax-wmin)
    w_to = wmin + pp[3]*(wmax-wmin)
    gamma_lo = gamma_min + pp[2]*(gamma_max-gamma_min)
    gamma_to = gamma_min + pp[2] * (gamma_max - gamma_min)
    L = Lmin + (Lmax - Lmin) * pp[4]

    p = [eps_inf, w_lo, gamma_lo, w_to, gamma_to, L]

    return curve_generation(w, p)[0] - y


## define the ranges for parameters ##
wmax = 900.0
wmin = 550.0
wpmax = 300.0
wpmin = 100.0
wd = 10.0
eps_min = 1.0
eps_max = 5.0
gamma_min = 1.0
gamma_max = 50.0
Lmax = 25.0
Lmin = 5.0
Nw = 351
w = np.linspace(wmin, wmax, Nw, endpoint=True)


## define the ANN ##
model = Sequential()
model.add(Conv1D(64, activation='relu', input_shape=(351, 2,), kernel_size=(25)))
model.add(Conv1D(128, activation='relu', kernel_size=(15)))
model.add(Conv1D(256, activation='relu', kernel_size=(5)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='relu'))

model.load_weights('ML_spheroid_paper_radiative_change_L_Adam_0.0001.h5') ## load model weights


## load the dataset for testing ##
with h5py.File('spheroid_paper_change_L_test_random1000.mat', 'r') as f:

    S_matrix = np.array(f.get('S_matrix'))
    Parameters = np.array(f.get('Parameters'))
    parameters = np.transpose(Parameters)

    Data = np.zeros(shape=(1000, 351, 2))
    for l in range(1, 1001):
        temp = [j[l - 1] for j in S_matrix]
        Data[l - 1, :, 0] = [k[0] for k in temp]
        Data[l - 1, :, 1] = [k[1] for k in temp]

Real_Parameters = np.zeros(shape=(1000, 5))
Parameters_ML = np.zeros(shape=(1000, 5))
Parameters_LS = np.zeros(shape=(1000, 5))
Time = np.zeros(shape=(1000, 1))


## evaluate the testing spectra ##
for k in range(1, 1001):
    print('step: ' + str(k))

    s3_bar_1 = np.concatenate((Data[k - 1, :, 0], Data[k - 1, :, 1]), axis=None) ## ground truth for least square optimization
    test_mat = np.zeros(shape=(1, np.size(w), 2))  ## prepare the matrix to put into the network
    test_mat[:, :, 0] = s3_bar_1[0:np.size(w)]
    test_mat[:, :, 1] = s3_bar_1[np.size(w):]

    parms_ML1 = model.predict(test_mat)[0]  ## pridicted parameters in range 0 to 1
    parms_ML = np.zeros(shape=(1, 5))
    parms_ML_LS = np.zeros(shape=(1, 5))
    real_parms = np.zeros(shape=(1, 5))

    real_parms[0, 0] = eps_min + (eps_max - eps_min) * parameters[k - 1, 0]  ## get the true parameters
    real_parms[0, 1] = wmin + (wmax - wmin) * parameters[k - 1, 1]
    real_parms[0, 2] = gamma_min + (gamma_max - gamma_min) * parameters[k - 1, 2]
    real_parms[0, 3] = wmin + (wmax - wmin) * parameters[k - 1, 3]
    real_parms[0, 4] = Lmin + (Lmax - Lmin) * parameters[k - 1, 4]

    parms_ML[0, 0] = eps_min + (eps_max - eps_min) * parms_ML1[0]  ## get the ANN predicted parameters
    parms_ML[0, 1] = wmin + (wmax - wmin) * parms_ML1[1]
    parms_ML[0, 2] = gamma_min + (gamma_max - gamma_min) * parms_ML1[2]
    parms_ML[0, 3] = wmin + (wmax - wmin) * parms_ML1[3]
    parms_ML[0, 4] = Lmin + (Lmax - Lmin) * parms_ML1[4]

    print('Running least_squares...')  ## running the least squares optimization for iHNN
    t_start = time.time()  ## record the time used
    Result = least_squares(func, parms_ML1, args=(w, s3_bar_1), bounds=([-0.2, -0.2, -0.2, -0.2, 0], [1.2, 1.2, 1.2, 1.2, 1.2]), verbose=1, method='trf')
    parms_ML_LS_1 = Result.x
    t_used = time.time() - t_start
    print('%5.1fs: least_squares done' % (t_used))

    parms_ML_LS[0, 0] = eps_min + (eps_max - eps_min) * parms_ML_LS_1[0]  ## get the iHNN predicted values
    parms_ML_LS[0, 1] = wmin + (wmax - wmin) * parms_ML_LS_1[1]
    parms_ML_LS[0, 2] = gamma_min + (gamma_max - gamma_min) * parms_ML_LS_1[2]
    parms_ML_LS[0, 3] = wmin + (wmax - wmin) * parms_ML_LS_1[3]
    parms_ML_LS[0, 4] = Lmin + (Lmax - Lmin) * parms_ML_LS_1[4]

    Real_Parameters[k - 1, :] = real_parms[0]  ## store the test instance
    Parameters_ML[k - 1, :] = parms_ML[0]
    Parameters_LS[k - 1, :] = parms_ML_LS[0]
    Time[k - 1, :] = t_used


## store the entire dataset ##
hf = h5py.File('spheroid_paper_radiative_change_L_ML_all_parameters_randomized_1000_Adam_0.0001.h5', 'w')
hf.create_dataset('R_P', data=Real_Parameters)
hf.create_dataset('P_ML', data=Parameters_ML)
hf.create_dataset('P_LS', data=Parameters_LS)
hf.create_dataset('T', data=Time)
hf.close()

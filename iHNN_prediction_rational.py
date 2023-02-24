from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D
import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d
import h5py
from scipy.optimize import least_squares, minimize
import time


## function generates the eps ##
def eps(x, eps_inf, w_lo, gamma_lo, w_to, gamma_to):
    res = eps_inf * (w_lo ** 2 - x ** 2 - 1j * gamma_lo * x) / (w_to ** 2 - x ** 2 - 1j * gamma_to * x)
    idx = [i for i, n in enumerate(res) if n.imag < 0]
    res[idx] = -res[idx]
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

    eps_inf, w_lo, gamma_lo, w_to, gamma_to = p[0], p[1], p[2], p[3], p[4]  ## parameters for calling eps function
    a = 1  ## define probe geometry
    L = 25 * a
    zmin = 0.02 * a  ## min and max tip positions
    zmax = 10 * a
    Nz = 200  ##  number of grid points for tip sample separation
    Nz1 = int(np.ceil((Nz - 1) / 2))
    ztip = np.concatenate((np.linspace(zmin ** (1 / 3), (0.5 - 0.5 / Nz1) ** (1 / 3), Nz1, endpoint=True) ** 3,
                           np.linspace(0.5, zmax, Nz-Nz1, endpoint=True)), axis=None)  ## get the probe-sample distance array

    Dz = 5 / 3 * a  ## Demodulation amplitude

    alpha = np.arccosh(1 + ztip / a)  ## use the rational approximation to get Betaks and Rks for the prob aspect ratio L/a = 25
                                      ## for more details, see the Supplementary Materials
    Z = ztip / a
    W = np.sqrt(a * L)
    F = np.sqrt(L ** 2 - W ** 2)
    xi0 = L / F
    chi0 = L ** 3 / (3 * xi0 ** 3) * (1 / 2 * np.log((xi0 + 1) / (xi0 - 1)) - 1 / xi0) ** (-1)

    A = np.array([[-0.015667, 1783.1, -762.76, 234.56, -36.399, 3], [0, 290.11, -93.002, 111.01, -25.733, 5], [0, 1391.3, -118.29, 157.55, -33.029, 7],
         [0, 1879.9, -40.018, 173.3, -36.251, 9], [-0.00017314, 2292.3, -85.286, 221.75, -47.517, 11], [-0.000094847, 3435.5, 42.551, 233.45, -45.678, 13],
         [0, 2547, 223.28, 254.23, -46.254, 15], [0, 1961.3, 770.95, 235.93, -27.808, 17], [0.000032595, 402.67, -308.28, 251.72, -65.583, 19]])

    B = np.array([[1522.2, -417.42, 83.048, -10.345, 1], [87.231, 11.949, 27.841, -3.4964, 1], [253.32, 149.14, -2.6274, -0.9961, 1],
         [246.31, 185.03, -8.2396, -0.12625, 1], [237.17, 180.3, -8.4205, -0.3866, 1], [291.59, 253.41, -20.329, 1.0728, 1],
         [185.07, 175.21, 7.941, -0.75722, 1], [122.51, 163.71, 34.251, -1.8957, 1], [23.24, -1.1364, 4.104, 1.6624, 1]])

    C = np.array([[1.4941, 282.17, 4811.1, 5141.1, 303.23, 3.9999], [12.255, 2371.9, 33226, 17089, 916.75, 12.001], [11.908, 36.881, 90005, 35207, 1844.7, 24.001],
         [4417.7, 224067, 288144, 65393, 3166.2, 40], [28.67, 5632.8, 196974, 87200, 4584.9, 60.028], [4.237, 4364.5, 226585, 122561, 6304.9, 84.685],
         [843.4, 98948, 316568, 166927, 8216.9, 116.26], [334.56, 77249, 350367, 214160, 10354, 146.83]])

    D = np.array([[0.000027594, 0.023552, 0.77084, 1], [0.0003052, 0.11961, 1.4472, 1], [0.00025669, 0.012624, 1.9006, 1], [0.09134, 2.6905, 4.1622, 1],
         [0.00048749, 0.066244, 1.5802, 1], [0.000077993, 0.044389, 1.2714, 1], [0.0087351, 0.44188, 1.4088, 1], [0.0034808, 0.28331, 1.1606, 1]])

    Betak = np.zeros((9, ztip.size))
    Rk = np.zeros((9, ztip.size))
    for i in range(ztip.size):
        A1 = [alpha[i] ** 0, alpha[i] ** 1, alpha[i] ** 2, alpha[i] ** 3, alpha[i] ** 4, alpha[i] ** 5]
        A1 = np.matlib.repmat(A1, 9, 1)

        B1 = [alpha[i] ** 0, alpha[i] ** 1, alpha[i] ** 2, alpha[i] ** 3, alpha[i] ** 4]
        B1 = np.matlib.repmat(B1, 9, 1)

        C1 = [Z[i] ** 0, Z[i] ** 1, Z[i] ** 2, Z[i] ** 3, Z[i] ** 4, Z[i] ** 5]
        C1 = np.matlib.repmat(C1, 8, 1)

        D1 = [Z[i] ** 0, Z[i] ** 1, Z[i] ** 2, Z[i] ** 3]
        D1 = np.matlib.repmat(D1, 8, 1)

        betak = np.exp(np.sum(A * A1, axis=1) / np.sum(B * B1, axis=1))
        rk = a ** 3 * Z[i] * np.sum(C * C1, axis=1) / np.sum(D * D1, axis=1)

        Rk8 = (chi0 - np.sum(rk / betak[0:8]))*betak[8]
        rk = np.append(rk, Rk8)
        Betak[:, i] = np.transpose(betak)
        Rk[:, i] = np.transpose(rk)

    eps_samp = eps(w, eps_inf, w_lo, gamma_lo, w_to, gamma_to)  ## call eps function to get sample eps
    beta_samp = (eps_samp-1) / (eps_samp+1)  ## corresponding beta
    eps_ref = 11.7+0.1*1j   ## same procedure for reference eps and beta. The reference material is Si
    beta_ref = (eps_ref-1)/(eps_ref+1)
    chi_samp = np.zeros((Nz, Nw), dtype=complex)  ## create sample chi array
    chi_ref = np.zeros((Nz, 1), dtype=complex)  ## create reference chi array
    Nh = 3  ## harmonic order

    for h in range(Nz):  ## store values in the chi arrays
        for k in range(9):
            chi_samp[h, :] = chi_samp[h, :] + Rk[k, h] / (Betak[k, h] - beta_samp)
            chi_ref[h, :] = chi_ref[h, :] + Rk[k, h] / (Betak[k, h] - beta_ref)

    w_mat = 2 * np.pi * np.matlib.repmat(w, 200, 1)  ## adding radiative corrections
    chi_ref = np.matlib.repmat(chi_ref, 1, 351)
    chi_samp = ((3 * (10 ** -6)) ** 3 * chi_samp) / (1 - 1j * 2 / 3 * (w_mat ** 3) * chi_samp * (3 * (10 ** -6)) ** 3)
    chi_ref = ((3 * (10 ** -6)) ** 3 * chi_ref) / (1 - 1j * 2 / 3 * (w_mat ** 3) * chi_ref * (3 * (10 ** -6)) ** 3)

    phi = np.linspace(0, np.pi, 500, endpoint=True)  ## performing demodulation
    chi3_samp = Demodulation(phi, chi_samp, Dz, Nh, ztip)
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
    wmax = 900
    wmin = 550
    eps_min = 1
    eps_max = 5
    gamma_min = 1
    gamma_max = 50
    eps_inf = eps_min + pp[0]*(eps_max-eps_min)
    w_lo = wmin + pp[1]*(wmax-wmin)
    w_to = wmin + pp[3]*(wmax-wmin)
    gamma_lo = gamma_min + pp[2]*(gamma_max-gamma_min)
    gamma_to = gamma_min + pp[2] * (gamma_max - gamma_min)
    p = [eps_inf, w_lo, gamma_lo, w_to, gamma_to]

    return curve_generation(w, p)[0] - y


## define the ranges for parameters ##
wmax = 900
wmin = 550
wpmax = 300
wpmin = 100
wd = 10
eps_min = 1
eps_max = 5
gamma_min = 1
gamma_max = 50
Lmax = 25
Lmin = 5
dw = 1
Nw = (wmax-wmin) // dw + 1
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
model.add(Dense(4, activation='relu'))

model.load_weights('ML_spheroid_paper_radiative_new_method_25a_relu_Adam_0.0001.h5')  ## load model weights


## load the dataset for testing ##
with h5py.File('spheroid_paper_radiative_new_method_25a_test_1.mat', 'r') as f:

    S_matrix = np.array(f.get('S_matrix'))
    Parameters = np.array(f.get('Parameters'))
    parameters = np.transpose(Parameters)
    parameters = parameters[:, 0:4]

    Data = np.zeros(shape=(1000, 351, 2))
    for l in range(1, 1001):
        temp = [j[l - 1] for j in S_matrix]
        Data[l - 1, :, 0] = [k[0] for k in temp]
        Data[l - 1, :, 1] = [k[1] for k in temp]

Real_Parameters = np.zeros(shape=(1000, 4))
Parameters_ML = np.zeros(shape=(1000, 4))
Parameters_LS = np.zeros(shape=(1000, 4))
LS_only = np.zeros(shape=(1000, 4))
Time_ML = np.zeros(shape=(1000, 1))
Time_LS_only = np.zeros(shape=(1000, 1))
Time_LS = np.zeros(shape=(1000, 1))


## evaluate the testing spectra ##
for k in range(1, 1001):
    print('step: ' + str(k))

    s3_bar_1 = np.concatenate((Data[k-1, :, 0], Data[k-1, :, 1]), axis=None)  ## ground truth for least square optimization
    test_mat = np.zeros(shape=(1, np.size(w), 2))  ## prepare the matrix to put into the network
    test_mat[:, :, 0] = s3_bar_1[0:np.size(w)]
    test_mat[:, :, 1] = s3_bar_1[np.size(w):]

    t_start = time.time()
    predicted_parms1 = model.predict(test_mat)[0]  ## predicted parameters in range 0 to 1
    t_used_ML = time.time() - t_start

    LS_initial = [0.5, 0.5, 0.5, 0.5]  ## initial values for pure least square optimization

    ## running the least squares optimization for iHNN ##
    t_start = time.time()
    Result = least_squares(func, predicted_parms1, args=(w, s3_bar_1), bounds=([0, 0, 0, 0], [1.2, 1.2, 1.2, 1.2]), verbose=1, method='trf')
    predicted_parms_LS_1 = Result.x
    t_used_LS = time.time() - t_start

    ## running pure least squares optimization ##
    t_start = time.time()
    Result1 = least_squares(func, LS_initial, args=(w, s3_bar_1), bounds=(0, 1), verbose=1, method='trf')
    LS_result = Result1.x
    t_used_LS_only = time.time() - t_start

    real_parms = np.zeros(shape=(1, 4))
    predicted_parms = np.zeros(shape=(1, 4))
    predicted_parms_LS = np.zeros(shape=(1, 4))
    LS_parms = np.zeros(shape=(1, 4))

    real_parms[0, 0] = eps_min + (eps_max - eps_min) * parameters[k-1, 0]  ## get the true parameters
    real_parms[0, 1] = wmin + (wmax - wmin) * parameters[k-1, 1]
    real_parms[0, 2] = gamma_min + (gamma_max - gamma_min) * parameters[k-1, 2]
    real_parms[0, 3] = wmin + (wmax - wmin) * parameters[k-1, 3]

    predicted_parms[0, 0] = eps_min + (eps_max - eps_min) * predicted_parms1[0]  ## get the ANN predicted parameters
    predicted_parms[0, 1] = wmin + (wmax - wmin) * predicted_parms1[1]
    predicted_parms[0, 2] = gamma_min + (gamma_max - gamma_min) * predicted_parms1[2]
    predicted_parms[0, 3] = wmin + (wmax - wmin) * predicted_parms1[3]

    predicted_parms_LS[0, 0] = eps_min + (eps_max - eps_min) * predicted_parms_LS_1[0]  ## get the iHNN predicted values
    predicted_parms_LS[0, 1] = wmin + (wmax - wmin) * predicted_parms_LS_1[1]
    predicted_parms_LS[0, 2] = gamma_min + (gamma_max - gamma_min) * predicted_parms_LS_1[2]
    predicted_parms_LS[0, 3] = wmin + (wmax - wmin) * predicted_parms_LS_1[3]

    LS_parms[0, 0] = eps_min + (eps_max - eps_min) * LS_result[0]  ## get the pure least squares predicted values
    LS_parms[0, 1] = wmin + (wmax - wmin) * LS_result[1]
    LS_parms[0, 2] = gamma_min + (gamma_max - gamma_min) * LS_result[2]
    LS_parms[0, 3] = wmin + (wmax - wmin) * LS_result[3]

    Real_Parameters[k-1, :] = real_parms[0]  ## store the test instance
    Parameters_ML[k-1, :] = predicted_parms[0]
    Parameters_LS[k-1, :] = predicted_parms_LS[0]
    LS_only[k-1, :] = LS_parms[0]
    Time_ML[k - 1, :] = t_used_ML
    Time_LS[k - 1, :] = t_used_LS
    Time_LS_only[k - 1, :] = t_used_LS_only


## store the entire dataset ##
hf = h5py.File('ML_spheroid_paper_radiative_clean_25a_relu_Adam_0.0001_1.h5', 'w')
hf.create_dataset('R_P', data=Real_Parameters)
hf.create_dataset('P_ML', data=Parameters_ML)
hf.create_dataset('P_LS', data=Parameters_LS)
hf.create_dataset('LS_only', data=LS_only)
hf.create_dataset('T_ML', data=Time_ML)
hf.create_dataset('T_LS', data=Time_LS)
hf.create_dataset('T_lS_only', data=Time_LS_only)
hf.close()
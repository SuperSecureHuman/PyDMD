"""
Variable Projection for DMD. Reformulation of original paper
(https://epubs.siam.org/doi/abs/10.1137/M1124176) s.t. sparse matrix computation
is substiuted by outer products. Further the optimization is
reformulated s.t. SciPy's nonlinear least squares optimizer
can handle "complex" parameters.

Default optimizer arguments:
    OPT_DEF_ARGS =
        {"method": "trf",
        "tr_solver": "exact",
        "loss": "linear",
        "x_scale": "jac",
        "gtol": 1e-8,
        "xtol": 1e-8,
        "ftol": 1e-8}
"""

import warnings
from typing import Any, Dict, Tuple, Union
import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.linalg import qr
from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots


OPT_DEF_ARGS: Dict[str, Any] = {  # pylint: disable=unused-variable
    "method": "trf",
    "tr_solver": "exact",
    "loss": "linear",
    "x_scale": "jac",
    "gtol": 1e-8,
    "xtol": 1e-8,
    "ftol": 1e-8,
}


def __svht(
    sigma_svd: np.ndarray,  # pylint: disable=unused-variable
    rows: int,
    cols: int,
    sigma: float = None,
) -> int:
    """
    Determine optimal rank for svd matrix approximation,
    based on https://arxiv.org/pdf/1305.5870.pdf.

    :param sigma_svd:  Diagonal matrix of "enonomy" SVD as 1d array.
    :type sigma_svd: np.ndarray
    :param rows: Number of rows of original data matrix.
    :type cols: int
    :param cols: Number of columns of original data matrix.
    :type cols: int
    :param sigma: Signal noise level if known, defaults to None.
    :type sigma: float, optional
    :raises ValueError: sigma_svd must be a 1d-array else exception is raised.
    :return: Optimal rank.
    :rtype: int
    """

    if len(sigma_svd.shape) != 1:
        raise ValueError("Expected 1d array for diagonal matrix!")

    beta = (
        float(cols) / float(rows) if rows > cols else float(rows) / float(cols)
    )
    tau_star = 0

    if sigma is not None:
        sigma = abs(sigma)
        lambda_star = np.sqrt(
            2 * (beta + 1)
            + (8 * beta / (beta + 1 + np.sqrt(beta * beta + 14 * beta + 1)))
        )
        tau_star = (
            lambda_star
            * sigma
            * np.sqrt(float(cols) if cols > rows else float(rows))
        )
    else:
        median = np.median(sigma_svd)
        beta_square = beta * beta
        beta_cubic = beta * beta_square
        omega = 0.56 * beta_cubic - 0.95 * beta_square + 1.82 * beta + 1.43
        tau_star = median * omega

    rank = np.where(sigma_svd >= tau_star)[0]
    r_out = 0
    if rank.size == 0:
        r_out = sigma_svd.shape[-1]

    else:
        r_out = rank[-1] + 1
    return r_out


def __compute_rank(
    sigma_x: np.ndarray,
    rows: int,
    cols: int,
    rank: Union[float, int],
    sigma: float = None,
) -> int:
    r"""
    Compute rank without duplicate SVD computation.

    :param sigma_x: Diagonal matrix of SVD as 1d array.
    :type sigma_x: np.ndarray
    :param rows: Number of rows of the original data matrix.
    :type rows: int
    :param cols: Number of columns of the original data matrix.
    :type cols: int
    :param rank: Desired input rank :math:`r`. If rank is an integer,
        and :math:`r > 0`, the desired rank is used iff possible.
        This depends on the size of parameter sigma_x. If the desired
        rank exceeds the size of sigma_x, then the minimum of desired rank
        and the size of sigma_x is used. If the rank is a float
        and :math:`0 < r < 1`, than the rank is determined by investigating
        the cumulative energy of the singular values.
    :type rank: Union[float, int]
    :param sigma: Noise level for SVHT algorithm. If it is unknown
        keep it as None. Defaults to None.
    :type sigma: float, optional
    :raises ValueError: ValueError is raised if rank is negative
        (:math:`r < 0`).
    :return: Optimal rank.
    :rtype: int
    """

    if 0 < rank < 1:
        cumulative_energy = np.cumsum(
            np.square(sigma_x) / np.square(sigma_x).sum()
        )
        __rank = np.searchsorted(cumulative_energy, rank) + 1
    elif rank == 0:
        __rank = __svht(sigma_x, rows, cols, sigma)
    elif rank >= 1:
        __rank = int(rank)
    else:
        raise ValueError(f"Invalid rank specified, provided {rank}!")
    return min(__rank, sigma_x.size)


def __compute_dmd_ev(
    x_current: np.ndarray,  # pylint: disable=unused-variable
    x_next: np.ndarray,
    rank: Union[float, int] = 0,
) -> np.ndarray:
    r"""
    Compute DMD eigenvalues.

    :param x_current: Observables :math:`\boldsymbol{X}`
    :type x_current: np.ndarray
    :param x_next: Observables :math:`\boldsymbol{X}'`
    :type x_current: np.ndarray
    :param rank: Desired rank. If rank :math:`r = 0`, the optimal rank is
        determined automatically. If rank is a float s.t. :math: `0 < r < 1`,
        the cumulative energy of the singular values is used
        to determine the optimal rank. If rank is an integer and :math:`r > 0`,
        the desired rank is used iff possible. Defaults to 0.
    :type rank: Union[float, int], optional
    :return: Diagonal matrix of eigenvalues
        :math:`\boldsymbol{\Lambda}` as 1d array.
    :rtype: np.ndarray
    """

    u_x, sigma_x, v_x_t = np.linalg.svd(
        x_current, full_matrices=False, hermitian=False
    )
    __rank = __compute_rank(
        sigma_x, x_current.shape[0], x_current.shape[1], rank
    )

    # columns of v need to be multiplicated with inverse sigma
    sigma_inv_approx = np.reciprocal(sigma_x[:__rank])

    a_approx = np.linalg.multi_dot(
        [
            u_x[:, :__rank].conj().T,
            x_next,
            sigma_inv_approx.reshape(1, -1) * v_x_t[:__rank, :].conj().T,
        ]
    )

    return np.linalg.eigvals(a_approx)


class __OptimizeHelper:
    """
    Helper Class to store intermediate results during the optimization.
    """

    __slots__ = ["phi", "phi_inv", "u_svd", "s_inv", "v_svd", "b_matrix", "rho"]

    def __init__(self, l_in: int, m_in: int, n_in: int) -> None:
        self.phi: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.u_svd: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.s_inv: np.ndarray = np.empty((l_in,), dtype=np.complex128)
        self.v_svd: np.ndarray = np.empty((l_in, l_in), dtype=np.complex128)
        self.b_matrix: np.ndarray = np.empty((l_in, n_in), dtype=np.complex128)
        self.rho: np.ndarray = np.empty((m_in, n_in), dtype=np.complex128)


def __compute_dmd_rho(
    alphas: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: __OptimizeHelper,
) -> np.ndarray:
    r"""
    Compute the real residual vector :math:`\boldsymbol{\rho}` for
    Levenberg-Marquardt update.

    :param alphas: DMD eigenvalues to optimize,
        where :math:`\alpha \in \mathbb{C}^l`,
        but here :math:`\alpha \in \mathbb{R}^{2l}`,
        since optimizer cannot deal with complex numbers.
    :type alphas: np.ndarray
    :param time: 1D time array.
    :type time: np.ndarray
    :param data: data :math:`\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian.
    :type opthelper: __OptimizeHelper
    :return: 1d residual vector for Levenberg-Marquardt update
        :math:`\boldsymbol{\rho} \in \mathbb{R}^{2mn}`.
    :rtype: np.ndarray
    """

    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    __alphas.real = alphas[: alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2 :]

    phi = np.exp(np.outer(time, __alphas))
    __u, __s, __v_t = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    __idx = np.where(__s.real != 0.0)[0]
    __s_inv = np.zeros_like(__s)
    __s_inv[__idx] = np.reciprocal(__s[__idx])

    rho = data - np.linalg.multi_dot([__u, __u.conj().T, data])
    rho_flat = np.ravel(rho)
    rho_out = np.zeros((2 * rho_flat.shape[-1],), dtype=np.float64)
    rho_out[: rho_flat.shape[-1]] = rho_flat.real
    rho_out[rho_flat.shape[-1] :] = rho_flat.imag

    opthelper.phi = phi
    opthelper.u_svd = __u
    opthelper.s_inv = __s_inv
    opthelper.v_svd = __v_t.conj().T
    opthelper.rho = rho
    opthelper.b_matrix = np.linalg.multi_dot(
        [
            opthelper.v_svd * __s_inv.reshape((1, -1)),
            opthelper.u_svd.conj().T,
            data,
        ]
    )
    return rho_out


def __compute_dmd_jac(
    alphas: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: __OptimizeHelper,
) -> np.ndarray:
    r"""
    Compute the real Jacobian.
    SciPy's nonlinear least squares optimizer requires real entities.
    Therefore, complex and real parts are split.

    :param alphas: DMD eigenvalues to optimize,
        where :math:`\alpha \in \mathbb{C}^l`,
        but here :math:`\alpha \in \mathbb{R}^{2l}` since optimizer cannot
        deal with complex numbers.
    :type alphas: np.ndarray
    :param time: 1D time array.
    :type time: np.ndarray
    :param data: data :math: `\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian. The entities are computed in `__compute_dmd_rho`.
    :type opthelper: __OptimizeHelper
    :return: Jacobian :math:`\boldsymbol{J} \in \mathbb{R}^{mn \times 2l}`.
    :rtype: np.ndarray
    """

    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    __alphas.real = alphas[: alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2 :]
    jac_out = np.zeros((2 * np.prod(data.shape), alphas.shape[-1]))

    for j in range(__alphas.shape[-1]):
        d_phi_j = time * opthelper.phi[:, j]
        __outer = np.outer(d_phi_j, opthelper.b_matrix[j, :])
        __a_j = __outer - np.linalg.multi_dot(
            [opthelper.u_svd, opthelper.u_svd.conj().T, __outer]
        )
        __g_j = np.linalg.multi_dot(
            [
                opthelper.u_svd * opthelper.s_inv.reshape((1, -1)),
                np.outer(
                    opthelper.v_svd[j, :].conj(), d_phi_j.conj() @ opthelper.rho
                ),
            ]
        )
        # Compute the jacobian J_mat_j = - (A_j + G_j).
        __jac = -__a_j - __g_j
        __jac_flat = np.ravel(__jac)

        # construct the overall jacobian for optimized
        # J_real = |Re{J} -Im{J}|
        #          |Im{J}  Re{J}|

        # construct real part for optimization
        jac_out[: jac_out.shape[0] // 2, j] = __jac_flat.real
        jac_out[jac_out.shape[0] // 2 :, j] = __jac_flat.imag

        # construct imaginary part for optimization
        jac_out[
            : jac_out.shape[0] // 2, __alphas.shape[-1] + j
        ] = -__jac_flat.imag
        jac_out[
            jac_out.shape[0] // 2 :, __alphas.shape[-1] + j
        ] = __jac_flat.real

    return jac_out


def __compute_dmd_varpro(
    alphas_init: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    opthelper: __OptimizeHelper,
    **optargs,
) -> OptimizeResult:
    r"""
    Compute Variable Projection (VarPro) for DMD using SciPy's
    nonlinear least squares optimizer.

    :type alphas_init: np.ndarray
    :param time: 1d time array.
    :type time: np.ndarray
    :param data: data :math:`\boldsymbol{Y} \n C^{m \times n}`.
        For DMD computation we set :math:`\boldsymbol{Y} = \boldsymbol{X}^T`.
    :type data: np.ndarray
    :param opthelper: Optimization helper to speed up computations
        mainly for Jacobian. The entities are computed in `__compute_dmd_rho`
        and are used in `__compute_dmd_jac`.
    :type opthelper: __OptimizeHelper
    :return: Optimization result.
    :rtype: OptimizeResult
    """

    return least_squares(
        __compute_dmd_rho,
        alphas_init,
        __compute_dmd_jac,
        **optargs,
        args=[time, data, opthelper],
    )


def select_best_samples_fast(data: np.ndarray, comp: float = 0.9) -> np.ndarray:
    r"""
    Select library samples using QR decomposition with column pivoting.

    :param data: Data matrix :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`.
    :type data: np.ndarray
    :param comp: Library compression :math:`c`, where :math:`0 < c < 1`.
        The best fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
        are selected. Defaults to 0.9.
    :type comp: float, optional
    :raises ValueError: ValueError is raised if data matrix is not a
        2d array.
    :raises ValueError: ValueError is raised of compression is not in required
        interval (:math:`0 < r < 1`).
    :return: Indices of selected samples as 1d array.
    :rtype: np.ndarray
    """

    if len(data.shape) != 2:
        raise ValueError("Expected 2D array!")

    if not 0 < comp < 1:
        raise ValueError("Compression must be in (0, 1)]")

    n_samples = int(data.shape[-1] * (1.0 - comp))
    pcolumn = qr(data, mode="economic", pivoting=True)[-1]
    __idx = pcolumn[:n_samples]
    return __idx


def compute_varprodmd_any(  # pylint: disable=unused-variable
    data: np.ndarray,
    time: np.ndarray,
    optargs: Dict[str, Any],
    rank: Union[float, int] = 0.0,
    use_proj: bool = True,
    compression: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OptimizeResult]:
    r"""
    Compute DMD given arbitary timesteps.

    :param data: data matrix s.t. :math:`X \n C^{n \times m}`.
    :type data: np.ndarray
    :param time: 1d array of timestamps.
    :type time: np.ndarray
    :param optargs: Arguments for 'least_squares' optimizer.
    :type optargs: Dict[str, Any]
    :param rank: Desired rank. If rank :math:`r = 0`, the optimal rank is
        determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
        the cumulative energy of the singular values is used
        to determine the optimal rank. If rank is an integer
        and :math:`r > 0`, the desired rank is used iff possible.
        Defaults to 0.
    :type rank: Union[float, int], optional
    :param use_proj: Perform variable projection in
        low dimensional space if `use_proj=True`, else in the original space.
        Defaults to True.
    :type use_proj: bool, optional
    :param compression: If libary compression :math:`c = 0`,
        all samples are used. If :math:`0 < c < 1`, the best
        fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
        are selected.
    :type compression: float, optional
    :raises ValueError: ValueError is raised if data matrix is not a
        2d array.
    :raises ValueError: ValueError is raised if time is not a
        1d array.
    :return: DMD modes :math:`\boldsymbol{\Phi}`, continuous DMD eigenvalues
        :math: `\boldsymbol{\Omega}` as 1d array,
        DMD eigenfunctions or amplitudes :math:`\boldsymbol{\varphi}`,
        optimization results of SciPy's nonlinear least squares optimizer.
    :rtype: Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  np.ndarray,
                  OptimizeResult]
    """

    if len(data.shape) != 2:
        raise ValueError("data needs to be 2D array")

    if len(time.shape) != 1:
        raise ValueError("time needs to be a 1D array")

    u_r, s_r, v_r_t = np.linalg.svd(data, full_matrices=False)
    __rank = __compute_rank(s_r, data.shape[0], data.shape[1], rank)
    u_r = u_r[:, :__rank]
    s_r = s_r[:__rank]
    v_r = v_r_t[:__rank, :].conj().T
    data_in = v_r.conj().T * s_r.reshape((-1, 1)) if use_proj else data

    # trapezoidal derivative approximation
    y_in = (data_in[:, :-1] + data_in[:, 1:]) / 2.0
    dt_in = time[1:] - time[:-1]
    z_in = (data_in[:, 1:] - data_in[:, :-1]) / dt_in.reshape((1, -1))
    omegas = __compute_dmd_ev(y_in, z_in, __rank)
    omegas_in = np.zeros((2 * omegas.shape[-1],), dtype=np.float64)
    omegas_in[: omegas.shape[-1]] = omegas.real
    omegas_in[omegas.shape[-1] :] = omegas.imag

    if compression > 0:
        __idx = select_best_samples_fast(data_in, compression)
        if __idx.size > 1:
            indices = __idx
        else:
            indices = np.arange(data_in.shape[-1])
        data_in = data_in[:, indices]
        time_in = time[indices]

    else:
        time_in = time
        indices = np.arange(data_in.shape[-1])

    if data_in.shape[-1] < omegas.shape[-1]:
        msg = "Attempting to solve underdeterimined system. "
        msg += "Decrease desired rank or compression!"
        warnings.warn(msg)

    opthelper = __OptimizeHelper(u_r.shape[-1], *data_in.shape)
    opt = __compute_dmd_varpro(
        omegas_in, time_in, data_in.T, opthelper, **optargs
    )
    omegas.real = opt.x[: opt.x.shape[-1] // 2]
    omegas.imag = opt.x[opt.x.shape[-1] // 2 :]
    xi = u_r @ opthelper.b_matrix.T if use_proj else opthelper.b_matrix.T
    eigenf = np.linalg.norm(xi, axis=0)
    return xi / eigenf.reshape((1, -1)), omegas, eigenf, indices, opt


def optdmd_predict(  # pylint: disable=unused-variable
    phi: np.ndarray,
    omegas: np.ndarray,
    eigenf: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    r"""
    Perform DMD prediction using computed modes, continuous eigenvalues
    and eigenfunctions/amplitudes.

    :param phi: DMD modes
        :math:`\boldsymbol{\Phi} \in \mathbb{C}^{n \times \left(m-1\right)}`.
    :type phi: np.ndarray
    :param omegas: Continuous diagonal matrix of eigenvalues
        :math:`\boldsymbol{\Omega} \in 
            \mathbb{C}^{\left(m-1\right) \times \left(m-1\right)}`
        as 1d array.
    :type omegas: np.ndarray
    :param eigenf: Eigenfunctions or amplitudes
        :math:`\boldsymbol{\varphi} \in \mathbb{C}^{m - 1}`.
    :type eigenf: np.ndarray
    :param time: 1d array of timestamps.
    :type time: np.ndarray
    :return: Reconstructed/predicted state :math:`\hat{\boldsymbol{X}}`.
    :rtype: np.ndarray
    """

    return phi @ (np.exp(np.outer(omegas, time)) * eigenf.reshape(-1, 1))


class VarProOperator(DMDOperator):
    """
    Variable Projection Operator for DMD.
    """

    def __init__(
        self,
        svd_rank: Union[float, int],
        exact: bool,
        sorted_eigs: Union[bool, str],
        compression: float,
        optargs: Dict[str, Any],
    ):
        r"""
        VarProOperator constructor.

        :param svd_rank: Desired SVD rank.
            If rank :math:`r = 0`, the optimal rank is
            determined automatically. If rank is a float s.t.
            :math:`0 < r < 1`, the cumulative energy
            of the singular values is used to determine
            the optimal rank.
            If rank is an integer and :math:`r > 0`,
            the desired rank is used iff possible.
        :type svd_rank: Union[float, int]
        :param exact: Perform computations in original state space
            if `exact=True`, else perform optimization
            in projected (low dimensional) space.
        :type exact: bool
        :param sorted_eigs: Sort eigenvalues.
            If `sorted_eigs=True`, the variance of the absolute values
            of the complex eigenvalues
            :math:`\sqrt{\omega_i \cdot \bar{\omega}_i}`,
            the variance absolute values of the real parts
            :math:`\left|\Re\{{\omega_i}\}\right|`
            and the variance of the absolute values
            of the imaginary parts :math:`\left|\Im\{{\omega_i}\}\right|`
            is computed.
            The eigenvalues are then sorted according
            to the highest variance (from highest to lowest).
            If `sorted_eigs=False`, no sorting is performed.
            If the parameter is a string and set to sorted_eigs='auto',
            the eigenvalues are sorted accoring to the variances
            of previous mentioned quantities.
            If `sorted_eigs='real'` the eigenvalues are sorted
            w.r.t. the absolute values of the real parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='imag'` the eigenvalues are sorted
            w.r.t. the absolute values of the imaginary parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='abs'` the eigenvalues are sorted
            w.r.t. the magnitude of the eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`
            (from highest to lowest).
        :type sorted_eigs: Union[bool, str]
        :param compression: If libary compression :math:`c = 0`,
            all samples are used. If :math:`0 < c < 1`, the best
            fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
            are selected.
        :type compression: float
        :param optargs: Arguments for 'least_squares' optimizer.
            Use `OPT_DEF_ARGS` as starting point.
        :type optargs: Dict[str, Any]
        """

        super().__init__(svd_rank, exact, False, None, sorted_eigs, False)
        self._sorted_eigs = sorted_eigs
        self._svd_rank = svd_rank
        self._exact = exact
        self._optargs: Dict[str, Any] = optargs
        self._compression: float = compression
        self._modes: np.ndarray = None
        self._eigenvalues: np.ndarray = None

    def compute_operator(
        self, X: np.ndarray, time: np.ndarray
    ) -> Tuple[np.ndarray, OptimizeResult, np.ndarray]:
        r"""
        Perform Variable Projection for DMD using SciPy's
        nonlinear least squares optimizer.

        :param X: Measurement :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`
        :type X: np.ndarray
        :param time: 1d array of timestamps where individual
            measurements :math:`\boldsymbol{x}_i \in \mathbb{C}^n`
            where taken.
        :type time: np.ndarray
        :raises ValueError: If `sorted_eigs` from constructor
            was set to an invalid string.
        :return: Eigenfunctions/amplitudes :math:`\boldsymbol{\varphi}^{m-1}`,
            OptimizeResult from SciPy's optimizer
            (optimal parameters and statistics),
            indices of selected measurements. If no compression (:math:`c = 0`)
            is used, all indices are returned, else the indices of the selected
            samples are used.
        :rtype: Tuple[np.ndarray, OptimizeResult, np.ndarray]
        """
        (
            self._modes,
            self._eigenvalues,
            eigenf,
            indices,
            opt,
        ) = compute_varprodmd_any(
            X,
            time,
            self._optargs,
            self._svd_rank,
            not self._exact,
            self._compression,
        )

        # overwrite for lazy sorting
        if isinstance(self._sorted_eigs, bool):
            if self._sorted_eigs:
                self._sorted_eigs = "auto"

        if isinstance(self._sorted_eigs, str):
            if self._sorted_eigs == "auto":
                eigs_real = self._eigenvalues.real
                eigs_imag = self._eigenvalues.imag
                __eigs_abs = np.abs(self._eigenvalues)
                var_real = np.var(eigs_real)
                var_imag = np.var(eigs_imag)
                var_abs = np.var(__eigs_abs)
                __array = np.array([var_real, var_imag, var_abs])
                eigs_abs = (eigs_real, eigs_imag, __eigs_abs)[
                    np.argmax(__array)
                ]

            elif self._sorted_eigs == "real":
                eigs_abs = np.abs(self._eigenvalues.real)

            elif self._sorted_eigs == "imag":
                eigs_abs = np.abs(self._eigenvalues.imag)

            elif self._sorted_eigs == "abs":
                eigs_abs = np.abs(self._eigenvalues)
            else:
                raise ValueError(f"{self._sorted_eigs} not supported!")

            idx = np.argsort(eigs_abs)[::-1]  # sort from biggest to smallest
            self._eigenvalues = self._eigenvalues[idx]
            self._modes = self._modes[:, idx]
            eigenf = eigenf[idx]

        return eigenf, opt, indices


class VarProDMD(DMDBase):
    """
    Variable Projection for DMD using SciPy's
    nonlinear least squares solver. The original problem
    is reformulated s.t. complex residuals and Jacobians, which are used
    by the Levenberg-Marquardt algorithm, are transormed to real numbers.
    Further simplifications (outer products) avoids using sparse matrices.
    """

    def __init__(
        self,
        svd_rank: Union[float, int] = 0,
        exact: bool = False,
        sorted_eigs: Union[bool, str] = False,
        compression: float = 0.0,
        optargs: Dict[str, Any] = None,
    ):
        r"""
        VarProDMD constructor.

        :param svd_rank: Desired SVD rank.
            If rank :math:`r = 0`, the optimal rank is
            determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
            the cumulative energy of the singular values is used
            to determine the optimal rank.
            If rank is an integer and :math:`r > 0`,
            the desired rank is used iff possible. Defaults to 0.
        :type svd_rank: Union[float, int], optional
        :param exact: Perform variable projection in
            low dimensional space if `exact=False`.
            Else the optimization is performed
            in the original space.
            Defaults to False.
        :type exact: bool, optional
        :param sorted_eigs: Sort eigenvalues.
            If `sorted_eigs=True`, the variance of the absolute values
            of the complex eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`,
            the variance absolute values of the real parts
            :math:`\left|\Re\{{\omega_i}\}\right|`
            and the variance of the absolute values of the imaginary parts
            :math:`\left|\Im\{{\omega_i}\}\right|` is computed.
            The eigenvalues are then sorted according
            to the highest variance (from highest to lowest).
            If `sorted_eigs=False`, no sorting is performed.
            If the parameter is a string and set to sorted_eigs='auto',
            the eigenvalues are sorted accoring to the variances
            of previous mentioned quantities.
            If `sorted_eigs='real'` the eigenvalues are sorted
            w.r.t. the absolute values of the real parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='imag'` the eigenvalues are sorted
            w.r.t. the absolute values of the imaginary parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='abs'` the eigenvalues are sorted
            w.r.t. the magnitude of the eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`
            (from highest to lowest).
            Defaults to False.
        :type sorted_eigs: Union[bool, str], optional
        :param compression: If libary compression :math:`c = 0`,
            all samples are used. If :math:`0 < c < 1`, the best
            fitting :math:`\lfloor \left(1 - c\right)m\rfloor` samples
            are selected.
        :type compression: float, optional
        :param optargs: Arguments for 'least_squares' optimizer.
            If set to None, `OPT_DEF_ARGS` are used as default parameters.
            Defaults to None.
        :type optargs: Dict[str, Any], optional
        """

        if optargs is None:
            optargs = OPT_DEF_ARGS

        self._Atilde = VarProOperator(
            svd_rank, exact, sorted_eigs, compression, optargs
        )
        self._optres: OptimizeResult = None
        self._snapshots_holder: Snapshots = None
        self._indices: np.ndarray = None
        self._modes_activation_bitmask_proxy = None

    def fit(self, X: np.ndarray, time: np.ndarray) -> object:
        r"""
        Fit the eigenvalues, modes and eigenfunctions/amplitudes
        to measurements.

        :param X: Measurements
            :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`.
        :type X: np.ndarray
        :param time: 1d array of timestamps where measurements were taken.
        :type time: np.ndarray
        :return: VarProDMD instance.
        :rtype: object
        """

        self._snapshots_holder = Snapshots(X)
        self._b, self._optres, self._indices = self._Atilde.compute_operator(
            X, time
        )
        self._original_time = time
        self._dmd_time = time[self._indices]

        return self

    def forecast(self, time: np.ndarray) -> np.ndarray:
        r"""
        Forecast measurements at given timestamps `time`.

        :param time: Desired times for forcasting as 1d array.
        :type time: np.ndarray
        :raises ValueError: If method `fit(X, time)` was not called.
        :return: Predicted measurements :math:`\hat{\boldsymbol{X}}`.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError(
                "Nothing fitted yet, need to call fit-method first!"
            )

        return optdmd_predict(
            self._Atilde.modes, self._Atilde.eigenvalues, self._b, time
        )

    @property
    def ssr(self) -> float:
        """
        Compute the square root of sum squared residual (SSR) taken from
        https://link.springer.com/article/10.1007/s10589-012-9492-9.
        The SSR gives insight w.r.t. signal qualities.
        A low SSR is desired. If SSR is high the model may be inaccurate.

        :raises ValueError: ValueError is raised if method
            `fit(X, time)` was not called.
        :return: SSR.
        :rtype: float
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        rho_flat_real = self._optres.fun
        rho_flat_imag = np.zeros(
            (rho_flat_real.size // 2,), dtype=np.complex128
        )
        rho_flat_imag.real = rho_flat_real[: rho_flat_real.size // 2]
        rho_flat_imag.imag = rho_flat_real[rho_flat_real.size // 2 :]

        sigma = np.linalg.norm(rho_flat_imag)
        denom = max(
            self._original_time.size
            - self._optres.jac.shape[0] // 2
            - self._optres.jac.shape[1] // 2,
            1,
        )
        ssr = sigma / np.sqrt(float(denom))

        return ssr

    @property
    def selected_samples(self) -> np.ndarray:
        r"""
        Return indices for creating the library.

        :raises ValueError: ValueError is raised if method 
            `fit(X, time)` was not called.
        :return: Indices of the selected samples.
            If no compression was performed :math:`\left(c = 0\right)`,
            all indices are returned, else indices of the
            library selection scheme using QR-Decomposition
            with column pivoting.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError(
                "Nothing fitted yet, need to call fit-method first!"
            )

        return self._indices

    @property
    def opt_stats(self) -> OptimizeResult:
        """
        Return optimization statistics of the Variable Projection
        optimization.

        :raises ValueError: ValueError is raised if method `fit(X, time)`
            was not called.
        :return: Optimization results including optimal weights
            (continuous eigenvalues) and number of iterations.
        :rtype: OptimizeResult
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return self._optres

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: matrix that contains all the time evolution, stored by row.
        :rtype: numpy.ndarray
        """

        t_omega = np.exp(np.outer(self.eigs, self._original_time))
        return self.amplitudes.reshape(-1, 1) * t_omega

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """

        return self.eigs.imag / (2 * np.pi)

    @property
    def growth_rate(self):
        """
        Get the growth rate values relative to the modes.

        :return: the Floquet values
        :rtype: numpy.ndarray
        """

        return self.eigs.real

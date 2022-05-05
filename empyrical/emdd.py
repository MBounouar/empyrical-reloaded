import numpy as np
from scipy import interpolate


def expected_max_drawdown(
    mu: float, sigma: float, t: float, gbm: bool = False
) -> float:
    """
    Determines the expected maximum drawdown of a brownian motion,
    given drift and diffusion
    If a geometric Brownian motion with stochastic differential equation
        dS(t) = Mu0 * S(t) * dt + Sigma0 * S(t) * dW(t) ,
    it converts to the form here by Ito's Lemma with X(t) = log(S(t)) such that
        Mu = Mu0 - 0.5 * Sigma0^2
        Sigma = Sigma0 .

    Parameters
    ----------
    mu : float
        The drift term of a Brownian motion with drift.
    sigma : float
        The diffusion term of a Brownian motion with drift.
    t : float
        A time period of interest
    gbp : bool
        If true, compute for geometric brownian motion

    Returns
    -------
    expected_max_drawdown : float

    Note
    -----
    See http://www.cs.rpi.edu/~magdon/ps/journal/drawdown_journal.pdf
    for more details.
    """

    sigma2 = np.power(sigma, 2)
    if gbm:
        mu = mu - 0.5 * sigma2

    def emdd_q(x: float, p: bool = True):
        """Q function based on lookup table"""

        if x < 0.0005:
            return 0.5 * np.sqrt(np.pi * x)

        if x > 5000:
            return 0.25 * np.log(x) + 0.4908

        if p:
            A = [
                0.0005,
                0.001,
                0.0015,
                0.002,
                0.0025,
                0.005,
                0.0075,
                0.01,
                0.0125,
                0.015,
                0.0175,
                0.02,
                0.0225,
                0.025,
                0.0275,
                0.03,
                0.0325,
                0.035,
                0.0375,
                0.04,
                0.0425,
                0.045,
                0.0475,
                0.05,
                0.055,
                0.06,
                0.065,
                0.07,
                0.075,
                0.08,
                0.085,
                0.09,
                0.095,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                500.0,
                1000.0,
                1500.0,
                2000.0,
                2500.0,
                3000.0,
                3500.0,
                4000.0,
                4500.0,
                5000.0,
            ]
            B = [
                0.01969,
                0.027694,
                0.033789,
                0.038896,
                0.043372,
                0.060721,
                0.073808,
                0.084693,
                0.094171,
                0.102651,
                0.110375,
                0.117503,
                0.124142,
                0.130374,
                0.136259,
                0.141842,
                0.147162,
                0.152249,
                0.157127,
                0.161817,
                0.166337,
                0.170702,
                0.174924,
                0.179015,
                0.186842,
                0.194248,
                0.201287,
                0.207999,
                0.214421,
                0.220581,
                0.226505,
                0.232212,
                0.237722,
                0.24305,
                0.288719,
                0.325071,
                0.355581,
                0.382016,
                0.405415,
                0.426452,
                0.445588,
                0.463159,
                0.588642,
                0.668992,
                0.72854,
                0.775976,
                0.815456,
                0.849298,
                0.878933,
                0.905305,
                0.92907,
                1.088998,
                1.184918,
                1.253794,
                1.307607,
                1.351794,
                1.389289,
                1.42186,
                1.450654,
                1.476457,
                1.647113,
                1.747485,
                1.818873,
                1.874323,
                1.919671,
                1.958037,
                1.991288,
                2.02063,
                2.046885,
                2.219765,
                2.320983,
                2.392826,
                2.448562,
                2.494109,
                2.532622,
                2.565985,
                2.595416,
                2.621743,
            ]

        else:
            if x > 5:
                return x + 0.5

            A = [
                0.0005,
                0.001,
                0.0015,
                0.002,
                0.0025,
                0.005,
                0.0075,
                0.01,
                0.0125,
                0.015,
                0.0175,
                0.02,
                0.0225,
                0.025,
                0.0275,
                0.03,
                0.0325,
                0.035,
                0.0375,
                0.04,
                0.0425,
                0.045,
                0.0475,
                0.05,
                0.055,
                0.06,
                0.065,
                0.07,
                0.075,
                0.08,
                0.085,
                0.09,
                0.095,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
                2.25,
                2.5,
                2.75,
                3.0,
                3.25,
                3.5,
                3.75,
                4.0,
                4.25,
                4.5,
                4.75,
                5.0,
            ]
            B = [
                0.019965,
                0.028394,
                0.034874,
                0.040369,
                0.045256,
                0.064633,
                0.079746,
                0.092708,
                0.104259,
                0.114814,
                0.124608,
                0.133772,
                0.142429,
                0.150739,
                0.158565,
                0.166229,
                0.173756,
                0.180793,
                0.187739,
                0.194489,
                0.201094,
                0.207572,
                0.213877,
                0.220056,
                0.231797,
                0.243374,
                0.254585,
                0.265472,
                0.27607,
                0.286406,
                0.296507,
                0.306393,
                0.316066,
                0.325586,
                0.413136,
                0.491599,
                0.564333,
                0.633007,
                0.698849,
                0.762455,
                0.824484,
                0.884593,
                1.17202,
                1.44552,
                1.70936,
                1.97074,
                2.22742,
                2.48396,
                2.73676,
                2.99094,
                3.24354,
                3.49252,
                3.74294,
                3.99519,
                4.24274,
                4.49238,
                4.73859,
                4.99043,
                5.24083,
                5.49882,
            ]

        return interpolate.interp1d(A, B)([x])[0]

    if (not np.isfinite(mu)) | (not np.isfinite(sigma)) | (sigma <= 0):
        return np.nan

    if mu == 0:
        return np.sqrt(np.pi / 2) * sigma * np.sqrt(t)

    alpha = (2 * sigma2) / mu
    x = (np.power(mu, 2) * t) / (2 * sigma2)
    Q = emdd_q(x, p=True if mu > 0 else False)
    return np.sign(mu) * alpha * Q


if __name__ == "__main__":

    print(expected_max_drawdown(mu=0, sigma=1, t=1000, gbm=True))
    # print(
    #     expected_max_drawdown(
    #         mu=np.array([0]),
    #         sigma=np.array([1]),
    #         t=1000,
    #         gbm=False,
    #     )
    # )

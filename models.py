measure""" Some toys around COVID-19.

# @Author: maurotoro <soyunkope>
# @Date:   2020-04-05T23:28:45+01:00
# @Email:  mauricio.toro@neuro.fchampalimaud.org
# @Last modified by:   soyunkope
# @Last modified time: 2020-04-19T21:18:23+01:00
# @Copyright: Now you owe me a beer, or two, I choose!

Testing examples, SIR, SEIR and SEIRC models,
Heavily based on
    https://github.com/coronafighter/coronaSEIR
And
    https://hub.gke.mybinder.org/user/marcelloperathoner-covid-19-6lazscw9/notebooks/jupyter/seirc.ipynb

The baisc idea is that we don`t have control on the incubation period, nor in
the infection rate. We can only change how much interactions people have
between each other. To model this, one can make diffferent betas for diferent
time points.

For example:
    - If the goverment puts a quarantine 30 days after the first detected case.
    - The quarantine lasts 30 days.
    - After the quarantine people changed their behavior making it half the
      contagious it was originally. New normality.

    * Parameters:

        days = [30, 30]  # 30 days after first case lockdown, for 30 days.
        betas = [0.44, 0.165, 0.22]  # beta before lockdow, during and after.


TODO/FIXME
----------
    - [ ] Better estimation of Betas.
    - [ ] Add estimated deaths from recovered/removed.
    - [ ] Add available beds in ICUs as column, to be ploted always.
    - [ ] Prettyplots?

"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from datetime import datetime

plt.ion()


def SIR_model(init, t, params):
    """S(usceptible)I(nfected)R(recovered) model.

    Parameters
    ----------
    t : float
        Time at the current moment of the evolution of the system.
    init : list
        State of each compartment at time zero.
    params : list
        Parameters for the model
            `betas`: Contact rates * `contagiousness`.
            `gamma`: Infection rate. How long does infection lasts.
            `days` : Times of beta changing measures.
                     The values are how much each beta lasts.
            `N`    : Total population.

    Returns
    -------
    dS, dI, dR
        Rates of change per compartent by time step.

    """
    # Parameters for the model
    betas, gamma, days, N = params
    # differemt contact rates, lockdown measures count!
    beta = get_beta(t, days, betas)
    # compartments
    S, I, R = init
    # ODEs for the model
    dS = - beta * S * I / N
    dI = beta * I * S / N - gamma * I
    dR = gamma * I
    return dS, dI, dR


def SEIR_ode(init, t, params):
    """S(usceptible)E(xposed)I(nfected)R(recovered) model.

    Parameters
    ----------
    t : float
        Time at the current moment of the evolution of the system.
    init : list
        State of each compartment at time zero.
    params : list
        Parameters for the model
            `alpha`: infection rate. Inverse of incubation period.
            `betas`: Contact rates * `contagiousness`.
            `gamma`: Infection rate. How long does infection lasts.
            `days` : Times of beta changing measures.
                     The values are how much each beta lasts.
            `N`    : Total population.

    Returns
    -------
    dS, dE, dI, dR
        Rates of change per compartent by time step.

    """
    # Parameters for the model
    alpha, betas, gamma, days, N = params
    # differemt contact rates, lockdown measures count!
    beta = get_beta(t, days, betas)
    # compartments
    S, E, I, R = init
    # ODEs for the model
    dS = - beta * S * I / N
    dE = beta * S * I / N - alpha * E
    dI = alpha * E - gamma * I
    dR = gamma * I
    return dS, dE, dI, dR


def SEIRC_ode(init, t, params):
    """S(usceptible)E(xposed)I(nfected)R(recovered)C(ritical) model.

    Parameters
    ----------
    t : float
        Time at the current moment of the evolution of the system.
    init : list
        State of each compartment at time zero.
    params : list
        Parameters for the model
            `alpha`: infection rate. Inverse of incubation period.
            `betas`: Contact rates * `contagiousness`.
            `gamma`: Infection rate. How long does infection lasts.
            `c1`   : Ration of Infected that become critical.
            `c2`   : Time a critical patient usses a bed.
            `days` : Times of beta changing measures.
                     The values are how much each beta lasts.
            `N`    : Total population.

    Returns
    -------
    dS, dE, dI, dR, dC
        Rates of change per compartent by time step.

    """
    # Parameters for the model
    alpha, betas, gamma, c1, c2, days, N = params
    # differemt contact rates, lockdown measures count!
    beta = get_beta(t, days, betas)
    # compartments
    S, E, I, R, C = init
    # ODEs for the model
    dS = -beta * I / N * S
    dE = beta * I / N * S - alpha * E
    dI = alpha * E - gamma * I
    dR = gamma * I
    dC = c1 * alpha * E - c2 * C
    return dS, dE, dI, dR, dC


def model_solver(model_ode, init, params, start_date, labels='SEIRC'):
    """Solve the ODE for a compartmental epidemic model.

    Parameters
    ----------
    model_ode : function
        A functional representation for an ODE.
        Needs to have as parameters (init, t, params).
    init : list
        State of each compartment at time zero.
    params : list
        Parameters for the model
    start_date : list(int, int, int)
        Starting date for the model, the list must be [YEAR, MONTH, DAY].
    labels : str or list(str, str, str).
        Labels to append to the model results.

    Returns
    -------
    pandas.DataFrame
        Results of the integration of the model as a DataFrame.

    """
    # SIR model has less labels...
    if len(params) == 4:
        labels = 'SIR'
    t = np.linspace(0, 365, num=365*10 + 1)
    start_date_dt = datetime(*start_date).strftime('%Y/%m/%d')
    nxt_year = start_date[0] + 1
    end_date = [nxt_year] + start_date[1:]
    end_date_dt = datetime(*end_date).strftime('%Y/%m/%d')
    sol = odeint(model_ode, init, t, args=(params,))
    dt = pd.date_range(start=start_date_dt, end=end_date_dt,
                       periods=t.shape[0]).rename('time')
    return pd.DataFrame({k: v for v, k in zip(sol.T, labels)}, index=dt)


def get_beta(t, days, betas):
    """Return the beta values for the models.

    Assuming political and societal measures taken at diferent days, use
    this to model how they affect the contagions in the models.

    Parameters
    ----------
    t : float
        Time stamp of the current moment.
    days : list(int, ..., int)
        Days when measure changing betas happened. Each one is with respect
        to the previous event.
    betas : list(float, ..., float)
        Changes in the conctats between individuals at each measure.
        Must have one more than the days, the original contact rate.

    Returns
    -------
    float
        Beta for the current moment

    """
    # Check if there are different days
    if isinstance(days, int):
        # if only one day, return the beta asociated
        return betas[0] if t < days else betas[1]
    else:
        # Each beta will be kept for days, last one until end of simulation.
        days = days + [500 - sum(days)]
        # List of betas to use per day
        betas_by_day = np.hstack([np.ones(day).astype(int) * b
                                  for day, b in zip(days, betas)])
        # return the beta for today:
        return betas_by_day[int(t)]


if __name__ == '__main__':
    # Parameters for the models
    # Infection rate, incubation_period**-1
    alpha = 1/2
    # Contact rates * `contagiousness`, before, during lockdown, after lockdown
    betas = [0.45, 0.165, 0.22]
    betas_learn = [0.45, 0.165, 0.22]
    betas_nolrn = [0.45, 0.165, 0.3]
    # How long does the infection last
    gamma = 1/7
    # Only for SEIRC
    # Proportion of infected that become critical
    c1 = 0.02
    # Bed use rate, (Bed time ocuppancy)**-1
    c2 = 1/28
    # Available beds in ICU
    c3 = 1334  # Estimated number of beds in Chile
    # Time until lockdown, duration of lockdown
    days = [30, 30]
    # Population definition
    N = 1000000  # 18050000    # Chilean Population ~2018
    # exposed at initial time step
    E0 = 1
    # initial values for the models
    init_SIR = [N-E0, E0, 0]
    init_SEIR = [N-E0, E0, 0, 0]
    init_SEIRC = [N-E0, E0, 0, 0, 0]
    # Params definitions
    params_SIR = [betas_nolrn, gamma, days, N]
    params_SEIR_nl = [alpha, betas_nolrn, gamma, days, N]
    params_SEIR_l = [alpha, betas_learn, gamma, days, N]
    params_SEIRC = [alpha, betas, gamma, c1, c2, days, N]

    # solve ODE for SEIR model where the population changes their bhv or not
    df_SEIR_nl = model_solver(SEIR_ode, init_SEIR, params_SEIR_nl, [2020, 3, 1])
    df_SEIR_l = model_solver(SEIR_ode, init_SEIR, params_SEIR_l, [2020, 3, 1])
    ax_nl, ax_l = [df.plot() for df in [df_SEIR_nl, df_SEIR_l]]
    _ = [ax.figure.tight_layout() for ax in [ax_nl, ax_l]]
    _ = [ax.set_title(tit)
         for ax, tit in zip([ax_nl, ax_l],
                            ["Population doesn't change after lockdown",
                             'Population changes after lockdown'])]
    _ = [ax.vlines(df_SEIR_nl.iloc[days[0]*10].name, 0, N, color='k',
                   linestyle='--', label='lockdown')
         for ax in [ax_nl, ax_l]]
    _ = [ax.vlines(df_SEIR_nl.iloc[sum(days)*10].name, 0, N, color='magenta',
                   linestyle=':', label='lock release')
         for ax in [ax_nl, ax_l]]

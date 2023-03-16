import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import parameters as par
import scipy.interpolate as sci
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.offline import plot
from tqdm import tqdm

plt.rcParams[
    'axes.grid'] = False
symbols = ['META', 'JPM', 'WMT', 'TGT', 'AAPL', 'AMGN']
past = datetime.timedelta(weeks=52, days=0, hours=0, minutes=0)
start = datetime.date.today() - 5 * past
end = datetime.date.today()
images = 'images/'
red = '#f92672'
green = '#a6e22e'
blue = '#7fffff'

df = pd.DataFrame()
for sym in symbols:
    df[sym] = yf.download(sym, start=start, end=end)['Close']
print(df.tail())

########################## exercice 5 - base 100 ##########################################


############################################################################################

rets = np.log(df / df.shift(1))
print(rets)

########################## exercice 6 - rendement composé en continu ######################


############################################################################################


########################## exercice 7 - histogramme ########################################


############################################################################################

mus = (1 + rets.mean()) ** 252 - 1

cov = rets.cov() * 252

n_assets = len(symbols)
n_portfolios = 2500

mean_variance_pairs = []

np.random.seed(75)

for i in range(n_portfolios):
    assets = np.random.choice(list(rets.columns), n_assets, replace=False)
    weights = np.random.rand(n_assets)
    weights = weights / sum(weights)
    portfolio_E_Variance = 0
    portfolio_E_Return = 0
    for i in range(len(assets)):
        portfolio_E_Return += weights[i] * mus.loc[assets[i]]
        for j in range(len(assets)):
            portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

    mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])

mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate = 0.01

fig = go.Figure(layout=go.Layout(title=go.layout.Title(text="Portfolios")))
fig.add_trace(go.Scatter(x=mean_variance_pairs[:, 1] ** 0.5, y=mean_variance_pairs[:, 0],
                         marker=dict(
                             color=(mean_variance_pairs[:, 0] - risk_free_rate) / (mean_variance_pairs[:, 1] ** 0.5),
                             showscale=True,
                             size=7,
                             line=dict(width=1),
                             colorscale="RdBu",
                             colorbar=dict(title="Sharpe<br>Ratio")
                         ),
                         mode='markers'))
fig.update_layout(xaxis=dict(title="Volatility"),
                  yaxis=dict(title='Returns'),
                  title='Portfolios')
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe ratio"))
plot(fig, filename=images + '4.html', config={'displayModeBar': False}, auto_open=False)

mean_variance_pairs = []
weights_list = []
tickers_list = []

for i in tqdm(range(2500)):
    next_i = False
    while True:
        assets = np.random.choice(list(rets.columns), n_assets, replace=False)
        weights = np.random.rand(n_assets)
        weights = weights / sum(weights)

        portfolio_E_Variance = 0
        portfolio_E_Return = 0
        for i in range(len(assets)):
            portfolio_E_Return += weights[i] * mus.loc[assets[i]]
            for j in range(len(assets)):
                portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

        for R, V in mean_variance_pairs:
            if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                next_i = True
                break
        if next_i:
            break

        mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
        weights_list.append(weights)
        tickers_list.append(assets)
        break

len(mean_variance_pairs)

# -- Plot the risk vs. return of randomly generated portfolios
# -- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate = 0.01  # -- Include risk free rate here

fig = go.Figure(layout=go.Layout(title=go.layout.Title(text="Optimal portfolios")))
fig.add_trace(go.Scatter(x=mean_variance_pairs[:, 1] ** 0.5, y=mean_variance_pairs[:, 0],
                         marker=dict(
                             color=(mean_variance_pairs[:, 0] - risk_free_rate) / (mean_variance_pairs[:, 1] ** 0.5),
                             showscale=True,
                             size=7,
                             line=dict(width=1),
                             colorscale="RdBu",
                             colorbar=dict(title="Sharpe<br>Ratio")
                         ),
                         mode='markers',
                         text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i
                               in range(len(tickers_list))]))
fig.update_layout(xaxis=dict(title='Volatility'),
                  yaxis=dict(title='Returns'),
                  title='Optimal portfolios')
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
plot(fig, filename=images + '5.html', config={'displayModeBar': False}, auto_open=False)

###################################
noa = len(symbols)

weights = np.random.random(noa)
weights /= np.sum(weights)

eweights = np.array(noa * [1. / noa, ])


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))


prets = []
pvols = []
for p in range(n_portfolios):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

### Droite de marche
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)


def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)


def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)


def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


opt = sco.fsolve(equations, [0.01, 6, 0.2])

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#################### exercice 8 - donut ###################################################


###################################################################################################

# R_mean = (1+rets.mean())**252 - 1
R_mean = rets.mean() * 252


def F(w):
    rf = 0.01
    w = np.array(w)
    Rp_opt = np.sum(w * R_mean)
    Vp_opt = np.sqrt(np.dot(w, np.dot(cov, w.T)))
    SR = (Rp_opt - rf) / Vp_opt
    return np.array([Rp_opt, Vp_opt, SR])


def SRmin_F(w):
    return -F(w)[2]


cons_SR = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
res = sco.minimize(SRmin_F, eweights, method='SLSQP', bounds=bnds, constraints=cons_SR)
rf = 0.01
slope = -res['fun']
Rm = np.sum(R_mean * res['x'])
Vm = (Rm - rf) / slope


def f(w):
    w = np.array(w)
    Rp_opt = np.sum(w * R_mean)
    Vp_opt = np.sqrt(np.dot(w, np.dot(cov, w.T)))
    return np.array([Rp_opt, Vp_opt])


def Vmin_f(w):
    return f(w)[1]


cons_vmin = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
result_vmin = sco.minimize(Vmin_f, eweights, method='SLSQP', bounds=bnds, constraints=cons_vmin)
Rp_vmin = np.sum(R_mean * result_vmin['x'])
Vp_vmin = result_vmin['fun']

fig = plt.figure()
plt.scatter(pvols, prets, c=(prets - rf) / pvols, marker='.', cmap='cool')
plt.colorbar(label="Ratio de Sharpe")
plt.plot(tvols, trets, 'deeppink', lw=3)
plt.xlim(0.18, 0.33)
plt.ylim(0, 0.30)
plt.plot(Vp_vmin, Rp_vmin, 'lime', marker='*', markersize=14)
cx = np.linspace(0.02, 0.5)
Vp_cml = (cx - rf) / slope
plt.plot(Vp_cml, cx, 'greenyellow', linestyle='dashed')
plt.plot(Vm, Rm, 'orange', marker='*', markersize=14)
plt.axhline(0, c='w', ls='--')
plt.axvline(0, c='w', ls='--')
plt.savefig(images + '7')

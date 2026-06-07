# exercise sheet June 2026

#%%

from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np
from  datetime import datetime
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# stats and optimisation
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# user-defined functions
import utility_functions as uf



#%% return data analysis
ret = pd.read_excel('data/Return_Data.xlsx')
ret.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
ret.set_index('date', inplace=True)

# data quality checks
uf.check_data_quality(ret)

# summary stats
summary = uf.compute_summary_stats(ret)
summary
summary.to_excel('output/summary_stats_assets.xlsx')

# correlation matrix and heatmap
corr_matrix = ret.corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)    
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
# -> asset 3 and 5 are good diversifiers

# histograms
ret.hist(bins=25, figsize=(12, 8))
plt.suptitle('Histograms of asset returns')

# line plot of cumulative returns
cumulative_returns = (1 + ret).cumprod()
start_date = ret.index[0] - pd.DateOffset(months=1)
cumulative_returns.loc[start_date] = 1
cumulative_returns = cumulative_returns.sort_index()
ax = cumulative_returns.plot(figsize=(12, 8), title='Cumulative returns of assets')
ax.axhline(1, color='black', linestyle='--', alpha=0.5) 
plt.show()


#%% bootstrapping returns
# objective is to increase the robustness of the optimisation but preserve asset return characteristics such as volatility clustering as well as the correlation structure between the assets as correlations tend to increase during stock market drawdowns

nboot = 100000
sample_path_length = 12
num_years = sample_path_length / 12
# probability of taking the next observation in the sample path instead of a random one -> used to preserve volatility clustering in the bootstrapped returns
sim_serial_prob = 2/3

# bootstrapping
# sample all the random numbers at once to improve performance (sampling them during the loop is very slow)
rng = np.random.default_rng(seed=123)
# indices to draw from the historical returns
sample_indices = rng.integers(0, len(ret), [nboot, sample_path_length])
# random values to test against the serial probability threshold
rng = np.random.default_rng(seed=123)
serial_draw = rng.uniform(0, 1, [nboot, sample_path_length])

# run the bootstrapping
for iboot in range(0, nboot):
    # move along the sampling path: if the serial_draw value is below the threshold then the index from sample_indices is replaced by the consecutive one from the previous entry
    for isample in range(1, sample_path_length):
        # only if serial_threshold is below the input serial probability do we take the next sequential observation, i.e. with probability of input_serial_prob are we using the next observation
        if (serial_draw[iboot, isample] < sim_serial_prob):
            # replace the random index in sample_indices with the consecutive one
            # circular bootstrapping, i.e. if the consecutive index is larger than the length of the sample data then start at the beginning of the sample data -> avoids undersampling the beginning of the sample data
            sample_indices[iboot, isample] = np.mod(sample_indices[iboot, isample - 1] + 1, len(ret))

# we have the indices in a matrix format [nboot x sample_path_length] which is not suitable to index into the historical return matrix so we need to convert sample_indices into a vector
sample_indices_flat = sample_indices.flatten()
# using the vector we can index into the historical return data to create a matrix of bootstrapped returns [nboot*sample_path_length x nasset] where every <sample_path_length> rows make up one sample path
boot_ret_2d = ret.iloc[sample_indices_flat, :]

# compare the mean of the historical distribution to the mean of the bootstrapped distribution
# geometric average returns across all simulations
boot_geo_return = np.exp(np.log(1 + boot_ret_2d).mean() * 12) - 1
historical_geo_return = np.exp(np.log(1 + ret).mean() * 12) - 1
# differences between bootstrapped and historical geometric returns -> very small
boot_geo_return - historical_geo_return

# histograms of bootstrapped returns
boot_ret_2d.hist(bins=25, figsize=(12, 8))
plt.suptitle('Histograms of bootstrapped asset returns')



#%% optimisation (min variance)
# we want to find the optimal portfolio weights which achieve the required return but with the least amount of risk

# required monthly return -> after 10 years the target portfolio value is 10m and we are starting with a value of 7m
required_return = (10 / 7) ** (1/(10*12)) - 1
num_assets = ret.shape[1]
# constraints
# weights must sum to 1
cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
# geometric monthly return target -> inequality constraint gives more flexibility to the optimiser and from an investment perspective it also makes sense to allow for higher returns if they come with lower volatility
cons.append({'type': 'ineq', 'fun': lambda x: np.exp(np.log(1 + boot_ret_2d @ x).mean()) - 1 - required_return})
# no short selling, i.e. bounds of 0 and 1 for each weight
bounds = [(0, 1) for _ in range(num_assets)]
# initial guess for the weights -> equally weighted portfolio
x0 = np.array([1/num_assets] * num_assets)
# objective function minimise portfolio variance -> often works better than directly minimising volatility as the square root function can cause issues for the optimiser
obj_func = lambda x: x.T @ boot_ret_2d.cov() @ x

# use scipy minimise function to find the optimal weights
opt_result = minimize(obj_func, 
                      x0, 
                      method='SLSQP', 
                      constraints=cons, 
                      bounds=bounds
                      )
opt_result.success, opt_result.x, opt_result.fun
# put optimal weights into a data frame
opt_weights = pd.DataFrame(opt_result.x, index=ret.columns, columns=['weight'])
opt_weights['weight'] = round(opt_weights['weight'] * 100, 1)
opt_weights
# export optimal weights to excel
opt_weights.to_excel(f'output/optimal_weights_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_min_var.xlsx')
# portfolio volatility with optimal weights based on historical data
np.sqrt(opt_result.x.T @ ret.cov() @ opt_result.x) * np.sqrt(12) * 100

# pie chart of optimal weights
opt_weights.plot(kind='pie', y='weight', autopct='%1.1f%%', title="Optimal Portfolio Allocation", legend=False)

# portfolio returns based on historical data
opt_portfolio_return = ret @ opt_result.x
# line chart of cumulative returns of the optimal portfolio based on a 7m starting value
cumulative_opt_portfolio_return = (1 + opt_portfolio_return).cumprod() * 7
start_date = opt_portfolio_return.index[0] - pd.DateOffset(months=1)
cumulative_opt_portfolio_return.loc[start_date] = 7
cumulative_opt_portfolio_return = cumulative_opt_portfolio_return.sort_index() 
ax = cumulative_opt_portfolio_return.plot(figsize=(12, 8), title='Cumulative returns of optimal portfolio', linewidth=2)
ax.axhline(10, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Portfolio value (M EUR)')
ax.set_xlabel('Date')
sns.despine(top=True, right=True)
plt.show()

# optimisation based on historical returns (i.e. no bootstrapping) produced almost identical optimal weights


#%% optimisation - min expected shortfall

# required monthly return -> after 10 years the target portfolio value is 10m and we are starting with a value of 7m
required_return = (10 / 7) ** (1/(10*12)) - 1
num_assets = ret.shape[1]
# constraints
# weights must sum to 1
cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
# geometric monthly return target -> inequality constraint gives more flexibility to the optimiser and from an investment perspective it also makes sense to allow for higher returns if they come with lower volatility
cons.append({'type': 'ineq', 'fun': lambda x: np.exp(np.log(1 + boot_ret_2d @ x).mean()) - 1 - required_return})
# no short selling, i.e. bounds of 0 and 1 for each weight
bounds = [(0, 1) for _ in range(num_assets)]
# initial guess for the weights 
# -> equally weighted portfolio
#x0 = np.array([1/num_assets] * num_assets)
# or solution from min variance which speeds up the optimiser
x0 = np.round(opt_result.x, 4) # x0.sum()
# objective function minimise expected shortfall at 95% confidence level
def obj_func_es(weights, returns, alpha=0.05):
    portfolio_returns = returns @ weights
    var = np.quantile(portfolio_returns, alpha)
    es = portfolio_returns[portfolio_returns < var].mean()
    # we need a negative sign here because otherwise we would find the weights with the lowest negative return
    return -es
# use scipy minimise function to find the optimal weights
# the trust-constr method is more robust than SLSQP for this optimisation problem but it also takes much longer to run
opt_result = minimize(obj_func_es, 
                      x0, 
                      args=(boot_ret_2d, 0.05),
                      method='SLSQP', #'SLSQP', 'trust-constr'
                      constraints=cons, 
                      bounds=bounds
                      )
opt_result.success, opt_result.x, opt_result.fun
# put optimal weights into a data frame
opt_weights = pd.DataFrame(opt_result.x, index=ret.columns, columns=['weight'])
opt_weights['weight'] = round(opt_weights['weight'] * 100, 1)
opt_weights
# export optimal weights to excel
opt_weights.to_excel(f'output/optimal_weights_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_min_es_trust-constr.xlsx')
# portfolio volatility with optimal weights based on historical data
np.sqrt(opt_result.x.T @ ret.cov() @ opt_result.x) * np.sqrt(12) * 100
# portfolio expected shortfall with optimal weights based on historical data
opt_portfolio_return = ret @ opt_result.x
var_95 = np.quantile(opt_portfolio_return, 0.05)    
es_95 = opt_portfolio_return[opt_portfolio_return < var_95].mean() * 100
print(f"Expected shortfall at 95% confidence level: {es_95:.1f}%")
# pie chart of optimal weights
opt_weights.plot(kind='pie', y='weight', autopct='%1.1f%%', title="Optimal Portfolio Allocation", legend=False)

# portfolio returns based on historical data
opt_portfolio_return = ret @ opt_result.x
# line chart of cumulative returns of the optimal portfolio based on a 7m starting value
cumulative_opt_portfolio_return = (1 + opt_portfolio_return).cumprod() * 7
start_date = opt_portfolio_return.index[0] - pd.DateOffset(months=1)
cumulative_opt_portfolio_return.loc[start_date] = 7
cumulative_opt_portfolio_return = cumulative_opt_portfolio_return.sort_index() 
ax = cumulative_opt_portfolio_return.plot(figsize=(12, 8), title='Cumulative returns of optimal portfolio', linewidth=2)
ax.axhline(10, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Portfolio value (M EUR)')
ax.set_xlabel('Date')
sns.despine(top=True, right=True)
plt.show()



#%% optimisation (min var, covar shrinkage)
# we want to find the optimal portfolio weights which achieve the required return but with the least amount of risk

# required monthly return -> after 10 years the target portfolio value is 10m and we are starting with a value of 7m
required_return = (10 / 7) ** (1/(10*12)) - 1
num_assets = ret.shape[1]
# constraints
# weights must sum to 1
cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
# geometric monthly return target -> inequality constraint gives more flexibility to the optimiser and from an investment perspective it also makes sense to allow for higher returns IF they come with lower volatility
cons.append({'type': 'ineq', 'fun': lambda x: np.exp(np.log(1 + boot_ret_2d @ x).mean()) - 1 - required_return})
# no short selling, i.e. bounds of 0 and 1 for each weight
bounds = [(0, 1) for _ in range(num_assets)]
# initial guess for the weights -> equally weighted portfolio
x0 = np.array([1/num_assets] * num_assets)
# objective function minimise portfolio variance -> often works better than directly minimising volatility as the square root function can cause issues for the optimiser
lw = LedoitWolf()
lw.fit(boot_ret_2d)
obj_func = lambda x: x.T @ lw.covariance_ @ x

# use scipy minimise function to find the optimal weights
opt_result = minimize(obj_func, 
                      x0, 
                      method='SLSQP', 
                      constraints=cons, 
                      bounds=bounds
                      )
opt_result.success, opt_result.x, opt_result.fun
# put optimal weights into a data frame
opt_weights = pd.DataFrame(opt_result.x, index=ret.columns, columns=['weight'])
opt_weights['weight'] = round(opt_weights['weight'] * 100, 1)
opt_weights
# export optimal weights to excel
opt_weights.to_excel(f'output/optimal_weights_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_min_var_shrinkage.xlsx')
# portfolio volatility with optimal weights based on historical data
np.sqrt(opt_result.x.T @ ret.cov() @ opt_result.x) * np.sqrt(12) * 100

# pie chart of optimal weights
opt_weights.plot(kind='pie', y='weight', autopct='%1.1f%%', title="Optimal Portfolio Allocation", legend=False)

# portfolio returns based on historical data
opt_portfolio_return = ret @ opt_result.x
# line chart of cumulative returns of the optimal portfolio based on a 7m starting value
cumulative_opt_portfolio_return = (1 + opt_portfolio_return).cumprod() * 7
start_date = opt_portfolio_return.index[0] - pd.DateOffset(months=1)
cumulative_opt_portfolio_return.loc[start_date] = 7
cumulative_opt_portfolio_return = cumulative_opt_portfolio_return.sort_index() 
ax = cumulative_opt_portfolio_return.plot(figsize=(12, 8), title='Cumulative returns of optimal portfolio', linewidth=2)
ax.axhline(10, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Portfolio value (M EUR)')
ax.set_xlabel('Date')
sns.despine(top=True, right=True)
plt.show()


#%% summary stats of optimised portfolios

# load all the allocations saved in the output folder
output_files = glob.glob('output/optimal_weights_*.xlsx')
all_df_with_names = []
for f in output_files:
    temp_df = pd.read_excel(f)
    base = os.path.basename(f)
    name_only = os.path.splitext(base)[0]
    if name_only.find('min') != -1:
        name_only = name_only[name_only.find('min'):]
    temp_df['opt_name'] = name_only 
    all_df_with_names.append(temp_df)
weight_df = pd.concat(all_df_with_names, ignore_index=True)
weight_df.rename(columns={'Unnamed: 0': 'asset'}, inplace=True)

weight_pivot = weight_df.pivot(index='asset', columns='opt_name', values='weight')

# compute portfolio returns
portfolio_returns = ret @ (weight_pivot.values/100)
portfolio_returns.columns = weight_pivot.columns

# summary stats
summary_portfolios = uf.compute_summary_stats(portfolio_returns)
summary_portfolios

# cumulative returns of the optimised portfolios
cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() * 7
start_date = portfolio_returns.index[0] - pd.DateOffset(months=1)
cumulative_portfolio_returns.loc[start_date] = 7
cumulative_portfolio_returns = cumulative_portfolio_returns.sort_index()

total_periods = len(cumulative_portfolio_returns)
years_index = np.arange(total_periods) / 12
plot_df = cumulative_portfolio_returns.copy()
plot_df.index = years_index

fig, ax = plt.subplots(figsize=(12, 8))
plot_df.plot(ax=ax, linewidth=2)
ax.set_title('Cumulative returns of optimised portfolios')
ax.axhline(10, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Portfolio value (M EUR)')
ax.set_xlabel('Year')
sns.despine(top=True, right=True)
plt.show()

# plot of drawdowns of the optimised portfolios (area chart)
drawdowns = (cumulative_portfolio_returns / cumulative_portfolio_returns.cummax() - 1) * 100
fig, ax = plt.subplots(figsize=(12, 8))
for column in drawdowns.columns:
    ax.plot(drawdowns.index, drawdowns[column], label=column, linewidth=2)
    ax.fill_between(drawdowns.index, drawdowns[column], 0, alpha=0.2)
ax.set_title('Drawdowns of optimised portfolios')
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
sns.despine(top=True, right=True)
plt.show()


#%% plots for presentation

# horizontal bar chart
weight_pivot_tmp = weight_pivot.loc[:, ['min_es', 'min_var']].copy()
weight_pivot_tmp.columns = ['Min ES', 'Min variance']
df_stacked = weight_pivot_tmp.T
custom_colours = [
    '#1A659E',  # dark blue
    '#90E0EF',  # light blue
    '#5C677D',  # dark grey
    '#D3D3D3',  # light grey
    '#F4A261',  # orange
    '#52B788'   # green
]

fig, ax = plt.subplots(figsize=(12, 6))
df_stacked.plot(kind='barh', stacked=True, ax=ax, width=0.8, color=custom_colours)
ax.set_title('Optimal allocations', fontsize=16)
ax.set_xlabel('Weight in %', fontsize=16)
ax.set_ylabel('Allocation', fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(title='Assets', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig('output/optimal_allocations.jpeg', dpi=400)


# compute portfolio returns
portfolio_returns = ret @ (weight_pivot_tmp.values/100)
portfolio_returns.columns = weight_pivot_tmp.columns

# cumulative return chart of the two allocations
cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() * 7
start_date = portfolio_returns.index[0] - pd.DateOffset(months=1)
cumulative_portfolio_returns.loc[start_date] = 7
cumulative_portfolio_returns = cumulative_portfolio_returns.sort_index()

total_periods = len(cumulative_portfolio_returns)
years_index = np.arange(total_periods) / 12
plot_df = cumulative_portfolio_returns.copy()
plot_df.index = years_index

fig, ax = plt.subplots(figsize=(12, 8))
plot_df.plot(ax=ax, linewidth=2)
ax.set_title('Cumulative returns of optimised portfolios', fontsize=16)
ax.axhline(10, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Portfolio value (M EUR)', fontsize=14)
ax.set_xlabel('Year', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
sns.despine(top=True, right=True)
ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=12, ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('output/optimal_allocations_cumulative_returns.jpeg', dpi=400)



#%%
# ideas
# - covariance matrix shrinkage -> same allocation



#%% testing

uf.calculate_value_at_risk(ret['Asset 1'])
uf.calculate_value_at_risk(ret['Asset 1'].to_list(), confidence_level=0.95)
uf.calculate_value_at_risk(ret['Asset 1'].values, confidence_level=0.95)
uf.calculate_value_at_risk(ret['Asset 1'], confidence_level=0.05)
uf.calculate_value_at_risk(ret['Asset 1'], confidence_level=95)
uf.calculate_value_at_risk(ret['Asset 1'], confidence_level=0.95, method='empirical')
uf.calculate_value_at_risk(ret['Asset 1'], confidence_level=0.95, method='parametric')
uf.calculate_value_at_risk(ret['Asset 1'], confidence_level=0.95, method='abc')
uf.calculate_value_at_risk(ret.loc[:, ['Asset 1', 'Asset 2']])

uf.calculate_expected_shortfall(ret['Asset 1'])
uf.calculate_expected_shortfall(ret['Asset 1'].to_list(), confidence_level=0.95)
uf.calculate_expected_shortfall(ret['Asset 1'].values, confidence_level=0.95)
uf.calculate_expected_shortfall(ret['Asset 1'], confidence_level=0.05)
uf.calculate_expected_shortfall(ret['Asset 1'], confidence_level=95)
uf.calculate_expected_shortfall(ret['Asset 1'], confidence_level=0.95, method='empirical')
uf.calculate_expected_shortfall(ret['Asset 1'], confidence_level=0.95, method='parametric')
uf.calculate_expected_shortfall(ret.loc[:, ['Asset 1', 'Asset 2']])

uf.calculate_total_return(ret['Asset 1'])
uf.calculate_total_return(ret['Asset 1'], method='cumulative')
uf.calculate_total_return(ret['Asset 1'], method='geometric_average')
uf.calculate_total_return(ret.loc[:, ['Asset 1', 'Asset 2']])



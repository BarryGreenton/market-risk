import numpy as np
from scipy.optimize import minimize

def calculate_var(returns, confidence_level=0.99):
    if len(returns) > 0:
        return -np.percentile(returns, 100 * (1 - confidence_level))
    return 0


def objective_function(equity_weight, equity_returns, rf=0.03):
    portfolio_returns = equity_returns * equity_weight
    portfolio_std = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (np.mean(portfolio_returns) * 252 - rf) / portfolio_std
    return -sharpe_ratio


def optimize_portfolio(equity_returns, var_limit):
    bounds = [(0, 1)]
    constraints = []

    def var_constraint(equity_weight):
        portfolio_returns = equity_returns * equity_weight
        portfolio_var = calculate_var(portfolio_returns, confidence_level=0.99)
        return var_limit - portfolio_var
    constraints.append({'type': 'ineq', 'fun': var_constraint})
    result = minimize(objective_function, x0=np.array([0.5]), args=(equity_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_equity_weight = result.x[0] if result.success else 0.5
    return (optimal_equity_weight, 1 - optimal_equity_weight)

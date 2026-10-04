import cvxpy as cp
w, c = cp.Variable(data.n_assets), cp.Variable()
z = w - data.w_prev
T = cp.norm1(z) / 2
L = cp.norm1(w)
#worst-case (robust) return
    

factor_return = (data.F @ data.factor_mean).T @ w
idio_return = data.idio_mean @ w
mean_return = factor_return + idio_return + data.risk_free * c
return_uncertainty = param.rho_mean @ cp.abs(w)
return_wc = mean_return - return_uncertainty
# worst-case (robust) risk
factor_risk = cp.norm2((data.F @ data.factor_covariance_chol).T @ w)
idio_risk = cp.norm2(cp.multiply(data.idio_volas, w))
risk = cp.norm2(cp.hstack([factor_risk, idio_risk]))
risk_uncertainty = param.rho_covariance**0.5 * data.volas @ cp.abs(w)
risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))
asset_holding_cost = data.kappa_short @ cp.pos(-w)
cash_holding_cost = data.kappa_borrow * cp.pos(-c)
holding_cost = asset_holding_cost + cash_holding_cost
spread_cost = data.kappa_spread @ cp.abs(z)
impact_cost = data.kappa_impact @ cp.power(cp.abs(z), 3 / 2)
trading_cost = spread_cost + impact_cost
objective = (
return_wc
- param.gamma_hold * holding_cost
- param.gamma_trade * trading_cost
)
constraints = [
cp.sum(w) + c == 1,
param.w_min <= w, w <= param.w_max,
L <= param.L_tar,
param.c_min <= c, c <= param.c_max,
param.z_min <= z, z <= param.z_max,
T <= param.T_tar,
risk_wc <= param.risk_target,
]
"""Discrete multi-stock trading environment with share-based actions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .costs import evaluate_transaction_costs
from .frictions import FrictionConfig
from ..utils.portfolio import (
    apply_tau_limit,
    integerize_allocations,
    project_to_l1_ball,
    project_to_simplex,
    validate_holdings_payload,
)


@dataclass(slots=True)
class ActionProjectionConfig:
    """Configuration for projecting raw actions onto feasible allocation weights."""

    mode: str = "simplex"
    temperature: float = 1.0
    include_cash_asset: bool = False
    max_leverage: float = 1.0


@dataclass(slots=True)
class SafetyConfig:
    enabled: bool = True
    med_on: float = 0.95
    med_off: float = 0.92
    hi_on: float = 0.99
    hi_off: float = 0.96
    gmax_normal: float = 1.0
    gmax_caution_high: float = 0.7
    gmax_caution_low: float = 0.3
    gmax_crisis: float = 0.1
    per_asset_cap: float = 0.2
    delta_w_caution: float = 0.15
    delta_w_crisis: float = 0.05
    delta_q_caution: float = 0.10
    delta_q_crisis: float = 0.05
    kappa: float = 0.2
    cooldown_days: int = 3
    buy_freeze_crisis: bool = True


@dataclass(slots=True)
class EnvironmentConfig:
    """Runtime options for the discrete trading environment."""

    initial_cash: float = 10_000_000_000.0
    n: int = 1  # maximum absolute trade size per step for each asset
    max_steps: Optional[int] = None
    reward_scaling: float = 1.0
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    risk_free_rate: float = 0.0
    action_mode: str = "weights"
    tau: float = 0.1
    lambda_turnover: float = 0.02
    time_feature_wrapper: bool = True
    max_leverage: float = 1.0
    projection: ActionProjectionConfig = field(default_factory=ActionProjectionConfig)
    stress: Optional[dict[str, Any]] = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)


class MultiStockTradingEnv(gym.Env[NDArray[np.float32], np.ndarray]):
    """Trading environment with discrete buy/hold/sell actions per stock."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, data: pd.DataFrame, config: EnvironmentConfig | None = None) -> None:
        super().__init__()
        if data is None or data.empty:
            raise ValueError("Market data frame cannot be empty")

        required_columns = {"date", "symbol", "close"}
        missing = required_columns - set(data.columns)
        if missing:
            raise KeyError(f"Data frame missing required columns: {', '.join(sorted(missing))}")

        self.raw_data = data.copy()
        self.raw_data["date"] = pd.to_datetime(self.raw_data["date"])
        self.raw_data.sort_values(["date", "symbol"], inplace=True)
        self.config = config or EnvironmentConfig()
        self.initial_cash = float(self.config.initial_cash)
        self._base_initial_cash = float(self.initial_cash)

        # Safety setup (must be initialized before panel for turbulence cache)
        self._safety_cfg: SafetyConfig = (
            self.config.safety if isinstance(self.config.safety, SafetyConfig) else SafetyConfig(**(self.config.safety or {}))  # type: ignore[arg-type]
        )
        self._regime_state: str = "normal"
        self._cooldown_remaining: int = 0
        self._current_gmax: float = self._safety_cfg.gmax_normal
        self._turb_map: dict[pd.Timestamp, tuple[float, float]] = {}

        self._prepare_panel()

        self._action_n = max(1, int(self.config.n))
        self._friction = self.config.friction
        stress_cfg = getattr(self.config, "stress", None) or {}
        if stress_cfg.get("enabled"):
            commission_mult = float(stress_cfg.get("commission_multiplier", 1.0))
            slippage_mult = float(stress_cfg.get("slippage_multiplier", 1.0))
            self._friction = replace(
                self._friction,
                commission_rate=self._friction.commission_rate * commission_mult,
                slippage_bps=self._friction.slippage_bps * slippage_mult,
            )
        self._commission_rate = float(self._friction.commission_rate)
        self.reward_scaling = float(self.config.reward_scaling)
        self.stress_profile = stress_cfg
        # Safety setup already initialized above

        asset_dim = len(self.symbols)
        if asset_dim == 0:
            raise ValueError("Environment requires at least one tradable symbol")

        self._action_mode = str(self.config.action_mode).lower()
        if self._action_mode not in {"weights", "shares"}:
            raise ValueError("action_mode must be either 'weights' or 'shares'")
        self._tau = max(0.0, float(self.config.tau))
        self._lambda_turnover = max(0.0, float(self.config.lambda_turnover))

        if self._action_mode == "weights":
            projection_cfg = self.config.projection or ActionProjectionConfig()
            if isinstance(projection_cfg, dict):
                projection_cfg = ActionProjectionConfig(**projection_cfg)
            self._projection_cfg = projection_cfg
            self._projection_dim = asset_dim + (1 if projection_cfg.include_cash_asset else 0)
            low = np.full(self._projection_dim, -1.0, dtype=np.float32)
            high = np.full(self._projection_dim, 1.0, dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete(
                np.full(asset_dim, 2 * self._action_n + 1, dtype=np.int64)
            )

        obs_dim = 1 + 2 * asset_dim  # cash + prices + holdings
        high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        available_steps = max(1, self.num_prices - 1)
        if self.config.max_steps is None:
            self.max_trade_steps = available_steps
        else:
            requested_steps = int(self.config.max_steps)
            if requested_steps <= 0:
                raise ValueError("max_steps must be positive when provided")
            self.max_trade_steps = min(requested_steps, available_steps)
        if self.max_trade_steps <= 0:
            raise ValueError("Not enough timesteps in data to perform trading")

        self._rng = np.random.default_rng()
        self._pending_portfolio_state: Optional[Dict[str, Any]] = None
        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.initial_cash = float(self._base_initial_cash)
        self.cash: float = self.initial_cash
        self.holdings: NDArray[np.int64] = np.zeros(len(self.symbols), dtype=np.int64)
        self.steps_elapsed: int = 0
        self.current_idx: int = 0
        self.terminal: bool = False
        self.asset_history: list[float] = [self.initial_cash]
        self.date_history: list[pd.Timestamp] = [self.dates[self.current_idx]]

        self.peak_value = self.initial_cash
        self.last_turnover = 0.0
        self.last_nav_return = 0.0
        self.last_log_return = 0.0
        self.last_drawdown = 0.0
        self.last_commission_cost = 0.0
        self.last_slippage_cost = 0.0
        self.last_cost_ratio = 0.0
        self.last_returns = np.zeros(len(self.symbols), dtype=np.float64)
        current_prices = self.price_matrix[self.current_idx]
        invested_value = float(np.dot(self.holdings, current_prices))
        portfolio_value = self.cash + invested_value
        if portfolio_value > 0.0:
            weights = (self.holdings * current_prices) / portfolio_value
            cash_fraction = self.cash / portfolio_value
        else:
            weights = np.zeros(len(self.symbols), dtype=np.float64)
            cash_fraction = 0.0
        self.last_weights = weights
        self.last_cash_fraction = cash_fraction
        self.prev_weights = weights.copy()
        self.prev_cash_fraction = cash_fraction
        self.last_gross_value = portfolio_value

        portfolio_state: Optional[Dict[str, Any]] = None
        if self._pending_portfolio_state is not None:
            portfolio_state = dict(self._pending_portfolio_state)
        if options and isinstance(options, dict) and options.get("portfolio_state") is not None:
            opt_state = options.get("portfolio_state")
            if portfolio_state is None:
                portfolio_state = dict(opt_state)
            else:
                portfolio_state.update(opt_state)
        self._pending_portfolio_state = None
        if portfolio_state:
            self._apply_portfolio_state(portfolio_state)

        observation = self._observation()
        info = self._info(reward_unscaled=0.0)
        return observation, info

    def set_portfolio_state(self, portfolio_state: Optional[Mapping[str, Any]]) -> None:
        """Queue a portfolio override to be applied on the next reset."""

        if portfolio_state is None:
            self._pending_portfolio_state = None
        else:
            self._pending_portfolio_state = dict(portfolio_state)

    def step(
        self, action: np.ndarray | NDArray[np.float64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if self.terminal or self.steps_elapsed >= self.max_trade_steps:
            self.terminal = True
            info = self._info(reward_unscaled=self.asset_history[-1] - self.initial_cash)
            info["episode_final"] = True
            info["final_reward"] = info["reward_unscaled"]
            info["final_portfolio_value"] = info["portfolio_value"]
            return self._observation(), 0.0, True, False, info
        current_prices = self.price_matrix[self.current_idx]
        nav_before = self.cash + float(np.dot(self.holdings, current_prices))

        if self._action_mode == "weights":
            action_arr = np.asarray(action, dtype=np.float64)
            if action_arr.ndim == 0:
                action_arr = np.array([float(action_arr)], dtype=np.float64)
            if action_arr.shape[0] != self._projection_dim:
                raise ValueError(
                    f"Expected action dimension {self._projection_dim}, received {action_arr.shape}"
                )

            asset_logits = action_arr[: len(self.symbols)]
            projection_mode = str(self._projection_cfg.mode).lower()
            if projection_mode == "simplex":
                logits_full = (
                    action_arr
                    if self._projection_cfg.include_cash_asset
                    else asset_logits
                )
                projected = project_to_simplex(
                    logits_full,
                    temperature=max(1e-6, float(self._projection_cfg.temperature)),
                )
                if self._projection_cfg.include_cash_asset:
                    target_asset_weights = projected[:-1]
                    target_cash_weight = float(projected[-1])
                else:
                    target_asset_weights = projected
                    target_cash_weight = max(0.0, 1.0 - float(projected.sum()))
            elif projection_mode == "l1":
                non_negative = np.clip(asset_logits, 0.0, None)
                projected = project_to_l1_ball(
                    non_negative,
                    radius=max(1e-6, float(self._projection_cfg.max_leverage)),
                )
                total_mass = float(projected.sum())
                if total_mass <= 1e-8:
                    denom = len(self.symbols) + 1
                    target_asset_weights = np.full(len(self.symbols), 1.0 / denom, dtype=np.float64)
                    target_cash_weight = 1.0 / denom
                else:
                    target_asset_weights = projected / total_mass
                    target_cash_weight = max(0.0, 1.0 - total_mass)
            else:
                raise ValueError(
                    f"Unsupported projection mode '{self._projection_cfg.mode}'."
                )

            target_asset_weights = np.clip(target_asset_weights, 0.0, None)
            target_cash_weight = max(0.0, float(target_cash_weight))
            target_full = np.concatenate(
                (target_asset_weights, np.array([target_cash_weight], dtype=np.float64))
            )
            total_target = float(target_full.sum())
            if not np.isfinite(total_target) or total_target <= 0:
                target_full = np.full(
                    len(target_full),
                    1.0 / len(target_full),
                    dtype=np.float64,
                )
            else:
                target_full = target_full / total_target

            prev_full = np.concatenate(
                (self.prev_weights.astype(np.float64, copy=False),
                 np.array([self.prev_cash_fraction], dtype=np.float64))
            )
            limited_full = apply_tau_limit(prev_full, target_full, self._tau)
            target_weights = limited_full[:-1]
            target_cash_weight = float(limited_full[-1])

            # Safety layer (regime-based exposure + throttles)
            buy_freeze_hits = 0
            weight_throttle_applied = False
            quantity_scale = 1.0
            q_port_val = float("nan")
            q_sys_val = float("nan")
            if self._safety_cfg.enabled and self._turb_map:
                q_port_val, q_sys_val = self._turb_map.get(self.dates[self.current_idx], (float("nan"), float("nan")))
                q_any = np.nanmax([q_port_val, q_sys_val]) if not (np.isnan(q_port_val) and np.isnan(q_sys_val)) else 0.0
                sc = self._safety_cfg
                # Hysteresis regime update
                if self._regime_state == "normal":
                    if q_any >= sc.hi_on:
                        self._regime_state = "crisis"; self._cooldown_remaining = sc.cooldown_days
                    elif q_any >= sc.med_on:
                        self._regime_state = "caution"
                elif self._regime_state == "caution":
                    if q_any >= sc.hi_on:
                        self._regime_state = "crisis"; self._cooldown_remaining = sc.cooldown_days
                    elif q_any < sc.med_off:
                        self._regime_state = "normal"
                elif self._regime_state == "crisis":
                    if q_any < sc.hi_off:
                        self._regime_state = "caution"; self._cooldown_remaining = sc.cooldown_days
                if self._regime_state == "caution" and self._cooldown_remaining > 0:
                    self._cooldown_remaining -= 1
                # Target gmax by regime
                if self._regime_state == "crisis":
                    g_target = sc.gmax_crisis
                elif self._regime_state == "caution":
                    if q_any <= sc.med_on:
                        g_target = sc.gmax_caution_high
                    elif q_any >= sc.hi_on:
                        g_target = sc.gmax_caution_low
                    else:
                        ratio = (q_any - sc.med_on) / max(1e-6, (sc.hi_on - sc.med_on))
                        g_target = sc.gmax_caution_high - ratio * (sc.gmax_caution_high - sc.gmax_caution_low)
                else:
                    g_target = sc.gmax_normal
                # Ramp towards target
                self._current_gmax = float(self._current_gmax + sc.kappa * (g_target - self._current_gmax))
                self._current_gmax = float(np.clip(self._current_gmax, sc.gmax_crisis, sc.gmax_normal))
                # Per-asset cap then scale to gmax
                capped = np.minimum(target_weights, float(sc.per_asset_cap))
                s = float(np.sum(capped))
                if s > 1e-12:
                    scale = min(1.0, self._current_gmax / s)
                    capped = capped * scale
                target_weights = capped
                target_cash_weight = max(0.0, 1.0 - float(np.sum(target_weights)))
                # Additional L1 throttle on weights (stronger than base tau)
                tau_extra = sc.delta_w_crisis if self._regime_state == "crisis" else (sc.delta_w_caution if self._regime_state == "caution" else None)
                if tau_extra is not None and tau_extra < self._tau:
                    prev_full2 = prev_full
                    target_full2 = np.concatenate((target_weights, np.array([target_cash_weight], dtype=np.float64)))
                    limited2 = apply_tau_limit(prev_full2, target_full2, tau_extra)
                    if np.any(np.abs(limited2 - target_full2) > 1e-12):
                        weight_throttle_applied = True
                    target_weights = limited2[:-1]
                    target_cash_weight = float(limited2[-1])

            price_map = {symbol: float(price) for symbol, price in zip(self.symbols, current_prices)}
            holdings_map, _, _ = integerize_allocations(
                {sym: w for sym, w in zip(self.symbols, target_weights)},
                price_map,
                nav=max(nav_before, 1e-8),
                symbols=self.symbols,
                lot_size=1,
            )
            validate_holdings_payload(self.symbols, holdings_map)
            desired_holdings = np.array([int(holdings_map[sym]) for sym in self.symbols], dtype=np.int64)
            trade_units = desired_holdings - self.holdings
            # Crisis buy-freeze
            if self._safety_cfg.enabled and self._regime_state == "crisis" and self._safety_cfg.buy_freeze_crisis:
                mask_pos = trade_units > 0
                buy_freeze_hits = int(np.sum(mask_pos))
                trade_units = np.where(mask_pos, 0, trade_units)
            # Quantity throttle (L1)
            if self._safety_cfg.enabled and np.any(trade_units != 0):
                total_h = float(np.sum(np.abs(self.holdings)))
                dq = self._safety_cfg.delta_q_crisis if self._regime_state == "crisis" else (self._safety_cfg.delta_q_caution if self._regime_state == "caution" else None)
                if dq is not None:
                    budget = float(dq) * max(1.0, total_h)
                    l1 = float(np.sum(np.abs(trade_units)))
                    if l1 > budget and budget > 0:
                        quantity_scale = float(budget / l1)
                        trade_units = np.floor(trade_units.astype(np.float64) * quantity_scale).astype(np.int64)
        else:
            action_arr = np.asarray(action, dtype=np.int64)
            if action_arr.ndim == 0:
                action_arr = np.array([action_arr], dtype=np.int64)
            if action_arr.shape[0] != len(self.symbols):
                raise ValueError(
                    f"Expected action dimension {len(self.symbols)}, received {action_arr.shape}"
                )
            trade_units = action_arr - self._action_n

        sell_notionals: list[float] = []
        sell_prices: list[float] = []

        # Execute sells first (negative trade units)
        for idx, units in enumerate(trade_units):
            if units >= 0:
                continue
            price = float(current_prices[idx])
            if price <= 0:
                continue
            available = int(self.holdings[idx])
            qty = min(available, int(-units))
            if qty <= 0:
                continue
            notional = price * qty
            self.holdings[idx] -= qty
            self.cash += notional
            sell_notionals.append(notional)
            sell_prices.append(price)

        buy_notionals: list[float] = []
        buy_prices: list[float] = []

        # Execute buys (positive trade units) using remaining cash
        for idx, units in enumerate(trade_units):
            if units <= 0:
                continue
            price = float(current_prices[idx])
            if price <= 0:
                continue
            max_affordable = int(self.cash // price)
            qty = min(int(units), max_affordable)
            if qty <= 0:
                continue
            cost = price * qty
            self.cash -= cost
            self.holdings[idx] += qty
            buy_notionals.append(cost)
            buy_prices.append(price)

        commission_cost, slippage_cost, trade_notional_total = evaluate_transaction_costs(
            buy_notionals=buy_notionals,
            buy_prices=buy_prices,
            sell_notionals=sell_notionals,
            sell_prices=sell_prices,
            friction=self._friction,
        )
        total_cost = commission_cost + slippage_cost

        cash_pre_cost = float(self.cash)

        self.steps_elapsed += 1
        self.current_idx = min(self.steps_elapsed, self.num_prices - 1)
        next_prices = self.price_matrix[self.current_idx]
        prev_portfolio_value = self.asset_history[-1]
        holdings_value = float(np.dot(self.holdings, next_prices))
        gross_portfolio_value = cash_pre_cost + holdings_value
        self.cash = cash_pre_cost - total_cost
        portfolio_value = self.cash + holdings_value

        self.peak_value = max(self.peak_value, portfolio_value)
        gross_denom = max(gross_portfolio_value, 1e-8)
        prev_denom = max(prev_portfolio_value, 1e-8)
        self.last_turnover = trade_notional_total / gross_denom if gross_denom > 0 else 0.0
        if prev_portfolio_value > 0:
            nav_return = portfolio_value / prev_portfolio_value - 1.0
            log_return = np.log(max(portfolio_value, 1e-8) / max(prev_portfolio_value, 1e-8))
        else:
            nav_return = 0.0
            log_return = 0.0
        self.last_nav_return = nav_return
        self.last_log_return = log_return
        self.last_cost_ratio = total_cost / gross_denom if gross_denom > 0 else 0.0
        if self.peak_value > 0:
            self.last_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            self.last_drawdown = 0.0
        returns_vec = np.zeros_like(current_prices, dtype=np.float64)
        non_zero_mask = current_prices > 0
        returns_vec[non_zero_mask] = (next_prices[non_zero_mask] - current_prices[non_zero_mask]) / current_prices[non_zero_mask]
        self.last_returns = returns_vec
        self.last_commission_cost = commission_cost
        self.last_slippage_cost = slippage_cost
        if gross_portfolio_value > 0:
            weights_exec = (self.holdings * next_prices) / gross_portfolio_value
            cash_fraction = self.cash / gross_portfolio_value
        else:
            weights_exec = np.zeros(len(self.symbols), dtype=np.float64)
            cash_fraction = 0.0
        self.last_weights = weights_exec
        self.last_cash_fraction = cash_fraction
        self.prev_weights = weights_exec
        self.prev_cash_fraction = cash_fraction
        self.last_gross_value = gross_portfolio_value
        # Persist safety/throttle diagnostics for info()
        self.last_q_port = q_port_val if 'q_port_val' in locals() else float('nan')
        self.last_q_sys = q_sys_val if 'q_sys_val' in locals() else float('nan')
        self.last_weight_throttle = bool(locals().get('weight_throttle_applied', False))
        self.last_quantity_scale = float(locals().get('quantity_scale', 1.0))
        self.last_buy_freeze_hits = int(locals().get('buy_freeze_hits', 0))
        self.last_cap_hit_ratio = float(locals().get('cap_hit_ratio', 0.0))

        self.asset_history.append(portfolio_value)
        self.date_history.append(self.dates[self.current_idx])

        if self.steps_elapsed >= self.max_trade_steps:
            self.terminal = True

        reward_unscaled = log_return - self._lambda_turnover * self.last_turnover
        reward = float(reward_unscaled * self.reward_scaling)

        observation = self._observation()
        info = self._info(reward_unscaled=reward_unscaled)
        info["reward"] = reward
        if self.terminal:
            info["episode_final"] = True
            info["final_reward"] = reward_unscaled
            info["final_portfolio_value"] = portfolio_value
        return observation, reward, self.terminal, False, info

    def render(self) -> Dict[str, float]:
        prices = self.price_matrix[self.current_idx]
        portfolio_value = self.cash + float(np.dot(self.holdings, prices))
        return {
            "cash": float(self.cash),
            "portfolio_value": float(portfolio_value),
        }

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_panel(self) -> None:
        self.symbols = sorted(self.raw_data["symbol"].astype(str).unique().tolist())
        self.dates = np.sort(self.raw_data["date"].unique())
        if self.dates.size < 2:
            raise ValueError("Market data must span at least two dates")

        self.num_prices = len(self.dates)
        num_symbols = len(self.symbols)
        price_matrix = np.zeros((self.num_prices, num_symbols), dtype=np.float32)

        for col_idx, symbol in enumerate(self.symbols):
            sym_frame = (
                self.raw_data[self.raw_data["symbol"].astype(str) == symbol]
                .set_index("date")
                .sort_index()
            )
            series = sym_frame["close"].reindex(self.dates).ffill()
            if series.isna().any():
                series = series.fillna(method="bfill")
            if series.isna().any():
                raise ValueError(f"Unable to fill prices for symbol {symbol}")
            price_matrix[:, col_idx] = series.to_numpy(dtype=np.float32)

        self.price_matrix = price_matrix
        self._symbol_to_index = {symbol: idx for idx, symbol in enumerate(self.symbols)}

        mean = price_matrix.mean(axis=0, keepdims=True)
        std = price_matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        self.normalized_prices = (price_matrix - mean) / std

        # Build turbulence per-date cache if available in data
        if "turb_port" in self.raw_data.columns or "turb_sys" in self.raw_data.columns:
            grp_port = self.raw_data.groupby("date")["turb_port"].first() if "turb_port" in self.raw_data.columns else None
            grp_sys = self.raw_data.groupby("date")["turb_sys"].first() if "turb_sys" in self.raw_data.columns else None
            for d in self.dates:
                qp = float(grp_port.loc[d]) if grp_port is not None and d in grp_port.index else float("nan")
                qs = float(grp_sys.loc[d]) if grp_sys is not None and d in grp_sys.index else float("nan")
                self._turb_map[d] = (qp, qs)

    def _observation(self) -> NDArray[np.float32]:
        prices_norm = self.normalized_prices[self.current_idx]
        baseline_cash = self.initial_cash if abs(self.initial_cash) > 1e-8 else 1.0
        cash_norm = (self.cash / baseline_cash) - 1.0
        holdings_norm = self.holdings.astype(np.float32, copy=False) / max(1, self._action_n)
        obs = np.concatenate(
            (
                np.array([cash_norm], dtype=np.float32),
                prices_norm.astype(np.float32, copy=False),
                holdings_norm.astype(np.float32, copy=False),
            )
        )
        return obs.astype(np.float32, copy=False)

    def _resolve_start_index(self, state: Mapping[str, Any]) -> int:
        idx_candidate = state.get("start_index")
        if idx_candidate is None:
            idx_candidate = state.get("start_idx")
        if idx_candidate is not None:
            idx = int(idx_candidate)
            if not 0 <= idx < self.num_prices:
                raise ValueError(
                    f"Requested start index {idx} outside price history (size={self.num_prices})."
                )
            return idx

        date_value = state.get("date") or state.get("start_date")
        if date_value is not None:
            ts = pd.to_datetime(date_value)
            matches = np.where(self.dates == ts.to_datetime64())[0]
            if matches.size == 0:
                raise ValueError(f"Start date {ts.date()} not available in loaded data.")
            return int(matches[0])

        return 0

    def _apply_portfolio_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return

        start_idx = self._resolve_start_index(state)
        self.current_idx = start_idx
        self.steps_elapsed = 0
        self.terminal = False

        prices = self.price_matrix[self.current_idx]
        holdings_vec = np.zeros(len(self.symbols), dtype=np.float64)

        holdings_map = state.get("holdings") or state.get("positions")
        if not holdings_map:
            raise ValueError(
                "Portfolio state must include 'holdings' mapping with per-symbol share counts."
            )

        normalized_keys: Dict[str, float] = {}
        for symbol, qty in holdings_map.items():
            key = str(symbol)
            if key not in self._symbol_to_index:
                raise KeyError(f"Unknown symbol '{key}' in portfolio override.")
            normalized_keys[key] = float(qty)

        missing = [symbol for symbol in self.symbols if symbol not in normalized_keys]
        if missing:
            raise KeyError(
                "Portfolio override missing symbols: "
                + ", ".join(missing)
                + ". Provide 0 shares explicitly when no position is held."
            )

        for symbol, qty in normalized_keys.items():
            slot = self._symbol_to_index[symbol]
            holdings_vec[slot] = int(np.round(qty))

        if "cash" not in state:
            raise ValueError("Portfolio state must include a 'cash' balance field.")
        base_cash = float(state["cash"])

        invested_value = float(np.dot(holdings_vec, prices))
        portfolio_value = base_cash + invested_value

        exposure = float(np.dot(np.abs(holdings_vec), np.abs(prices)))
        nav_baseline = max(abs(portfolio_value), 1e-8)
        if exposure > float(self.config.max_leverage) * nav_baseline + 1e-6:
            raise ValueError(
                "Resolved holdings exceed leverage constraint: "
                f"gross={exposure:.2f}, nav={portfolio_value:.2f}, "
                f"max_leverage={self.config.max_leverage}."
            )

        self.cash = float(base_cash)
        self.holdings = holdings_vec.astype(np.int64, copy=False)
        self.asset_history = [float(portfolio_value)]
        self.date_history = [self.dates[self.current_idx]]
        self.peak_value = float(portfolio_value)
        self.last_turnover = 0.0
        self.last_nav_return = 0.0
        self.last_log_return = 0.0
        self.last_drawdown = 0.0
        self.last_commission_cost = 0.0
        self.last_slippage_cost = 0.0
        self.last_cost_ratio = 0.0
        self.last_returns = np.zeros(len(self.symbols), dtype=np.float64)
        if portfolio_value > 0:
            weights = (self.holdings * prices) / portfolio_value
            cash_fraction = self.cash / portfolio_value
        else:
            weights = np.full(len(self.symbols), 1.0 / len(self.symbols), dtype=np.float64)
            cash_fraction = 1.0
        self.last_weights = weights
        self.last_cash_fraction = cash_fraction
        self.prev_weights = weights.copy()
        self.prev_cash_fraction = cash_fraction
        self.last_gross_value = float(portfolio_value)
        self.initial_cash = float(portfolio_value)

    def resolve_portfolio_start_index(self, state: Mapping[str, Any]) -> int:
        """Public wrapper for determining the start index for a portfolio override."""

        return self._resolve_start_index(state)

    def price_map(self, index: Optional[int] = None) -> Dict[str, float]:
        """Return a symbol->price mapping at the requested episode index."""

        idx = self.current_idx if index is None else int(index)
        if idx < 0 or idx >= self.num_prices:
            raise ValueError(f"Index {idx} outside valid price history range (0, {self.num_prices - 1}).")
        return {
            symbol: float(self.price_matrix[idx, slot])
            for slot, symbol in enumerate(self.symbols)
        }

    def _info(self, *, reward_unscaled: float) -> Dict[str, Any]:
        prices = self.price_matrix[self.current_idx]
        portfolio_value = self.cash + float(np.dot(self.holdings, prices))
        return {
            "cash": float(self.cash),
            "holdings": self.holdings.astype(np.int64, copy=False),
            "prices": prices.astype(np.float32, copy=False),
            "returns": self.last_returns.astype(np.float32, copy=False),
            "portfolio_value": float(portfolio_value),
            "nav": float(portfolio_value),
            "nav_return": float(self.last_nav_return),
            "log_return": float(self.last_log_return),
            "turnover": float(self.last_turnover),
            "drawdown": float(self.last_drawdown),
            "commission_cost": float(self.last_commission_cost),
            "slippage_cost": float(self.last_slippage_cost),
            "cost_ratio": float(self.last_cost_ratio),
            "reward_unscaled": float(reward_unscaled),
            "date": self.dates[self.current_idx],
            "weights": self.last_weights.astype(np.float32, copy=False),
            "cash_fraction": float(self.last_cash_fraction),
            "gross_value": float(self.last_gross_value),
            # Safety/turbulence diagnostics (best effort)
            "q_port": float(self._turb_map.get(self.dates[self.current_idx], (np.nan, np.nan))[0]) if self._turb_map else np.nan,
            "q_sys": float(self._turb_map.get(self.dates[self.current_idx], (np.nan, np.nan))[1]) if self._turb_map else np.nan,
            "regime": self._regime_state,
            "g_max": float(self._current_gmax),
            "safety_weight_throttle": bool(getattr(self, "last_weight_throttle", False)),
            "safety_quantity_scale": float(getattr(self, "last_quantity_scale", 1.0)),
            "buy_freeze_hits": int(getattr(self, "last_buy_freeze_hits", 0)),
            "cap_hit_ratio": float(getattr(self, "last_cap_hit_ratio", 0.0)),
        }

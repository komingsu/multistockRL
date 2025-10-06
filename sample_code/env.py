from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

matplotlib.use("Agg")


class StockTradingEnv(gym.Env):
    """
    Parameters:
    ----------
    df : pd.DataFrame
        주식 시장 데이터를 포함하는 데이터프레임.
    stock_dim : int
        거래할 주식 종목의 수.
    hmax : int
        한 번에 거래할 수 있는 최대 주식 수.
    initial_amount : int
        에이전트가 거래를 시작할 때 보유하고 있는 초기 금액.
    num_stock_shares : list[int]
        각 주식 종목에 대해 에이전트가 보유하고 있는 주식 수의 리스트.
    buy_cost_pct : list[float]
        매수 거래에 따른 비용 비율.
    sell_cost_pct : list[float]
        매도 거래에 따른 비용 비율.
    reward_scaling : float
        보상의 스케일링 인자.
    state_space : int
        관측 공간의 차원.
    action_space : int
        행동 공간의 차원.
    tech_indicator_list : list[str]
        기술적 지표의 이름을 포함하는 리스트.
    turbulence_threshold : float, optional
        터뷸런스 임계값. 이 값 이상이면 시장이 불안정하다고 간주.
    risk_indicator_col : str, default "turbulence"
        위험 지표의 컬럼 이름.
    make_plots : bool, default False
        시뮬레이션 중에 플롯을 만들지 여부.
    print_verbosity : int, default 10
        출력의 상세 수준을 조절하는 정수.
    day : int, default 0
        에피소드 시작일을 나타내는 정수.
    initial : bool, default True
        초기화 여부. 초기화할 경우 True.
    previous_state : list, default []
        이전 상태를 기억하는 리스트.
    model_name : str, default ""
        모델 이름.
    mode : str, default ""
        현재 모드 (예: 훈련, 테스트 등).
    iteration : str, default ""
        현재 반복 횟수.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        """
        주식 매도 로직을 수행하는 함수.

        Parameters:
        index : int
            현재 매도하려는 주식의 인덱스.
        action : int
            매도할 주식의 수. 양수인 경우 매수를 의미하며, 음수인 경우 매도를 의미함.

        Returns:
        sell_num_shares : int
            실제 매도한 주식의 수.

        주의: action 값이 음수일 경우만 매도가 실행됨.
        """

        def _do_sell_normal():
            # 주식이 매도 가능한 상태인지 확인 (기술 지표를 통해 매매 가능 여부를 판단)
            if self.state[index + 2 * self.stock_dim + 1] != True:
                # 주식의 현재 가격이 0 이상일 때에만 매도 가능
                if self.state[index + self.stock_dim + 1] > 0:
                    # 현재 보유한 주식 수와 매도하려는 주식 수 중 더 작은 값으로 매도 수량 결정
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    # 매도 금액 계산 (매도 수량 * 주식 가격 * (1 - 매도 비용 비율))
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # 현금 자산 업데이트 (매도 금액 추가)
                    self.state[0] += sell_amount

                    # 주식 보유 수 업데이트 (매도 수량만큼 감소)
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    # 매도 비용 업데이트
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    # 거래 수 업데이트
                    self.trades += 1
                else:
                    # 매도할 주식이 없는 경우, 매도 수량은 0
                    sell_num_shares = 0
            else:
                # 주식이 매도 불가능한 상태인 경우, 매도 수량은 0
                sell_num_shares = 0

            return sell_num_shares

        # 시장의 불안정성이 임계값 이상인지에 따라 행동을 결정
        if self.turbulence_threshold is not None:
            # 시장 불안정성이 높을 때
            if self.turbulence >= self.turbulence_threshold:
                # 주식 가격이 0 이상이고, 보유 주식이 있을 때 모든 주식 매도
                if self.state[index + 1] > 0 and self.state[index + self.stock_dim + 1] > 0:
                    # 보유한 모든 주식 매도
                    sell_num_shares = self.state[index + self.stock_dim + 1]
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # 현금 자산 업데이트 (매도 금액 추가)
                    self.state[0] += sell_amount
                    # 주식 보유 수를 0으로 설정
                    self.state[index + self.stock_dim + 1] = 0
                    # 매도 비용 업데이트
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    # 거래 수 업데이트
                    self.trades += 1
                else:
                    # 매도할 주식이 없는 경우, 매도 수량은 0
                    sell_num_shares = 0
            else:
                # 시장이 비교적 안정적일 때 정상적인 매도 로직 수행
                sell_num_shares = _do_sell_normal()
        else:
            # 불안정성 임계값이 설정되어 있지 않을 때 정상적인 매도 로직 수행
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """
        주식 매수 로직을 수행하는 함수입니다.

        Parameters:
        index : int
            현재 매수하려는 주식의 인덱스입니다.
        action : int
            매수할 주식의 수입니다. 양수일 경우 매수를 의미합니다.

        Returns:
        buy_num_shares : int
            실제 매수한 주식의 수입니다.

        이 함수는 에이전트가 결정한 액션에 따라 주식을 매수합니다. 주식 매수가 가능한지 여부를 
        체크하고, 가능할 경우에만 매수를 진행합니다.
        """

        def _do_buy():
            # 주식 매수 가능 여부 체크 (기술 지표 등을 통해 판단)
            if self.state[index + 2 * self.stock_dim + 1] != True:
                # 주식 가격이 0 이상인 경우에만 매수를 진행합니다 (데이터가 누락되지 않은 날짜에만 매수)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # 매수 가능한 주식 수를 계산할 때 거래 비용을 고려합니다.

                # 실제로 매수할 수 있는 주식 수를 계산합니다 (가용 금액과 액션 중 작은 값)
                buy_num_shares = min(available_amount, action)
                # 매수 금액을 계산합니다 (주식 가격 * 매수 수량 * (1 + 매수 비용 비율))
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                # 현금 잔고를 업데이트합니다 (매수 금액만큼 감소)
                self.state[0] -= buy_amount

                # 보유 주식 수를 업데이트합니다 (매수 수량만큼 증가)
                self.state[index + self.stock_dim + 1] += buy_num_shares

                # 매수 비용을 업데이트합니다.
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                # 거래 횟수를 증가시킵니다.
                self.trades += 1
            else:
                # 매수 불가능 상태일 경우, 매수 수량은 0입니다.
                buy_num_shares = 0

            return buy_num_shares

        # 시장의 불안정성이 정의된 임계값 이하일 때만 매수를 진행합니다.
        if self.turbulence_threshold is None:
            # 불안정성 임계값이 설정되지 않은 경우, 정상 매수 로직을 수행합니다.
            buy_num_shares = _do_buy()
        else:
            # 불안정성 임계값이 설정된 경우, 불안정성이 임계값 이하일 때만 매수를 수행합니다.
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                # 불안정성이 임계값 이상일 경우, 매수를 진행하지 않고 매수 수량을 0으로 설정합니다.
                buy_num_shares = 0

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        """
        환경에서 한 스텝을 진행하고 결과를 반환하는 메서드입니다.

        Parameters:
        actions : np.array
            에이전트가 선택한 행동들의 배열로, 각 주식에 대한 매수 또는 매도 액션을 포함합니다.

        Returns:
        self.state : np.array
            새로운 상태의 배열입니다.
        self.reward : float
            이번 스텝에서 얻은 보상의 양입니다.
        self.terminal : bool
            에피소드가 종료되었는지 여부를 나타냅니다. 모든 거래일이 끝나면 True가 됩니다.
        False : bool
            추가 정보 제공용, 여기서는 사용되지 않으므로 항상 False를 반환합니다.
        {} : dict
            추가 정보 제공용, 여기서는 사용되지 않으므로 빈 딕셔너리를 반환합니다.

        이 메서드는 에이전트의 행동 배열을 받아 각 주식에 대한 매수/매도를 진행하고,
        새로운 상태, 보상, 종료 여부 등을 계산합니다.
        """
        # 에피소드 종료 여부 판단: 현재 일자가 데이터의 유일한 인덱스 개수보다 많거나 같으면 종료
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        # 만약 에피소드가 종료되었다면 (self.terminal == True)
        if self.terminal:
            # 에피소드 종료 시 수행할 로직
            
            # 포트폴리오의 총 자산 가치를 계산
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            
            # 총 자산 가치의 시계열 데이터를 데이터프레임으로 저장
            df_total_value = pd.DataFrame(self.asset_memory)
            
            # 에피소드 전체에서의 총 보상을 계산 (초기 자산 대비 총 자산의 증가)
            tot_reward = end_total_asset - self.asset_memory[0]
            
            # 계산된 총 자산 가치와 날짜를 데이터프레임에 추가
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            
            # 일일 수익률을 계산하고 데이터프레임에 추가
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            
            # 샤프 지수 계산 (일일 수익률의 평균 / 일일 수익률의 표준편차)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5) *  # 연간화 계수
                    df_total_value["daily_return"].mean() /
                    df_total_value["daily_return"].std()
                )
            
            # 보상의 시계열 데이터를 데이터프레임으로 저장
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            
            # 설정된 print_verbosity에 따라 로깅 정보 출력
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
                
            # 모델 이름과 모드가 설정되었다면 결과를 파일로 저장
            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # 마지막으로 현재 상태, 보상, 종료 여부 등을 반환합니다
            return self.state, self.reward, self.terminal, False, {}

        else:
            # 에이전트가 결정한 액션을 확장하여 실제 매수 또는 매도할 주식 수를 결정합니다.
            actions = actions * self.hmax  # 액션 값은 -1에서 1 사이로 스케일되어 있으며, 여기서 hmax를 곱해 실제 매수/매도 수량으로 변환합니다.
            actions = actions.astype(int)  # 매수/매도는 주식의 일부분을 살 수 없으므로 정수로 변환합니다.
            
            # 시장의 불안정성이 설정된 임계값을 초과하는지 검사합니다.
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    # 시장이 불안정한 경우, 모든 주식을 매도하는 액션으로 설정합니다.
                    actions = np.array([-self.hmax] * self.stock_dim)
                    
            # 매 스텝 시작 전의 총 자산을 계산합니다.
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            # 액션 배열을 정렬하여 매도해야 할 주식과 매수해야 할 주식의 인덱스를 각각 구합니다.
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]  # 매도 액션이 있는 인덱스
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]  # 매수 액션이 있는 인덱스
            # 매도 액션을 수행합니다.
            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")
                
            # 매수 액션을 수행합니다.
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])
                
            # 수행된 액션들을 기록합니다.
            self.actions_memory.append(actions)
            self.actions_memory.append(actions)

            # state: s -> s+1, 다음 거래일로 넘어갑니다.
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # 시장의 불안정성을 업데이트합니다.
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            # 상태를 업데이트합니다.
            self.state = self._update_state()
            
            # 스텝 종료 후의 총 자산을 계산합니다.
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # 자산 변화를 기록합니다.
            self.asset_memory.append(end_total_asset)
            # 거래일을 기록합니다.
            self.date_memory.append(self._get_date())
            # 이번 스텝에서의 보상을 계산합니다.
            self.reward = end_total_asset - begin_total_asset
            # 보상을 기록합니다.
            self.rewards_memory.append(self.reward)
            # 보상에 스케일을 적용합니다.
            self.reward = self.reward * self.reward_scaling
            # 현재 상태를 기록합니다.
            self.state_memory.append(self.state)
            
        # 현재 상태, 보상, 종료 여부, 추가 정보를 반환합니다.
        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        환경을 초기 상태로 리셋하는 메서드입니다.

        Parameters:
        seed : int, optional
            난수 생성기 시드 값입니다.
        options : dict, optional
            리셋 옵션을 지정하는 딕셔너리입니다.

        Returns:
        self.state : np.array
            리셋된 환경의 초기 상태 배열입니다.
        
        이 메서드는 새로운 에피소드를 시작하기 전에 호출됩니다. 현재 일자를 0으로 설정하고
        데이터프레임에서 해당 일자의 데이터를 불러와 상태를 초기화합니다.
        """

        # 거래를 시작할 날짜를 0으로 설정하여 새로운 에피소드를 시작합니다.
        self.day = 0
        # 현재 날짜에 해당하는 데이터를 데이터프레임에서 가져옵니다.
        self.data = self.df.loc[self.day, :]
        # 상태를 초기 상태로 설정하는 내부 메서드를 호출합니다.
        self.state = self._initiate_state()

        # 만약 초기 상태에서 시작하는 경우
        if self.initial:
            # 초기 자산 메모리를 설정합니다. 이는 초기 현금과 각 주식의 초기 주식 수와 가격을 기반으로 계산됩니다.
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        # 만약 이전 상태에서 시작하는 경우
        else:
            # 이전 에피소드의 마지막 총 자산을 계산합니다.
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            # 자산 메모리를 이전 에피소드의 마지막 총 자산으로 설정합니다.
            self.asset_memory = [previous_total_asset]

        # 터뷸런스와 비용, 거래 수를 0으로 설정합니다.
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # 에피소드 종료 여부를 False로 설정합니다.
        self.terminal = False
        # 에피소드를 나타내는 숫자를 1 증가시킵니다.
        self.episode += 1

        # 보상과 행동 메모리를 비우고 현재 날짜를 메모리에 추가합니다.
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        # 초기 상태를 반환합니다.
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        """
        환경의 초기 상태 또는 이전 상태를 기반으로 현재 상태를 설정하는 메서드입니다.

        Returns:
        state : list
            현재 상태를 나타내는 리스트로, 현금 잔액, 각 주식의 가격, 보유 주식 수,
            그리고 선택된 기술적 지표들의 값을 포함합니다.

        이 메서드는 reset 메서드에 의해 호출되며, 에이전트의 초기 상태 또는 이전 상태를 준비합니다.
        """

        # 환경이 처음 시작할 때의 상태를 설정합니다.
        if self.initial:
            # 단일 주식이 아닌 여러 주식에 대한 상태를 초기화할 경우
            if len(self.df.tic.unique()) > 1:
                # 초기 자본을 포함하여 상태 리스트를 구성합니다.
                state = (
                    [self.initial_amount]  # 시작할 때 보유 현금
                    + self.data.close.values.tolist()  # 모든 주식의 가격
                    + self.num_stock_shares  # 에이전트가 초기에 보유한 각 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # 단일 주식에 대한 상태를 초기화할 경우
                state = (
                    [self.initial_amount]  # 시작할 때 보유 현금
                    + [self.data.close]  # 주식의 가격
                    + [0] * self.stock_dim  # 보유 주식 수는 0으로 설정
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # 이전 상태를 사용하여 상태를 설정할 경우
            if len(self.df.tic.unique()) > 1:
                # 여러 주식에 대한 상태를 이전 상태에서 업데이트합니다.
                state = (
                    [self.previous_state[0]]  # 이전 자본
                    + self.data.close.values.tolist()  # 모든 주식의 가격
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]  # 이전에 보유한 각 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # 단일 주식에 대한 상태를 이전 상태에서 업데이트합니다.
                state = (
                    [self.previous_state[0]]  # 이전 자본
                    + [self.data.close]  # 주식의 가격
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]  # 이전에 보유한 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )

        # 설정된 상태를 반환합니다.
        return state

    def _update_state(self):
        """
        현재 거래일의 주식 가격과 기술적 지표들의 값을 반영하여 환경의 상태를 업데이트하는 메서드입니다.

        Returns:
        state : list
            업데이트된 상태를 나타내는 리스트입니다. 현금 잔액, 각 주식의 가격, 보유 주식 수,
            그리고 선택된 기술적 지표들의 값을 포함합니다.

        이 메서드는 매 거래일 마다 호출되어 에이전트의 상태를 최신 정보로 업데이트합니다.
        """

        # 여러 주식에 대한 환경일 경우
        if len(self.df.tic.unique()) > 1:
            # 현재 보유 현금은 유지하고 주식 가격을 업데이트합니다.
            state = (
                [self.state[0]]  # 현재 보유 현금
                + self.data.close.values.tolist()  # 새로운 거래일의 각 주식 가격
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # 현재 보유 주식 수는 유지
                # 선택된 기술적 지표들의 새로운 값을 추가합니다.
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        # 단일 주식에 대한 환경일 경우
        else:
            # 현재 보유 현금은 유지하고 주식 가격을 업데이트합니다.
            state = (
                [self.state[0]]  # 현재 보유 현금
                + [self.data.close]  # 새로운 거래일의 주식 가격
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # 현재 보유 주식 수는 유지
                # 선택된 기술적 지표들의 새로운 값을 추가합니다.
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        # 업데이트된 상태를 반환합니다.
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_state_memory(self):
        """
        거래 과정에서의 상태를 기록하여 메모리에 저장하는 메서드입니다.

        이 메서드는 각 스텝마다의 에이전트 상태를 데이터프레임 형태로 저장합니다.
        여러 주식을 다루는 경우와 단일 주식을 다루는 경우로 나뉘어 처리합니다.

        Returns:
        df_states : pd.DataFrame
            각 거래일에 대한 상태 정보를 포함하는 데이터프레임입니다.
        """

        # 거래일 리스트를 가져옵니다. 마지막 거래일은 제외합니다.
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        # 상태 메모리에서 상태 리스트를 가져옵니다.
        state_list = self.state_memory

        # 여러 주식에 대한 처리: 여러 주식을 다루는 경우 해당되는 컬럼 이름을 지정합니다.
        # 제공된 데이터에 맞게 컬럼 이름을 설정합니다.  
        if len(self.df.tic.unique()) > 1:
            # 여러 주식을 다루는 경우의 상태 데이터프레임을 생성합니다.
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    # "Bitcoin_price", "Gold_price" 등의 컬럼은 예시로 사용되었습니다.
                    # 실제 데이터에 맞는 컬럼으로 대체합니다.
                    "close_price",
                    # "Bitcoin_num", "Gold_num" 등의 컬럼은 보유 주식 수를 나타냈습니다.
                    # "num_shares" 등 실제 데이터에 맞는 컬럼으로 대체할 수 있습니다.
                    "num_shares",
                    # "Bitcoin_Disable", "Gold_Disable" 등의 컬럼은 해당 주식의 거래 가능 여부를 나타냈습니다.
                    # 실제 사용하는 데이터에 따라 필요 없는 컬럼은 제거할 수 있습니다.
                    # 여기서는 예시로 든 컬럼을 제거하고, 실제 데이터의 특성에 맞는 컬럼 이름으로 대체합니다.
                    "volume",
                    "vix",
                    "turbulence",
                    # 기술적 지표 등 추가 컬럼
                    "sma_10",
                    "rsi",
                    # 나머지 필요한 컬럼을 추가합니다.
                ],
            )
            df_states.index = df_date.date
        else:
            # 단일 주식에 대한 처리: 상태 데이터프레임을 생성합니다.
            df_states = pd.DataFrame({"date": date_list, "states": state_list})

        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            action_list = action_list[::2]
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        e = VecNormalize(e, norm_obs=True, norm_reward=True, clip_obs=10.0)
        return e, obs
import gym
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class StockTradingEnv_lookback(gym.Env):
    def __init__(self, config):
        # self.file_path = config["file_path"]                   # model input data
        self.df = config["df"]
        self.check_df()
        self.stock_dim = self.df.tic.nunique()                 # number of unique stocks
        self.stocks = self.df.tic.sort_values().unique()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim, ))
        self.hmax = config["hmax"]                             # maximum number of shares to trade
        self.buy_cost_pct = config["buy_cost_pct"]             # buy cost
        self.sell_cost_pct = config["sell_cost_pct"]           # sell cost
        self.reward_scaling = config["reward_scaling"]         # scaling factor for reward, good for training
        self.initial_cash = config["initial_cash"] if "initial_cash" in config.keys() else 1e6
        self.initial_total_asset = config["initial_cash"]       # In order to meet finrl requirements
        self.T_plus_one = config["T_plus_one"] if "T_plus_one" in config.keys() else False
        
        # about lookbacks
        self.lookback_days = config["lookback_days"]
        self.total_time_steps = self.df.time.nunique()
        self.total_time_steps -= (self.lookback_days-1)
        self.dates = self.df.time.sort_values().unique()[(self.lookback_days-1):]
        print(f"Total timesteps: {self.total_time_steps}. Starting from {self.dates[0]} to {self.dates[-1]}")
        
        # about trading
        self.buy_price_on = config["price_on"] if "price_on" in config.keys() else "close"
        self.sell_price_on = config["price_on"] if "price_on" in config.keys() else "close"
        self.relative_action = config["relative_action"] if "relative_action" in config.keys() else False  # True, False
        self.full_buy = config["full_buy"] if "relative_action" in config.keys() else False                # True False
        
        # about showing
        self.cash_showing = config["cash_showing"] if "cash_showing" in config.keys() else "ori"                         # "ori", "log","no" 
        self.stock_hold_showing = config["stock_hold_showing"] if "stock_hold_showing" in config.keys() else "actual_holding"  # "actual_holding","pct_holding","not_showing"
        self.holding_at_beginning = config["holding_at_beginning"] if "holding_at_beginning" in config.keys() else "zero_holding"  # "zero_holding","random_holding","same_pct"
        self.output_type = config["output_type"]   # "numpy_array","torch_tensor"

        # about observations
        unselected_columns = ["tic","time","index"]
        self.selected_columns = [i for i in self.df.columns if i not in unselected_columns]
        self.observation_space = len(self.selected_columns) * self.stock_dim *self.lookback_days
        if self.stock_hold_showing != "not_showing":
            self.observation_space += self.stock_dim
        if self.cash_showing != "no":
            self.observation_space += 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high = np.inf, shape=(self.observation_space,))

        # about device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.episode = 0 
        # self._read_file()
    
    def check_df(self):
        assert "tic" in self.df.columns, "Please rename the stock name column to \' tic\'"
        assert "close" in self.df.columns, "Please rename the stock close price column to \' close\'"
        assert "time" in self.df.columns, "Please rename the time steps column to \' time\'"

    def _read_file(self):
        pass
        
    def reset(self):
        # prepare observations and prices
        self.data = []
        self.buy_price = []
        self.sell_price = []
        for _,tmp in self.df.groupby("time"):
            self.data.append(tmp[self.selected_columns].values) # [stocks, features]
            self.buy_price.append(tmp[self.buy_price_on].values) # [stocks]
            self.sell_price.append(tmp[self.sell_price_on].values) # [stocks]

        self.data = torch.tensor(np.stack(self.data,axis=0).astype(float))          # [timesteps, stocks, features]
        self.buy_price = torch.tensor(np.stack(self.buy_price,axis=0).astype(float))  # [timesteps, stocks]
        self.sell_price = torch.tensor(np.stack(self.sell_price,axis=0).astype(float))  # [timesteps, stocks]
        
        self.data = self.data.unfold(dimension=0,size = self.lookback_days,step = 1)  # (timesteps,stocks,features,lookback_days)
        self.buy_price = self.buy_price.unfold(dimension=0,size = self.lookback_days,step = 1)  # (timesteps,stocks,lookback_days)
        self.sell_price = self.sell_price.unfold(dimension=0,size = self.lookback_days,step = 1)  # (timesteps,stocks,lookback_days)
        self.middle_price = (self.buy_price + self.sell_price)/2   # (timesteps,stocks,lookback_days)
        
        # initialize 
        self.buy_trade_cost = []
        self.sell_trade_cost = []
        self.today = 0
        self.day_count = 0                           
        self.reward = 0                                                         
        self.trades = 0                              
        self.terminal = False
        self.episode += 1
        self.asset_memory = [torch.tensor(self.initial_cash,device=self.device)]
        self.rewards_memory = []  
        self.actions_memory = []

        self.today_data = self.data[self.today].to(self.device)          # stocks,features,lookback_days
        self.today_buy_price = self.buy_price[self.today][:,-1].to(self.device) # stocks
        self.today_sell_price = self.sell_price[self.today][:,-1].to(self.device) # stocks
        self.today_middle_price = self.middle_price[self.today][:,-1].to(self.device) # stocks
        self.today_cash = torch.tensor(self.initial_cash,device=self.device)

        # self.lastday_data = self.data[self.today].to(self.device)          # stocks,features,lookback_days
        # self.lastday_price = self.middle_price[self.today].to(self.device) # stocks,lookback_days

        # holding at the beginning
        if self.holding_at_beginning == "zero_holding":
            self.today_hold = torch.zeros(self.stock_dim,device=self.device)  # stocks
            self.buy_trade_cost.append(torch.tensor(0.0,device=self.device))
            
        else:
            with torch.no_grad():
                if self.holding_at_beginning == "random_holding":
                    self.today_hold = torch.softmax(torch.rand(self.stock_dim,device=self.device),dim = -1)  # stocks

                elif self.holding_at_beginning == "same_pct":
                    self.today_hold = torch.softmax(torch.ones(self.stock_dim,device=self.device),dim = -1) # stocks
            
            self.today_hold *= self.today_cash
            self.today_hold /= (self.today_buy_price * (1 + self.buy_cost_pct))
            self.today_hold = torch.floor(self.today_hold)
            self.today_cash -= torch.sum(self.today_buy_price* (1 + self.buy_cost_pct)* self.today_hold,dim = -1)
            self.buy_trade_cost.append(torch.sum(self.today_buy_price* self.buy_cost_pct* self.today_hold,dim = -1))

        self.sell_trade_cost.append(torch.tensor(0.0,device=self.device))
        self.actions_memory.append(self.today_hold.cpu().numpy())
        
        return self.make_state()

    def update_state(self):
        self.today +=1 
        self.terminal = self.today == (self.total_time_steps-1)
        if not self.terminal:
            
            self.today_data = self.data[self.today].to(self.device)          # stocks,features,lookback_days
            self.today_buy_price = self.buy_price[self.today][:,-1].to(self.device) # stocks
            self.today_sell_price = self.sell_price[self.today][:,-1].to(self.device) # stocks
            self.today_middle_price = self.middle_price[self.today][:,-1].to(self.device) # stocks    

    def step(self,actions):   
        # updating
        self.day_count += 1

        last_data = self.today_data
        last_buy_price = self.today_buy_price
        last_sell_price = self.today_sell_price
        last_middle_price = self.today_middle_price
        last_cash = self.today_cash
        last_hold = self.today_hold
        last_asset = last_cash + torch.sum( last_middle_price * last_hold )
        
        self.update_state()

        today_data = self.today_data
        today_buy_price = self.today_buy_price
        today_sell_price = self.today_sell_price
        today_middle_price = self.today_middle_price

        # process actions
        actions = torch.tensor(actions,device=self.device)
        actions *= self.hmax
        actions = actions.to(dtype = torch.float64)
        assert len(actions.shape) == 1, "The actions mast be one dimension"
        if self.relative_action:
            actions -= torch.mean(actions,keepdim=True,dim = -1)
        
        # cal rewards
        next_hold = last_hold + actions
        
        # check next_hold > 0:
        zero_next_hold = torch.zeros_like(next_hold,dtype=torch.float64).to(self.device)
        next_hold = torch.where( next_hold < 0, zero_next_hold, next_hold)
        hold_change = next_hold - last_hold

        # sell
        zero_actions = torch.zeros_like( hold_change,dtype=torch.float64 ).to(self.device)
        sell_actions = torch.where( hold_change >= 0, zero_actions, hold_change) # <0
        sell_actions = torch.ceil( sell_actions )        # Integer hands
        sell_money = torch.sum( - last_sell_price * sell_actions * ( 1 - self.sell_cost_pct ))
        assert not torch.where(sell_money < 0)[0].any()
        sell_cost = torch.sum( - last_sell_price * sell_actions * self.sell_cost_pct)
        self.sell_trade_cost.append(sell_cost)

        avaliable_cash = sell_money + last_cash
        ones_actions = torch.ones_like( hold_change,dtype=torch.float64 ).to(self.device)
        not_sell_actions = torch.where( sell_actions == 0, ones_actions, zero_actions) # <0
        if torch.sum(not_sell_actions,dim = -1) != torch.sum(ones_actions):
            # buy
            buy_actions = torch.where( hold_change <= 0, zero_actions, hold_change )  # >0
            buy_money = last_buy_price * buy_actions
            buy_money_all = torch.sum( last_buy_price * buy_actions * (1 + self.sell_cost_pct))
            buy_money_sum = torch.sum(buy_money,dim = -1)
        
            if buy_money_sum == 0:
                buy_money_pct = torch.softmax(not_sell_actions,dim = -1)
            else:
                buy_money_pct = buy_money / buy_money_sum

            # full buy and no avaliable cash
            if self.full_buy:
                buy_money = avaliable_cash * buy_money_pct
                buy_actions = buy_money / (last_buy_price* (1 + self.sell_cost_pct))
            else:  # not full buy
                if buy_money_all < avaliable_cash:
                    buy_money = avaliable_cash * buy_money_pct
                    buy_actions = buy_money / (last_buy_price* (1 + self.sell_cost_pct))
            buy_actions = torch.floor( buy_actions )        # Integer hands
        else:
            buy_actions = zero_actions

        buy_money = torch.sum( last_buy_price * buy_actions * (1 + self.sell_cost_pct))
        buy_cost = torch.sum(last_buy_price * buy_actions * self.sell_cost_pct)
        self.buy_trade_cost.append(buy_cost)
        next_cash = avaliable_cash - buy_money
        assert not torch.where(next_cash<0)[0].any()

        actions = buy_actions + sell_actions
        next_hold = last_hold + actions
        
        self.today_cash = next_cash
        self.today_hold = next_hold

        next_asset = next_cash  + torch.sum( self.today_middle_price * next_hold )
        self.reward = next_asset - last_asset   
        self.rewards_memory.append(self.reward)
        self.asset_memory.append(next_asset) 
        self.actions_memory.append(actions.cpu().numpy())
        # self.cash_memory.append(next_cash)

        if self.terminal:
            print(f"Episode: {self.episode}")
            total_reward = torch.stack(self.rewards_memory,dim = -1)
            total_reward = torch.sum(total_reward)
            total_reward_pct = total_reward/self.initial_total_asset
            print(f"Cumulated Return:  {total_reward}.Cumulated Return Rate: {total_reward_pct}")
            
        return self.make_state(),self.make_reward(), self.terminal,{}

    def render(self, figsize= (20,5)):
        pd.DataFrame([0]+[i.cpu().item() for i in self.rewards_memory]).plot(title ="rewards_memory",figsize = figsize,legend = False)
        plt.show()
        pd.DataFrame([i.cpu().item() for i in self.asset_memory]).plot(title ="asset_memory",figsize = figsize,legend = False)

    def make_state(self):
        with torch.no_grad():
            output = torch.flatten(self.today_data,start_dim=0)
            
            if self.stock_hold_showing != "not_showing":
                stock_hold = self.today_hold
                if self.stock_hold_showing == "pct_holding":
                    stock_hold /= torch.sum(stock_hold,-1,keepdim=True)
                output = torch.concat([stock_hold,output],dim = -1)

            if self.cash_showing != "no":
                today_cash = self.today_cash 
                if self.cash_showing == "log":
                    assert today_cash > 0 , (today_cash,self.today)
                    today_cash = torch.log1p(today_cash)
                output = torch.concat([today_cash.reshape(-1),output],dim = -1)

        if self.output_type == "numpy_array":
            output = output.cpu().numpy()
        return output

    def make_reward(self):
        output = self.reward.detach().cpu() * self.reward_scaling
        if self.output_type == "numpy_array":
            output = output.numpy()
            
        return output

    def save_cost_memory(self):
        date_list = self.dates      # date
        buy_cost_list = [i.cpu().item() for i in self.buy_trade_cost]     # buy_cost_list
        sell_cost_list = [i.cpu().item() for i in self.sell_trade_cost]     # sell_cost_list
        df_account_value = pd.DataFrame({'date':date_list,'buy_trade_cost':buy_cost_list,'sell_trade_cost':sell_cost_list})
        return df_account_value

    def save_asset_memory(self):
        date_list = self.dates      # date
        asset_list = [i.cpu().item() for i in self.asset_memory]     # assets
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.stocks
        df_actions.index = self.dates

        return df_actions
    
    def save_reward_memory(self):
        date_list = self.dates      # date
        rewards_list =[0]+[i.cpu().item() for i in self.rewards_memory]     # reward
        print(len(date_list),len(rewards_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':rewards_list})
        return df_account_value

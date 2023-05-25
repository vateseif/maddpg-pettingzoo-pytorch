from copy import deepcopy
from typing import List, Optional


import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, use_mpc=False):
        self.use_mpc = use_mpc
        if use_mpc:
          self.mpc_layer = MPCLayer()
        self.actor = MLPNetwork(obs_dim, act_dim).to(device)
        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])
        obs = obs.to(device)
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        batch_size = logits.shape[0]
        if self.use_mpc:
          logits = torch.tanh(logits)
          actions = self.mpc_layer.solve(obs, logits)  # (B, 2)
          actions = torch.cat((torch.zeros(batch_size, 1).to(device), actions), 1)
        else:
          actions = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return actions, logits
        return actions

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        obs = obs.to(device)
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        batch_size = logits.shape[0]
        if self.use_mpc:
          logits = torch.tanh(logits)
          actions = self.mpc_layer.solve(obs, logits)  # (B, 2)
          actions = torch.cat((torch.zeros(batch_size, 1).to(device), actions), 1)
        else:
          actions = F.gumbel_softmax(logits, hard=True)
        return actions

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


class MPCLayer:

  def __init__(self) -> None:
    self.N = 1    
    self.Ns = 4
    self.Na = 4
    self.T = 5
    self.dt = 0.1              # TODO get val from env
    self.tau = 0.25            # TODO get val from env
    self.eps = 1e-3
    self.size = 0.075
    self.max_speed = None
    self.sensitivity = 5.
    self.LQR_ITER = 100
    self.u_upper = 1.
    self.u_lower = 0.

    self.u_init = None
    self.batch_size = None

    # model dy namics
    self.Dx = None
    
    # cost
    self.cost = None

    # controller
    self.controller = None

  def _init_model(self):
    # dynamics of 1 single agent
    A1 = torch.tensor([[0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, -self.tau/self.dt, 0], 
                  [0, 0, 0, -self.tau/self.dt]])
    B1 = torch.tensor([[0, 0, 0, 0], 
                  [0, 0, 0, 0], 
                  [1, -1, 0, 0], 
                  [0, 0, 1, -1]]) 
    # discrete dynamics
    Ad = torch.eye(A1.shape[0]) + A1*self.dt
    Bd = B1*self.dt*self.sensitivity
    # extend dynamics over horizon and batch size
    A = Ad.repeat(self.T, self.batch_size, 1, 1)
    B = Bd.repeat(self.T, self.batch_size, 1, 1)
    F = torch.cat((A, B), dim=3).to(device)
    return LinDx(F)

  def _init_cost(self, xd: torch.Tensor):
    # Quadratic cost
    xd = xd                                                                       # (B, Ns) desired states
    w = torch.tensor([1., 1., 1e-1, 1e-1]).to(device)                             # (Ns,) state weights
    q = torch.cat((w, 1e-1*torch.ones(self.Na).to(device)))                     # (Ns+Na,) state-action weights
    Q = torch.diag(q).repeat(self.T, self.batch_size, 1, 1)                       # (T, B, Ns+Na, Ns+Na) weight matrix
    px = -torch.sqrt(w) * xd                                                                  # (B, Ns) linear cost vector
    p = torch.cat((px, 1e-2*torch.ones((self.batch_size, self.Na)).to(device)), 1)   # (T, B, Ns+Na) linear cost vector for state-action
    p = p.repeat(self.T, 1, 1)
    cost = QuadCost(Q, p)

    return cost


  def solve(self, obs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    batch_size, _ = logits.shape
    # Update dynamics if batch_size changes
    if batch_size != self.batch_size:
      self.batch_size = batch_size
      self.Dx = self._init_model()
      self.u_init = None
    # Init cost wrt to desired states
    xd = torch.cat((logits, torch.zeros((self.batch_size, 2)).to(device)), 1)
    self.cost = self._init_cost(xd)
    # recreate controller using updated u_init (kind of wasteful right?)
    ctrl = mpc.MPC(self.Ns, self.Na, self.T, u_lower=self.u_lower, u_upper=self.u_upper, 
                  lqr_iter=self.LQR_ITER, exit_unconverged=False, eps=1e-2,
                  n_batch=self.batch_size, backprop=True, verbose=0, u_init=self.u_init,
                  grad_method=mpc.GradMethods.AUTO_DIFF)
    # initial state (always 0 in reference frame)
    x_init = torch.cat((torch.zeros(batch_size, 2).to(device), obs[:, :2]), 1)
    # solve mpc problem
    _, nominal_actions, _ = ctrl(x_init, self.cost, self.Dx)
    #print(logits)
    # update u_init for warming starting at next step
    #self.u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self.batch_size, self.Na).to(device)), dim=0)
    # return first action
    return nominal_actions[0]   # (B, Na)



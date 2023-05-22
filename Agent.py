from copy import deepcopy
from typing import List, Optional

import lmpc
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, use_mpc=False):
        self.use_mpc = use_mpc
        self.actor = MLPNetwork(obs_dim, act_dim).to(device)
        if use_mpc:
          self.mpc = LMPCLayer()

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
        
        logits = self.actor(obs.to(device))  # torch.Size([batch_size, action_size])
        print(logits)
        batch_dim, act_dim = logits.shape
        # action = self.gumbel_softmax(logits)
        if self.use_mpc:
          actions = []
          for i in range(batch_dim):
            x0 = np.concatenate((np.ones((2,1))*1e-3, np.array([[obs[i][0]], [obs[i][1]]])))
            action = self.mpc.act(x0, torch.reshape(logits[i], (act_dim,1)).detach().numpy())
            actions.append(torch.tensor(action))
          actions = torch.vstack(actions)
        else:
          actions = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return actions, logits
        return actions

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs.to(device))  # torch.Size([batch_size, action_size])
        batch_dim, act_dim = logits.shape
        # action = self.gumbel_softmax(logits)
        if self.use_mpc:
          actions = []
          for i in range(batch_dim):
            x0 = np.concatenate((np.ones((2,1))*1e-3, np.array([[obs[i][0]], [obs[i][1]]])))
            action = self.mpc.act(x0, torch.reshape(logits[i], (act_dim,1)).detach().numpy())
            actions.append(torch.tensor(action))
          actions = torch.vstack(actions)
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


class LMPCLayer:

  def __init__(self, locality: Optional[str]=None) -> None:
    self.N = 1
    self.locality = locality
    
    self.Ns = 4
    self.Na = 4
    self.T = 15
    self.dt = 0.1              # TODO get val from env
    self.tau = 0.25            # TODO get val from env
    self.eps = 1e-3
    self.size = 0.075
    self.max_speed = None
    self.sensitivity = 5.

    # system model
    self.sys = self._init_model()
    
    # controller
    self.controller = self._init_controller()

  def _init_model(self):
    # dynamics of 1 single agent
    A1 = np.array([[0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, -self.tau/self.dt, 0], 
                  [0, 0, 0, -self.tau/self.dt]])
    B1 = np.array([[0, 0, 0, 0], 
                  [0, 0, 0, 0], 
                  [1, -1, 0, 0], 
                  [0, 0, 1, -1]]) 
    # dynamics of n agents
    A = np.kron(np.eye(self.N), A1)
    B = np.kron(np.eye(self.N), B1)
    # discrete dynamics
    Ad = np.eye(A.shape[0]) + A*self.dt
    Bd = B*self.dt*self.sensitivity
    
    # init sys
    sys = lmpc.DistributedLTI(self.N, self.Ns, self.Na)
    sys.loadAB(Ad, Bd)

    # locality model
    if self.locality != None:
      locality = None
      sys << locality
      pass
    return sys

  def _init_controller(self):
    # controller
    controller = lmpc.LMPC(self.T)
    controller << self.sys
    
    # box constraints control inputs
    controller.addConstraint(lmpc.BoundConstraint('u', 'upper', (1-self.eps)*np.ones((self.sys.Nu,1))))
    controller.addConstraint(lmpc.BoundConstraint('u', 'lower', self.eps * np.ones((self.sys.Nu,1))))

    # objective
    G = np.concatenate((np.eye(2), np.zeros((2,2))), axis=1)
    G = np.kron(np.eye(self.N), G)
    Q = np.eye(self.N * 2) # position x and y for each agent
    controller.addObjectiveFun(lmpc.objectives.TerminalQuadForm(Q, np.zeros((self.N*2,1)), G))

    controller._setupSolver(np.zeros((self.sys.Nx, 1)))

    return controller

  def act(self, x0, logits):
    self.controller.objectives[0].xTd.value = logits
    u, _, _ = self.controller.solve(x0, "SCS")
    action = np.concatenate(([0], u.squeeze()[0:self.Na]), dtype=np.float32)
      
    return action


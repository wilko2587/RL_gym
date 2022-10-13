from datetime import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class FFnet(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_layers=(128,), activation=nn.ReLU,
                 l2=0., lr=1e-3,
                 dropout=None, random_state=1,
                 device="CPU",
                 layer_std=None,
                 output_function='softmax'):
        '''

        @param: input_size: int. number of inputs to model.
        @param: output_size. int. number of output nodes
        @param: hidden_layers. tuple. contains integers dictating the number of nodes in the
            network's hidden layers
        @param: activation: activation function
        @param: l2: float. l2 regularisation constant
        @param: lr: float. learning rate
        @param: dropout: tuple. contains integers dictating dropout rate for each hidden layer.
            must be same dimension as hidden_layers.
        @param: device: torch device to put the model on
        @param: output_layer_init_std: stdev of initial weights in output layer.
        @param: output_function: function to apply after final layer of network.
        '''

        super(FFnet, self).__init__()

        torch.manual_seed(random_state)

        if dropout is not None:
            assert isinstance(dropout, list)
            assert len(dropout) == len(hidden_layers)

        self.activation = activation
        self.hidden_layers = hidden_layers
        self.l2 = l2
        self.lr = lr
        self.dropout = dropout
        self.criterion = nn.MSELoss()

        if layer_std is None:
            layer_std = 1.

        # Build input layer
        sequentials = []
        layer = nn.Linear(input_size, self.hidden_layers[0], dtype=torch.double).to(device)
        torch.nn.init.uniform_(layer.weight, -layer_std/np.sqrt(layer.weight.size(1)), layer_std/np.sqrt(layer.weight.size(1)))
        sequentials.append(layer)

        # Build hidden layers
        for i in range(len(self.hidden_layers) - 1):
            sequentials.append(self.activation().to(device))
            if self.dropout is not None: sequentials.append(nn.Dropout(self.dropout[i]))
            layer = nn.Linear(self.hidden_layers[i],
                                         self.hidden_layers[i + 1], dtype=torch.double).to(device)
            torch.nn.init.uniform_(layer.weight, -layer_std / np.sqrt(layer.weight.size(1)), layer_std/np.sqrt(layer.weight.size(1)))
            #nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            sequentials.append(layer)

        # Build output layer
        sequentials.append(self.activation().to(device))
        out_layer = nn.Linear(self.hidden_layers[-1], output_size, dtype=torch.double)
        torch.nn.init.uniform_(out_layer.weight, -layer_std / np.sqrt(out_layer.weight.size(1)), layer_std/np.sqrt(out_layer.weight.size(1)))
        sequentials.append(out_layer)
        if str(output_function).lower() == 'tanh':
            sequentials.append(nn.Tanh().to(device))
        self.output_function=output_function
        self.stack = nn.Sequential(*sequentials).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=lr,
                                    weight_decay=l2)

    def forward(self, X):
        logits = self.stack(X)
        if str(self.output_function).lower()=='softmax' and len(X.shape) == 2:
            return torch.nn.functional.softmax(logits, dim=1)
        elif str(self.output_function).lower()=='softmax' and len(X.shape) == 1:
            return torch.nn.functional.softmax(logits, dim=0)
        else:
            return logits


class REINFORCEAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 400

        self.gamma = 0.95   # discount rate

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # create policy model
        self.model = FFnet(input_size=self.state_size,
                           output_size= self.action_size,
                           hidden_layers=(128,),
                           lr=1e-3,
                           device=self.device)

        self.short_memory = [] # to contain action probabilities, reward, action for an episode

    def act(self, state):
        # implement the epsilon-greedy policy
        if len(state.shape) == 1: # catch for dimensionality issues
            state = np.array(state).reshape([1, len(state)])
        action_ps = self.model(torch.DoubleTensor(state))
        action = np.random.choice(range(self.action_size), p=action_ps.detach().cpu().numpy().flatten())
        return action, action_ps

    def finish_episode(self):
        # learn when the episode is finished

        actions = [m[1] for m in self.short_memory]
        action_ps = [m[2] for m in self.short_memory]
        action_ps = torch.cat(action_ps)
        actions = torch.tensor(np.array(actions))
        action_prob = torch.sum(action_ps * torch.nn.functional.one_hot(actions), dim=1)

        rewards = [m[3] for m in self.short_memory]
        discounted_rewards = torch.DoubleTensor(np.array([rewards[i]*self.gamma**i for i in range(len(rewards))]))

        Q = (discounted_rewards + torch.sum(discounted_rewards, dim=0) - torch.cumsum(discounted_rewards, dim=0)) / torch.tensor(np.array([self.gamma**i for i in range(len(rewards))]))

        G = -(Q * torch.log(action_prob)).sum()

        self.model.optimizer.zero_grad()
        G.backward()
        self.model.optimizer.step()

        self.short_memory = [] # empty memory
        self.model.zero_grad()

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def training(self):

        scores = []
        for e in range(self.EPISODES):
            self.short_memory = [] # wipe memory after each episode
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0

            actions = []
            while not done:
                self.env.render()

                action, action_probs = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)

                if done:
                    reward = -100.

                self.short_memory.append((state, action, action_probs, reward))

                state = next_state
                i += 1
                if done:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("episode: {}/{}, score: {}, time: {}".format(e+1, self.EPISODES, i, timestampStr))
                    scores.append(i)
                    self.finish_episode()
        return scores

    # test function if you want to test the learned model
    def test(self):
        self.load("./save/cartpole-dqn-training.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state, verbose=0))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e+1, self.EPISODES, i))
                    break

if __name__ == "__main__":
    all_scores = []

    agent = REINFORCEAgent()
    scores = agent.training()
    all_scores.append(scores)

    plt.figure()
    plt.plot(range(len(scores)), scores, label='example learning curve')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('score')

    plt.show()

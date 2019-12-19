
import numpy as np
import torch as tr
from torch.autograd import Variable

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
class Net(tr.nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        #super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = tr.nn.Conv2d(4, 32, kernel_size=3, padding=1)  # 32 ge 3x3 4 channel
        self.conv2 = tr.nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 ge 3x3
        self.conv3 = tr.nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128 ge 3x3
        
        # action policy layers
        self.act_conv1 = tr.nn.Conv2d(128, 4, kernel_size=1)# 4 ge 1x1 jiang mei
        self.act_fc1 = tr.nn.Linear(4*board_width*board_height,#quan lian jie ceng
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = tr.nn.Conv2d(128, 2, kernel_size=1)# 2ge 1x1 jiang wei
        self.val_fc1 = tr.nn.Linear(2*board_width*board_height, 64)# quan lian jie ceng 64 ge shen jing yuan
        self.val_fc2 = tr.nn.Linear(64, 1)#quan lian jie ceng

    def forward(self, x):
        # common layers
        #x input
        y = tr.nn.functional.relu(self.conv1(x))
        y = tr.nn.functional.relu(self.conv2(y))
        y = tr.nn.functional.relu(self.conv3(y))
        
        # action policy layers
        y_act = tr.nn.functional.relu(self.act_conv1(y))#policy
        y_act = y_act.view(-1, 4*self.board_width*self.board_height)#??
        y_act = tr.nn.functional.log_softmax(self.act_fc1(y_act))
        
        # state value layers
        y_val = tr.nn.functional.relu(self.val_conv1(y))#value
        y_val = y_val.view(-1, 2*self.board_width*self.board_height)#??
        y_val = tr.nn.functional.relu(self.val_fc1(y_val))
        y_val = tr.nn.functional.tanh(self.val_fc2(y_val))
        return y_act, y_val
    
class PolicyValueNet():
    def __init__(self, board_width, board_height,
                 model_file=None):
        #self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        #if self.use_gpu:
       #     self.policy_value_net = Net(board_width, board_height).cuda()
       # else:
        self.policy_value_net = Net(board_width, board_height)#!!!!!!!!!!!!!!
        self.optimizer = tr.optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = tr.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """

        state_batch = Variable(tr.FloatTensor(state_batch))
        log_act_probs, value = self.policy_value_net(state_batch)#
        act_probs = np.exp(log_act_probs.data.numpy())
        return act_probs, value.data.numpy()

    def policy_value_fn(self, board):#!!!!
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
#       if self.use_gpu:
#          log_act_probs, value = self.policy_value_net(
#                  Variable(tr.from_numpy(current_state)).cuda().float())
  #          act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
#        else:
        log_act_probs, value = self.policy_value_net(
                Variable(tr.from_numpy(current_state)).float()) #current state
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
            
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""

        state_batch = Variable(tr.FloatTensor(state_batch))
        mcts_probs = Variable(tr.FloatTensor(mcts_probs))
        winner_batch = Variable(tr.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = tr.nn.functional.mse_loss(value.view(-1), winner_batch)
        policy_loss = -tr.mean(tr.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        
        # calc policy entropy, for monitoring only
        entropy = -tr.mean(
                tr.sum(tr.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param() 
        tr.save(net_params, model_file)

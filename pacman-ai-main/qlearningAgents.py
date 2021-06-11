# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalue = util.Counter()                          # Khởi tạo Qvalue

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalue[state, action]
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:                          # Giới hạn chiều dài tối đa được thép thử của Agent
            return 0.0                                      # quá chiều dài cho phép, về lại vị trí đầu
        maxqvalue = -999999                                 # Phạt agent vì không tìm ra đường đi
        for action in legalActions:
            if self.getQValue(state, action) > maxqvalue:   # thử từng Actions, gán Q value cho actions có kết quả tốt nhất.
                maxqvalue = self.getQValue(state, action)
        return maxqvalue
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = [None]                                 # Tái khới động lại bestaction
        legalActions = self.getLegalActions(state)          # Truy nhập toàn bộ các khả năng
        maxqvalue = -999999                                 # Đưa maxqvalue về giá trị nhỏ nhất
        for action in legalActions:                         # Thử từng action có thể
            if self.getQValue(state, action) > maxqvalue:   # Nếu giá Q tại state > max
                maxqvalue = self.getQValue(state, action)   # Maxq = giá trị hiện tại
                bestAction = [action]                       # Best = giá trị đó
            elif self.getQValue(state, action) == maxqvalue:# nếu không
                bestAction.append(action)                   # thêm action đó vào list

        return random.choice(bestAction)                    # Tiếp tục làm hoạt động ngẫu nhiên.
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)        # Gọi các giá trị state trong legalAc
        action = None                                     # Khởi động action
        "*** YOUR CODE HERE ***"

        if util.flipCoin(self.epsilon):                   # Take ramdom action.
            return random.choice(legalActions)            # Trả về giá trị legal actions
        else:
            return self.computeActionFromQValues(state)   # Hết random trở về tính giá trị Action từ Q

        # util.raiseNotDefined()

        # return action

    def update(self, state, action, nextState, reward):   # Tính toán T tự sample và cập nhật q
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        sample = reward + self.discount * self.computeValueFromQValues(nextState)                      # [R(s,a,s')+alpha*v(s')] 
        key = state, action
        self.qvalue[key] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * sample    
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor                                                            # Khởi tại biến cho Exploration function
        features = f.getFeatures(state, action)
        qvalue = 0                                                                        # Trả Qvalue về 0
        for feature in features.keys():                                                   # Thử từng feature
            qvalue += self.weights[feature] * features[feature]
        return qvalue
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        actionsFromNextState = self.getLegalActions(nextState)                            # gán s'
        maxqnext = -999999
        for act in actionsFromNextState:                                                  # thử tất cả các s', chọn s' max
            if self.getQValue(nextState, act) > maxqnext:
                maxqnext = self.getQValue(nextState, act)                                 # Tính Q(s')max
        if maxqnext == -999999:                                                           # Nếu s' ko đi dx thì gán  Q(s')max = 0
            maxqnext = 0
        diff = (reward + (self.discount * maxqnext)) - self.getQValue(state, action)      # nếu đi được thì tính diff = [r+alpha*maxQ(s',a')] - Q(s,a)
        features = self.featExtractor.getFeatures(state, action)
        self.qvalue[(state, action)] += self.alpha * diff
        for feature in features.keys():
            self.weights[feature] += self.alpha * diff * features[feature]
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # Thoát code
            pass
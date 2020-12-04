import torch

import torch.nn.functional as F


class RBM():

    def __init__(self, num_v, num_h, k=1, learning_rate=1e-3, momentum_coefficient=0.5,
                 weight_decay=1e-4, cuda=False):
        self.num_v = num_v
        self.num_h = num_h
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.device = torch.device('cpu')
        self.cuda = cuda

        self.weights = torch.randn(num_v, num_h) * 0.01
        self.v_bias = torch.ones(num_v) * 0.5
        self.h_bias = torch.zeros(num_h)

        self.weights_momentum = torch.zeros(num_v, num_h)
        self.v_bias_momentum = torch.zeros(num_v)
        self.h_bias_momentum = torch.zeros(num_h)

        if self.cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                print("CUDA device was selected successfully.")
            else:
                self.device = torch.device('cpu')
                print("No CUDA device is available. Setting to default.")
            print(f"PyTorch target device: {self.device}")

            self.weights = self.weights.to(self.device)
            self.v_bias = self.v_bias.to(self.device)
            self.h_bias = self.h_bias.to(self.device)

            self.weights_momentum = self.weights_momentum.to(self.device)
            self.v_bias_momentum = self.v_bias_momentum.to(self.device)
            self.h_bias_momentum = self.h_bias_momentum.to(self.device)

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.h_bias
        hidden_probabilities = torch.sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.v_bias
        visible_probabilities = torch.sigmoid(visible_activations)
        return visible_probabilities

    def forward(self, input_data, train=True):
        # Positive phase
        pos_hidden_probabilities = self.sample_hidden(input_data)
        pos_hidden_activations = (pos_hidden_probabilities >= torch.rand(self.num_h).to(self.device)).float()
        positive_associations = torch.matmul(input_data.t(), pos_hidden_activations)

        # Negative phase
        hidden_activations = pos_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= torch.rand(self.num_h).to(self.device)).float()

        neg_visible_probabilities = visible_probabilities
        neg_hidden_probabilities = hidden_probabilities
        negative_associations = torch.matmul(neg_visible_probabilities.t(), neg_hidden_probabilities)

        if train:
            # Update parameters
            self.weights_momentum *= self.momentum_coefficient
            self.weights_momentum += (positive_associations - negative_associations)

            self.v_bias_momentum *= self.momentum_coefficient
            self.v_bias_momentum += torch.sum(input_data - neg_visible_probabilities, dim=0)

            self.h_bias_momentum *= self.momentum_coefficient
            self.h_bias_momentum += torch.sum(pos_hidden_probabilities - neg_hidden_probabilities, dim=0)

            batch_size = input_data.size(0)

            self.weights += self.weights_momentum * self.learning_rate / batch_size
            self.v_bias += self.v_bias_momentum * self.learning_rate / batch_size
            self.h_bias += self.h_bias_momentum * self.learning_rate / batch_size

            self.weights -= self.weights * self.weight_decay  # L2 weight decay

        return input_data, neg_visible_probabilities

    def free_energy(self, v):
        """
        Free energy function.

        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}

        @param v: (Tensor) The visible variable.
        @return: (FloatTensor) The free energy value.
        """
        v_a_term = torch.matmul(v, self.v_bias.t())
        h_b_term = torch.sum(F.softplus(F.linear(v, self.weights, self.h_bias)), dim=1)
        return torch.mean(-h_b_term - v_a_term)

    def contrastive_divergence(self, input_data):
        # Do k sampling steps and updates
        input_data, neg_visible_probabilities = self.forward(input_data)

        # Compute reconstruction error
        error = F.mse_loss(input_data, neg_visible_probabilities)

        return error

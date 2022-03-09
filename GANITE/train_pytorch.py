import numpy as np
from .ganite_pytorch import *
import torch

from utils.utils_pytorch import batch_generator


def ganite(train_x, train_t, train_yf, parameters):
    """GANITE module.
    Args:
      - train_x: features in training data
      - train_t: treatments in training data
      - train_y: observed outcomes in training data
      - test_x: features in testing data
      - parameters: GANITE network parameters
        - h_dim: hidden dimensions
        - batch_size: the number of samples in each batch
        - iterations: the number of iterations for training
        - alpha: hyper-parameter to adjust the loss importance

    Returns:
      - test_y_hat: estimated potential outcome for testing set
    """

    # Parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    alpha = parameters['alpha']

    no, dim = train_x.shape

    ## Structure
    # 1. Generator
    cf_gen = Generator(dim, h_dim)
    # Y_tilde_logit = generator(X, T, Y)
    # Y_tilde = tf.nn.sigmoid(Y_tilde_logit)

    # 2. Discriminator
    cf_dis = Discriminator(dim, h_dim)
    # D_logit = discriminator(X, T, Y, Y_tilde)

    # 3. Inference network
    inf = Inference(dim, h_dim)
    # Y_hat_logit = inference(X)
    # Y_hat = tf.nn.sigmoid(Y_hat_logit)

    ## Loss functions
    # 1. Discriminator loss
    # D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=T, logits=D_logit))
    #
    # # 2. Generator loss
    # G_loss_GAN = -D_loss
    # G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=Y, logits=(T * tf.reshape(Y_tilde_logit[:, 1], [-1, 1]) + \
    #                       (1. - T) * tf.reshape(Y_tilde_logit[:, 0], [-1, 1]))))
    #
    # G_loss = G_loss_Factual + alpha * G_loss_GAN
    #
    # # 3. Inference loss
    # I_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=(T) * Y + (1 - T) * tf.reshape(Y_tilde[:, 1], [-1, 1]), logits=tf.reshape(Y_hat_logit[:, 1], [-1, 1])))
    # I_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=(1 - T) * Y + (T) * tf.reshape(Y_tilde[:, 0], [-1, 1]), logits=tf.reshape(Y_hat_logit[:, 0], [-1, 1])))

    # I_loss = I_loss1 + I_loss2

    ## Solver
    G_solver = torch.optim.Adam(cf_gen.parameters(), lr=3e-3)
    D_solver = torch.optim.Adam(cf_dis.parameters(), lr=3e-3)
    I_solver = torch.optim.Adam(inf.parameters(), lr=3e-3)

    ## GANITE training

    print('Start training Generator and Discriminator')
    # 1. Train Generator and Discriminator
    for it in range(iterations):

        for _ in range(2):
            # Discriminator training
            for param in cf_gen.parameters():
                param.requires_grad = False
            D_solver.zero_grad()
            X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_yf, batch_size)
            Y_tilde_logit = cf_gen(torch.FloatTensor(X_mb), torch.FloatTensor(T_mb), torch.FloatTensor(Y_mb))
            Y_tilde = torch.sigmoid(Y_tilde_logit)
            D_logit = cf_dis(torch.FloatTensor(X_mb), torch.FloatTensor(T_mb), torch.FloatTensor(Y_mb), Y_tilde)
            # print(D_logit, T_mb)
            D_loss_curr = torch.mean(
                torch.nn.functional.binary_cross_entropy_with_logits(D_logit, torch.FloatTensor(T_mb)))
            D_loss_curr.backward(retain_graph=True)
            D_solver.step()

            for param in cf_gen.parameters():
                param.requires_grad = True

        # Generator training
        G_solver.zero_grad()

        for param in cf_dis.parameters():
            param.requires_grad = False
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_yf, batch_size)
        X_mb = torch.FloatTensor(X_mb)
        T_mb = torch.FloatTensor(T_mb)
        Y_mb = torch.FloatTensor(Y_mb)

        Y_tilde_logit = cf_gen(X_mb, T_mb, Y_mb)
        Y_tilde = torch.sigmoid(Y_tilde_logit)
        D_logit = cf_dis(X_mb, T_mb, Y_mb, Y_tilde)
        G_loss_GAN = -torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(D_logit, T_mb))
        G_loss_Factual = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            T_mb * Y_tilde_logit[:, 1].reshape(-1, 1) + (
                    1. - T_mb) * Y_tilde_logit[:, 0].reshape(-1, 1), Y_mb))
        G_loss_curr = G_loss_Factual + alpha * G_loss_GAN
        G_loss_curr.backward()
        G_solver.step()

        for param in cf_dis.parameters():
            param.requires_grad = True

        # Check point
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) + ', D loss: ' +
                  str(np.round(D_loss_curr.item(), 4)) + ', G loss: ' + str(np.round(G_loss_curr.item(), 4)))

    # for param in cf_gen.parameters():
    #     param.requires_grad = False
    for param in cf_dis.parameters():
        param.requires_grad = False
    print('Start training Inference network')
    # 2. Train Inference network
    for it in range(iterations):
        I_solver.zero_grad()
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_yf, batch_size)
        X_mb = torch.FloatTensor(X_mb)
        T_mb = torch.FloatTensor(T_mb)
        Y_mb = torch.FloatTensor(Y_mb)
        # print(X_mb.shape)
        Y_tilde_logit = cf_gen(X_mb, T_mb, Y_mb)
        Y_tilde = torch.sigmoid(Y_tilde_logit)
        Y_hat_logit = inf(X_mb)
        # print(Y_hat_logit.shape)

        I_loss1 = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            Y_hat_logit[:, 1].reshape(-1, 1), T_mb * Y_mb + (1 - T_mb) * Y_tilde[:, 1].reshape(-1, 1))
        )
        I_loss2 = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            Y_hat_logit[:, 0].reshape(-1, 1), (1 - T_mb) * Y_mb + T_mb * Y_tilde[:, 0].reshape(-1, 1))
        )

        I_loss_curr = I_loss1 + I_loss2
        I_loss_curr.backward()
        I_solver.step()

        # Check point
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) +
                  ', I loss: ' + str(np.round(I_loss_curr.item(), 4)))

    inf.eval()
    return inf


def ganite_predict(model, x):
    # Generate the potential outcomes

    Y_hat_logit = model(torch.FloatTensor(x))
    y = torch.sigmoid(Y_hat_logit).detach().numpy()
    y0 = y[:, 0]
    y1 = y[:, 1]
    # print(y0, y1)
    return y0, y1
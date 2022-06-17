# Generalization-Error-and-Lipschitz-Constant-Bounds-for-Polynomial-Nets

Official PyTorch implementation of the Projected Stochastic Gradient Descent in Polynomial Nets as described on the ICLR'22 paper "Controlling the Complexity and Lipschitz Constant improves Polynomial Nets".

Specifically, we include the code for projected during training and attack during testing.

## Run experiment

Use our single bound projection method for PN-Conv in Fashion-MNIST: `python main.py`.

Use other datasets: Modify `parser.add_argument('--dataset', type=str, default='FashionMNIST')` in `main.py`.

Use other model: Choose another model in `model.py`.

Use other attacks: Modify `parser.add_argument('--epsilon_test', type=float, default=0.01)`, `parser.add_argument('--eps_iter_test', type=float, default=0.01)`, `parser.add_argument('--nb_iter_test', type=int, default=1)` in `main.py`.


Use other hyperparameters: Modify corresponding hyperparameters in `main.py`.


## More experiments (L2 regularization, Jacobian regularization, adversarial training) to be updated

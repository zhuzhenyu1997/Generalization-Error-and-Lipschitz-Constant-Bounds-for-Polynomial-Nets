# Generalization-Error-and-Lipschitz-Constant-Bounds-for-Polynomial-Nets

Official PyTorch implementation of the Projected Stochastic Gradient Descent in Polynomial Nets as described on the ICLR'22 paper "[Controlling the Complexity and Lipschitz Constant improves Polynomial Nets](https://openreview.net/pdf?id=dQ7Cy_ndl1s)" and its extension (under review).

Specifically, we include the code for projected during training and attack during testing.

## Browsing the folder and files

The folder and files structure is the following:

`utils/dataset.py`: Describes some parsers for dataset; normally you should not modify it for provided dataset(s).

`utils/evaluation.py`: A class for calculating the average.

`utils/param_parser.py`: Some classes for some extra parameter parsers.

`utils/projection.py`: A library contains a series of projection operators.

`utils/seq_parser.py`: A library controls the learning rate for step descent.

`main.py`: It is used for experiment setting and to make the code starting.

`model.py`: It contains fully connected polynomial network model and the convolutional polynomial network model we mentioned in the paper.

`solver.py`: It contains the training and testing part of the code.

## Run experiment

To run the experiment for single bound projection method for PN-Conv in Fashion-MNIST, you can execute the following command:

&emsp; `python main.py`

For other experiment settings:

&emsp; Use other datasets: Modify `parser.add_argument('--dataset', type=str, default='FashionMNIST')` in `main.py`.

&emsp; Use other model: Choose another model in `model.py`.

&emsp; Use other attacks: Modify `parser.add_argument('--epsilon_test', type=float, default=0.01)`, `parser.add_argument('--eps_iter_test', type=float, default=0.01)`, `parser.add_argument('--nb_iter_test', type=int, default=1)` in `main.py`.

&emsp; Use other hyperparameters: Modify corresponding hyperparameters in `main.py`.

More experiments (L2 regularization, Jacobian regularization, adversarial training) to be updated.

## Requirements

The code requires the following librarys.

Tested on a Linux machine with:

&emsp; * numpy=1.20.1

&emsp; * torch=1.8.1

&emsp; * torchvision=0.4.2

&emsp; * torchattacks=2.14.1

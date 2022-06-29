import torch
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div,
                torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div,
                    torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(
            avoid_zero_div,
            torch.sum(grad ** 2, red_ind, keepdim=True)
        )
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(
            torch.float) * opt_pert_norm + (
                           square > avoid_zero_div
                   ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def fast_gradient_method(model, x, eps=0.3, norm=np.inf,
                         clip_min=None, clip_max=None, y=None, targeted=False,
                         sanity_checks=False):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter)
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect. Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
        asserts.append(assert_le)

    x = x.clone().detach().to(torch.float).requires_grad_(True)
    print(model(x).shape)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model(x)[0], 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model(x)[0], y)

    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def basic_iterative_method(
        model,
        x,
        y=None,
        targeted=False,
        eps=0.15,
        eps_iter=0.01,
        n_iter=50,
        clip_max=None,
        clip_min=None):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model(x)[0], 1)

    eta = torch.zeros(x.shape).to(x.device)

    for i in range(n_iter):
        out = model(x + eta)[0]
        loss = F.cross_entropy(out, y)
        if targeted:
            loss = -loss
        loss.backward()
        eta += eps_iter * torch.sign(x.grad.data)
        eta.clamp_(-eps, eps)
        x.grad.data.zero_()

    x_adv = x + eta

    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
        x_adv.clamp_(clip_min, clip_max)

    return x_adv.detach()


INF = float("inf")


def carlini_wagner_l2(
    model_fn,
    x,
    n_classes,
    y=None,
    targeted=False,
    lr=5e-3,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1e-2,
    binary_search_steps=5,
    max_iterations=1000,
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    """

    def compare(pred, label, is_logits=False):
        """
        A helper function to compare prediction against a label.
        Returns true if the attack is considered successful.
        :param pred: can be either a 1D tensor of logits or a predicted
                class (int).
        :param label: int. A label to compare against.
        :param is_logits: (optional) bool. If True, treat pred as an
                array of logits. Default is False.
        """

        # Convert logits to predicted class if necessary
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)

        return pred == label if targeted else pred != label

    def arctanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        pred = model_fn(x)[0]
        y = torch.argmax(pred, 1)

    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const

    o_bestl2 = [INF] * len(x)
    o_bestscore = [-1.0] * len(x)
    x = torch.clamp(x, clip_min, clip_max)
    ox = x.clone().detach()  # save the original x
    o_bestattack = x.clone().detach()

    # Map images into the tanh-space
    x = (x - clip_min) / (clip_max - clip_min)
    x = torch.clamp(x, 0, 1)
    x = x * 2 - 1
    x = arctanh(x * 0.999999)

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True).to(x.device)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float)

    # Define loss functions and optimizer
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device),
    )
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
    optimizer = torch.optim.Adam([modifier], lr=lr)

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        # Initialize some values needed for the inner loop
        bestl2 = [INF] * len(x)
        bestscore = [-1.0] * len(x)

        # Inner loop performing attack iterations
        for i in tqdm(range(max_iterations), leave=False):
            # One attack step
            new_x = (torch.tanh(modifier + x) + 1) / 2
            new_x = new_x * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)[0]

            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox)
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            # Update best results
            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                y_n = y[n]
                succeeded = compare(logits_n, y_n, is_logits=True)
                if l2_n < o_bestl2[n] and succeeded:
                    pred_n = torch.argmax(logits_n)
                    o_bestl2[n] = l2_n
                    o_bestscore[n] = pred_n
                    o_bestattack[n] = new_x_n
                    # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                    bestl2[n] = l2_n
                    bestscore[n] = pred_n
                elif l2_n < bestl2[n] and succeeded:
                    bestl2[n] = l2_n
                    bestscore[n] = torch.argmax(logits_n)

        # Binary search step
        for n in range(len(x)):
            y_n = y[n]

            if compare(bestscore[n], y_n) and bestscore[n] != -1:
                # Success, divide const by two
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                # Failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[n] = max(lower_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    const[n] *= 10

    return o_bestattack.detach()


def projected_gradient_descent(
        model_fn,
        x,
        eps,
        eps_iter,
        nb_iter,
        norm,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x)[0], 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def spsa(
        model_fn,
        x,
        eps,
        nb_iter,
        norm=np.inf,
        clip_min=-np.inf,
        clip_max=np.inf,
        y=None,
        targeted=False,
        early_stop_loss_threshold=None,
        learning_rate=0.01,
        delta=0.01,
        spsa_samples=128,
        spsa_iters=1,
        is_debug=False,
        sanity_checks=True,
):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which is useful when
    the model is non-differentiable, or more generally, the gradients do not point in useful
    directions.
    :param model_fn: A callable that takes an input tensor and returns the model logits.
    :param x: Input tensor.
    :param eps: The size of the maximum perturbation, measured in the L-infinity norm.
    :param nb_iter: The number of optimization steps.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: If specified, the minimum input value.
    :param clip_max: If specified, the maximum input value.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the
              default, will try to make the label incorrect. Targeted will instead try to
              move in the direction of being more like y.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end as
              soon as the loss is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size
              (the number of evaluated inputs for each update) is `spsa_samples *
              spsa_iters`
    :param spsa_iters:  Number of model evaluations before performing an update, where each
              evaluation is on `spsa_samples` different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    if y is not None and len(x) != len(y):
        raise ValueError(
            "number of inputs {} is different from number of labels {}".format(
                len(x), len(y)
            )
        )
    if y is None:
        y = torch.argmax(model_fn(x)[0], dim=1)

    # The rest of the function doesn't support batches of size greater than 1,
    # so if the batch is bigger we split it up.
    if len(x) != 1:
        adv_x = []
        for x_single, y_single in tqdm(zip(x, y), leave=False, total=len(x)):
            adv_x_single = spsa(
                model_fn=model_fn,
                x=x_single.unsqueeze(0),
                eps=eps,
                nb_iter=nb_iter,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y_single.unsqueeze(0),
                targeted=targeted,
                early_stop_loss_threshold=early_stop_loss_threshold,
                learning_rate=learning_rate,
                delta=delta,
                spsa_samples=spsa_samples,
                spsa_iters=spsa_iters,
                is_debug=is_debug,
                sanity_checks=sanity_checks,
            )
            adv_x.append(adv_x_single)
        return torch.cat(adv_x)

    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x

    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    asserts.append(torch.all(x >= clip_min))
    asserts.append(torch.all(x <= clip_max))

    if is_debug:
        print("Starting SPSA attack with eps = {}".format(eps))

    perturbation = (torch.rand_like(x) * 2 - 1) * eps
    _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for i in range(nb_iter):

        def loss_fn(pert):
            """
            Margin logit loss, with correct sign for targeted vs untargeted loss.
            """
            logits = model_fn(x + pert)[0]
            loss_multiplier = 1 if targeted else -1
            return loss_multiplier * _margin_logit_loss(logits,
                                                        y.expand(len(pert)))

        spsa_grad = _compute_spsa_gradient(
            loss_fn, x, delta=delta, samples=spsa_samples, iters=spsa_iters
        )
        perturbation.grad = spsa_grad
        optimizer.step()

        _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)

        loss = loss_fn(perturbation).item()
        if is_debug:
            print("Iteration {}: loss = {}".format(i, loss))
        if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:
            break

    adv_x = torch.clamp((x + perturbation).detach(), clip_min, clip_max)

    if norm == np.inf:
        asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
    else:
        asserts.append(
            torch.all(
                torch.abs(
                    torch.renorm(adv_x - x, p=norm, dim=0, maxnorm=eps) - (
                            adv_x - x)
                )
                < 1e-6
            )
        )
    asserts.append(torch.all(adv_x >= clip_min))
    asserts.append(torch.all(adv_x <= clip_max))

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def _project_perturbation(
        perturbation, norm, epsilon, input_image, clip_min=-np.inf,
        clip_max=np.inf
):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = clip_eta(perturbation, norm, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation, clip_min,
                            clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())

    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        with torch.no_grad():
            loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = (
                torch.mean(loss_vals * torch.sign(delta_x), dim=0,
                           keepdim=True) / delta
        )
        grad_list.append(avg_grad)

    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)


def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.
    The loss is high when `label` is unlikely (targeted by default).
    """

    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float("-inf")),
        logits,
    )
    max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

    return max_incorrect_logits - correct_logits

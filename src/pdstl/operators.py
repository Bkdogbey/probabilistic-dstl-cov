import torch
import numpy as np


# =============================================================================
# BASE STL FORMULA
# =============================================================================
class STL_Formula(torch.nn.Module):
    """
    Base class for Probabilistic STL formulas.
    """

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        """
        Compute robustness trace for belief trajectory.

        Args:
           belief_trajectory: BeliefTrajectory object
           scale: smoothing parameter (scale > 0 for smooth, <= 0 for exact)
           keepdim: keep dimensions

        Returns:
           [B,T,2] probability bounds on robustness trace where
           [..., 0] is lower bound and [..., 1] is upper bound
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, **kwargs):
        """Forward pass delegates to robustness_trace"""
        return self.robustness_trace(belief_trajectory, **kwargs)

    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")

    def __and__(self, other):
        """Overload & operator for And"""
        return And(self, other)

    def __or__(self, other):
        """Overload | operator for Or"""
        return Or(self, other)

    def __invert__(self):
        """Overload ~ operator for Negation"""
        return Negation(self)


# =============================================================================
# SMOOTH MIN / MAX
# =============================================================================
class Minish(torch.nn.Module):
    """Compute minimum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Compute maximum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.max(dim=dim, keepdim=keepdim)[0]


# =============================================================================
# BOOLEAN OPERATORS
# =============================================================================
class Negation(STL_Formula):
    """
    Negation: ¬ϕ
    Swaps and complements bounds: [lower, upper] -> [1 - upper, 1 - lower]
    """

    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        lower = 1.0 - trace[..., 1]
        upper = 1.0 - trace[..., 0]
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"¬({self.subformula})"


class And(STL_Formula):
    """
    Conjunction: ϕ₁ ∧ ϕ₂
    Frechet bounds: lower = max(l1 + l2 - 1, 0),  upper = min(u1, u2)
    """

    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        trace2 = self.subformula2(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        l1, u1 = trace1[..., 0:1], trace1[..., 1:2]
        l2, u2 = trace2[..., 0:1], trace2[..., 1:2]
        lower = torch.maximum(l1 + l2 - 1.0, torch.zeros_like(l1))
        upper = torch.minimum(u1, u2)
        return torch.cat([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


class Or(STL_Formula):
    """
    Disjunction: ϕ₁ ∨ ϕ₂
    Frechet bounds: lower = max(l1, l2),  upper = min(u1 + u2, 1)
    """

    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        trace2 = self.subformula2(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        l1, u1 = trace1[..., 0:1], trace1[..., 1:2]
        l2, u2 = trace2[..., 0:1], trace2[..., 1:2]
        lower = torch.maximum(l1, l2)
        upper = torch.minimum(u1 + u2, torch.ones_like(u1))
        return torch.cat([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∨ ({self.subformula2})"


# =============================================================================
# TEMPORAL OPERATORS BASE
# =============================================================================
class Temporal_Operator(STL_Formula):
    """
    Base class for temporal operators (Always, Eventually).
    Uses a backward RNN pass for O(T) forward-looking semantics.
    """

    def __init__(self, subformula, interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval

        if not self.interval:
            self.rnn_dim = 1
        else:
            a, b = self._interval
            self.rnn_dim = int(max(1, a)) if np.isinf(b) else int(b + 1)

        self.operation = None  # set by subclass (Minish or Maxish)

        self.M = (
            torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1))
            .requires_grad_(False)
            .float()
        )
        self.b = torch.zeros(self.rnn_dim).unsqueeze(-1).requires_grad_(False).float()
        self.b[-1] = 1.0

    def _initialize_rnn_cell(self, x):
        if x.is_cuda:
            self.M = self.M.cuda()
            self.b = self.b.cuda()
        h0 = x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()
        count = 0.0
        if (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0 = x[:, :1, :]
            return ((d0, h0), count)
        return (h0, count)

    def _apply_shift(self, h0, x):
        batch, rnn_dim, bounds = h0.shape
        h0_reshaped = h0.permute(0, 2, 1)
        h0_flat = h0_reshaped.reshape(-1, rnn_dim)
        shifted_flat = torch.matmul(h0_flat, self.M.t())
        shifted = shifted_flat.reshape(batch, bounds, rnn_dim).permute(0, 2, 1)
        b_broadcast = self.b.view(1, -1, 1)
        x_broadcast = x.squeeze(1).unsqueeze(1)
        return shifted + b_broadcast * x_broadcast

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        raise NotImplementedError

    def _run_cell(self, x, scale):
        outputs = []
        hc = self._initialize_rnn_cell(x)
        for xs_i in torch.split(x, 1, dim=1):
            o, hc = self._rnn_cell(xs_i, hc, scale)
            outputs.append(o)
        return torch.cat(outputs, dim=1)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        trace_reversed = torch.flip(trace, dims=[1])
        output_reversed = self._run_cell(trace_reversed, scale=scale)
        return torch.flip(output_reversed, dims=[1])


# =============================================================================
# ALWAYS
# =============================================================================
class Always(Temporal_Operator):
    """
    □_I ϕ: Always operator — computes smooth min over time interval.
    """

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        h0, c = hc
        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)
            output = self.operation(input_, scale, dim=1, keepdim=True)
            state = (output, None)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, scale, dim=1, keepdim=True)
            state = ((output, self._apply_shift(h0, x)), None)
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)
        return output, state

    def __str__(self):
        if self.interval is None:
            return f"□({self.subformula})"
        return f"□_{self._interval}({self.subformula})"


# =============================================================================
# EVENTUALLY
# =============================================================================
class Eventually(Temporal_Operator):
    """
    ◇_I ϕ: Eventually operator — computes smooth max over time interval.
    """

    def __init__(self, subformula, interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        h0, c = hc
        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)
            output = self.operation(input_, scale, dim=1, keepdim=True)
            state = (output, None)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, scale, dim=1, keepdim=True)
            state = ((output, self._apply_shift(h0, x)), None)
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)
        return output, state

    def __str__(self):
        if self.interval is None:
            return f"◇({self.subformula})"
        return f"◇_{self._interval}({self.subformula})"

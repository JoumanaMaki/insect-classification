import torch
from ..manifolds.lorentz.math import expmap0, logmap0, expmap, project_u, project, parallel_transport0back
from ..tensor import ManifoldParameter, ManifoldTensor


@torch.no_grad()
def move_parameters_scale(man_params, state):
    for point in man_params:
        if point[1] != point[0].manifold.k:
            if point[0].manifold.k == 0:
                print("Warning: Manifold")

            param_state = state[point[0]]
            exp_avg = param_state["exp_avg"]

            # stabalize
            point[0].copy_(project(point[0], k=point[1]))

            exp_avg.copy_(project_u(point[0], exp_avg, k=point[1]))
            exp_avg = parallel_transport0back(point[0], exp_avg, k=point[1])

            point[0].copy_(point[0] * point[0].manifold.k.sqrt() / point[1].sqrt())

            exp_avg.copy_(point[0].manifold.transp0(point[0], exp_avg))
        stabilize_param(state, point[0])

@torch.no_grad()
def move_parameters(man_params, state):
    for point in man_params:
        if point[1] != point[0].manifold.k:
            if point[0].manifold.k == 0:
                print("Warning: Manifold")

            param_state = state[point[0]]
            exp_avg = param_state["exp_avg"]

            # stabalize
            point[0].copy_(project(point[0], k=point[1]))

            exp_avg.copy_(project_u(point[0], exp_avg, k=point[1]))
            exp_avg = parallel_transport0back(point[0], exp_avg, k=point[1])

            tangent = logmap0(point[0], k=point[1], dim=-1)
            # tangent = point[0].manifold.rescale_to_max(tangent).squeeze()

            # tangent = rescale_to_max(tangent, point[0].manifold).squeeze()

            point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

            exp_avg.copy_(point[0].manifold.transp0(point[0], exp_avg))
        stabilize_param(state, point[0])


@torch.no_grad()
def stabilize_group(state, group):
    for p in group["params"]:
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            continue
        state = state[p]
        if not state:  # due to None grads
            continue
        manifold = p.manifold
        exp_avg = state["exp_avg"]
        p.copy_(manifold.projx(p))
        exp_avg.copy_(manifold.proju(p, exp_avg))

@torch.no_grad()
def stabilize_param(state, p):
    state = state[p]
    if not state:  # due to None grads
        return
    manifold = p.manifold
    exp_avg = state["exp_avg"]
    p.copy_(manifold.projx(p))
    exp_avg.copy_(manifold.proju(p, exp_avg))


@torch.no_grad()
def move_parameters_rsgd_scale(man_params, param_groups, state):
    for point in man_params:
        if point[1] != point[0].manifold.k:
            if point[0].manifold.k == 0:
                print("Warning: Manifold")

            # stabalize
            momentum = param_groups[1]["momentum"]

            point[0].copy_(project(point[0], k=point[1]))
            if momentum > 0:
                param_state = state[point[0]]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(project_u(point[0], buf, k=point[1]))
                    buf = parallel_transport0back(point[0], buf, k=point[1])

            point[0].copy_(point[0] * point[0].manifold.k.sqrt() / point[1].sqrt())

            if momentum > 0:
                param_state = state[point[0]]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    # buf = point[0].manifold.rescale_to_max(buf)
                    buf.copy_(point[0].manifold.transp0(point[0], buf))

            stabilize_param_rsgd(point, momentum, state)


@torch.no_grad()
def move_parameters_rsgd(man_params, param_groups, state):
    for point in man_params:
        if point[1] != point[0].manifold.k:
            if point[0].manifold.k == 0:
                print("Warning: Manifold")

            # stabalize
            momentum = param_groups[1]["momentum"]

            point[0].copy_(project(point[0], k=point[1]))
            if momentum > 0:
                param_state = state[point[0]]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(project_u(point[0], buf, k=point[1]))
                    buf = parallel_transport0back(point[0], buf, k=point[1])

            tangent = logmap0(point[0], k=point[1], dim=-1)
            # tangent = point[0].manifold.rescale_to_max(tangent)

            #tangent = rescale_to_max(tangent, point[0].manifold).squeeze()

            point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

            if momentum > 0:
                param_state = state[point[0]]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    # buf = point[0].manifold.rescale_to_max(buf)
                    buf.copy_(point[0].manifold.transp0(point[0], buf))

            stabilize_param_rsgd(point, momentum, state)

@torch.no_grad()
def stabilize_group_rsgd(group, state):
    for p in group["params"]:
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            continue
        manifold = p.manifold
        momentum = group["momentum"]
        p.copy_(manifold.projx(p))
        if momentum > 0:
            param_state = state[p]
            if not param_state:  # due to None grads
                continue
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                buf.copy_(manifold.proju(p, buf))

@torch.no_grad()
def stabilize_param_rsgd(p, momentum, state):
    if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
        return
    manifold = p.manifold
    p.copy_(manifold.projx(p))
    if momentum > 0:
        param_state = state[p]
        if not param_state:  # due to None grads
            return
        if "momentum_buffer" in param_state:
            buf = param_state["momentum_buffer"]
            buf.copy_(manifold.proju(p, buf))

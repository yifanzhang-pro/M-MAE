import torch
import torch.distributed as dist


def uniformity_loss_square(features):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T 
    loss = sim.pow(2).mean()
    return loss

def centering_matrix(m):
    J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
    return J_m


def uniformity_loss_TCR(features, uniformity_mu=1., centering=False):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    if centering:
        J_m = centering_matrix(features.shape[0]).detach().to(features.device)
        sim = features.T @ J_m @ features
    else:
        sim = features.T @ features 

    # loss = $- \log \det (\mathbf{I} + mu / m * Z Z^{\top})$
    loss = -torch.logdet(torch.eye(sim.shape[0]).to(features.device) + uniformity_mu / sim.shape[0] * sim)
    return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

import paddle
from .builder import SAMPLINGS

class SamplingBase(object):

    def __init__(self, net, num_steps, step_size, refsig, noise_ratio=1, 
                 noise_decay=False, langevin_clip=-1, activate_eval=False, **kwargs): 

        super().__init__() 
        self.energy_function = net
        self.num_steps, self.step_size = num_steps, step_size 
        self.refsig, self.noise_decay, self.noise_ratio = refsig, noise_decay, noise_ratio
        self.activate_eval, self.langevin_clip = activate_eval, langevin_clip
        self.default_shape = None 

@SAMPLINGS.register()
class SamplingLangevin(SamplingBase):
    """ This class implements the Langevin Dynamic. 
        Gradient clipping and noise decay is supported.
        TODO: logger, warm start etc.
    """

    def sampling(self, init=None, noise=1, logger=None, shape=None, **kargs): 

        if self.default_shape is None and shape is None: 
            if init is None: 
                raise RuntimeError("Cannot perform sampling without init in first round.")
            self.default_shape = init.shape
        else: 
            if init is None: 
                if shape is not None:
                    self.default_shape = shape
                init = paddle.rand(shape=self.default_shape) * 2 - 1 if self.refsig == 0 else paddle.randn(shape=self.default_shape) * self.refsig
        if logger is not None:
            #TODO: logger to be written.
            raise NotImplementedError
            energy_his, du_his = [], []
        # For some network include batchnorm, eval() need to be set. 
        if self.activate_eval:
            self.energy_function.eval()
            
        u = init.detach()
        for i in range(self.num_steps):
            u.stop_gradient = False
            energy = self.energy_function(u, **kargs)
            grad = paddle.grad([energy.sum()], [u], retain_graph=True)[0]

            # Gradient descent part in langevin
            du = 0.5 * self.step_size * self.step_size * grad

            # Prior p_0(X), refsig is 0 if uniform prior is used.
            if self.refsig != 0: 
                du -= 0.5 * self.step_size * self.step_size * u / self.refsig / self.refsig

            # Noise part in Langevin
            # Noise_decay: in one Langevin Dynamic, decrease the noise linearly. 
            if noise > 0: 
                noise_decay = max(0, (self.num_steps - i - 5) / (self.num_steps - 5)) if self.noise_decay else 1
                du += noise * self.noise_ratio * noise_decay * self.step_size * paddle.randn(shape=u.shape)

            # Clip the result.
            if self.langevin_clip > 0:
                u = paddle.clip(u + du, -self.langevin_clip, self.langevin_clip).detach()
            else: 
                u = (u + du).detach()

            if logger is not None:
                #TODO: logger to be written.
                energy_his.append(float(energy.mean().cpu().data))
                du_his.append(float(paddle.abs(grad).mean().cpu().data))  
                
        if self.activate_eval:      
            self.energy_function.train()
        
        if logger is not None:
            #TODO: logger to be written.
            logger[0].experiment.add_figure("training/energy", util_torch.plot_fig(energy_his), logger[1])
            logger[0].experiment.add_figure("training/du", util_torch.plot_fig(du_his), logger[1])

        return u.detach()

    def __call__(self, *args, **kargs): 
        
        return self.sampling(*args, **kargs)
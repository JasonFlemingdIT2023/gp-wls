import torch

class MaternKernel:
    
    def __init__(self, length_scale: float=1.0,output_variance: float=1.0, nu: float=2.5):
        self.length_scale = length_scale
        self.output_variance = output_variance
        self.nu = nu
        
    def _compute_distance(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        '''
        X1 = (n,d), X2 = (m,d), n = number of points, d= dimension
        Returns: (n,m) distance matrix
        
        none for boradcasting
        Example:
        
        X1[:, none, :].shape = (2, 1, 5)
        X2[none, :, :].shape = (1, 3, 5)
        
        Problem in first 2 dimensions
        
        '''
        diff = X1[:, None, :] - X2[None, :, :] #(n,m,d)
        return torch.sqrt(torch.sum(diff**2, dim=-1)) #(n,m)
    
    def _matern_12(self, r: torch.Tensor) -> torch.Tensor:
         return self.output_variance * torch.exp(-r / self.length_scale)


    def _matern_32(self, r: torch.Tensor) -> torch.Tensor:
        sqrt3r = (3 ** 0.5) * r / self.length_scale
        return self.output_variance * (1 + sqrt3r) * torch.exp(-sqrt3r)
    

    def _matern_52(self, r: torch.Tensor) -> torch.Tensor:
        sqrt5r = (5 ** 0.5) * r / self.length_scale
        return self.output_variance * (1 + sqrt5r + (5 * r**2) / (3* self.length_scale**2)) * torch.exp(-sqrt5r)
    
    def __call__(self, X1: torch.Tensor,X2: torch.Tensor) -> torch.Tensor:
       r = self._compute_distance(X1, X2)
    
       if self.nu == 0.5:
           return self._matern_12(r)
       elif self.nu == 1.5:
           return self._matern_32(r)
       elif self.nu == 2.5:
           return self._matern_52(r)
       else:
           raise ValueError(f"nu must be 0.5, 1.5, or 2.5 - got {self.nu}")
    


X = torch.tensor([[0.0], [1.0], [2.0]])

k = MaternKernel(length_scale=1.0, output_variance=1.0, nu=2.5)
K = k(X, X)
print(K)

        
        
    
    
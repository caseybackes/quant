from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Normal
from torchdiffeq import odeint
from tqdm import tqdm

import yaml
import matplotlib.pyplot as plt



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        
        with torch.no_grad():
            self.net[-1].bias.data.fill_(0.1)
    
    def forward(self, t, x):
        # Extract current radius and add it to dynamics
        r = torch.sqrt(x[..., 0]**2 + x[..., 1]**2).unsqueeze(-1)
        dynamics = self.net(x)
        # Add explicit rotational component
        dynamics[:, 0] += -x[:, 1] * 0.5  # Encourage CCW rotation
        dynamics[:, 1] += x[:, 0] * 0.5
        return dynamics


class Loss:
    def __init__(self, loss_type='mse'):
        self.loss_type = loss_type
    
    def __call__(self, x, pred, mu, logvar):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, x)
        elif self.loss_type == 'mae':
            return F.l1_loss(pred, x)
        elif self.loss_type == 'cosine':
            return F.cosine_similarity(pred, x)
        elif self.loss_type == 'l2':
            return F.pairwise_distance(pred, x)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred, x)
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy(pred, x)
        elif self.loss_type == 'kld':
            return 0.5 * torch.sum(logvar.exp() + mu**2 - 1 - logvar)
        elif self.loss_type == 'jsd':
            return 0.5 * torch.sum(logvar.exp() + mu**2 - 1 - logvar)
        else:
            raise ValueError("Invalid loss type")


class LatentSODE(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, latent_dim=None, output_dim=None, cfg='cfg.yaml'):
        super().__init__()
        if not all([input_dim, hidden_dim, latent_dim, output_dim]):
            if not cfg:
                raise ValueError("Either specify all dimensions or provide a path to configuration file in `cfg` argument")
            if not Path(cfg).exists():
                raise FileNotFoundError(f"Configuration file not found at {cfg}")
            breakpoint()
            with open(cfg, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
                """
                config.yaml: 
                input_dim: 2
                hidden_dim: 64
                latent_dim: 6
                output_dim: 2
                loss: mse # mse | mae | cosine | l2 | l1 | bce | kld | jsd
                integration_method: rk4 # euler | rk4
                """
            required_cfg_keys = ['input_dim', 'hidden_dim', 'latent_dim', 'output_dim', 'loss', 'integration_method']
            if not all([key in cfg for key in required_cfg_keys]):
                raise ValueError(f"Configuration file must contain keys: {required_cfg_keys}")
            self.cfg = cfg
        
        
        
        
        self._loss_function = Loss(cfg['loss'])
        self.integration_method = cfg['integration_method']
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
    
    def forward(self, x, time_steps):
        # No encoding/decoding - direct ODE on input space
        z = odeint(self.ode_func, x, time_steps, method=self.integration_method)
        breakpoint()
        return z, None, None
    
    
    def loss_function(self, x, pred, **kwargs):
        return self._loss_function(x, pred, **kwargs)

    # def loss_function(self, x, pred, mu, logvar):
        
        
        # # Simple MSE loss on the trajectories
        # loss = F.mse_loss(pred[0], x)  # Initial points
        
        # # Add spiral consistency
        # dr = torch.sqrt(pred[1:, :, 0]**2 + pred[1:, :, 1]**2) - \
        #      torch.sqrt(pred[:-1, :, 0]**2 + pred[:-1, :, 1]**2)
        # radius_growth = torch.mean((dr - 0.1)**2)  # Encourage constant radius growth
        
        # return loss + 0.1 * radius_growth


def train_example():
    # Model parameters
    cfg = 'cfg.yaml'
    # input_dim = 2
    # hidden_dim = 64
    # latent_dim = 6
    # output_dim = 2
    
    # model = LatentSODE(input_dim, hidden_dim, latent_dim, output_dim)
    model = LatentSODE(cfg=cfg)
    breakpoint()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    batch_size = 64
    epochs = 10
    time_steps = torch.linspace(0, 4*torch.pi, 60)

    def generate_spiral(n_points, noise=0.00):
        t = torch.linspace(0, 2*torch.pi, n_points)
        r = t/(2*torch.pi)  # Linear radius growth
        x = r * torch.cos(t) + noise * torch.randn(n_points)
        y = r * torch.sin(t) + noise * torch.randn(n_points)
        
        # Normalize the data
        data = torch.stack([x, y], dim=1)
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        return (data - mean) / (std + 1e-5)

    x = generate_spiral(200, noise = 0.035)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(DEVICE)
    x = x.to(DEVICE)
    time_steps = time_steps.to(DEVICE)

    model.train()
    best_loss = float('inf')
    
    with tqdm(range(epochs), desc="Training") as epoch_pbar:
        for epoch in epoch_pbar:
            total_loss = 0
            # Shuffle data at the start of each epoch
            indices = torch.randperm(x.size(0))
            n_batches = 0
            
            with tqdm(range(0, x.size(0), batch_size), desc=f"Epoch {epoch+1}", leave=False) as batch_pbar:
                # Process data in batches
                for i in batch_pbar:
                    batch_idx = indices[i:i + batch_size]
                    batch_x = x[batch_idx]
                    
                    # Zero gradients for this batch
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred, mu, logvar = model(batch_x, time_steps)
                    
                    # Calculate loss
                    loss = model.loss_function(batch_x, pred, mu, logvar)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                
            # Calculate average loss for the epoch
            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model.pt')
                # make a plot at this point
                plt.figure(figsize=(15, 6))
                with torch.no_grad():
                    n_viz = 5
                    test_x = x[:n_viz]
                    time_steps_viz = torch.linspace(0, 4*torch.pi, 200).to(DEVICE)
                    pred, _, _ = model(test_x, time_steps_viz)
                    
                    # Plot original spiral
                    plt.subplot(121)
                    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), alpha=0.5, s=1, label='Training Data')
                    plt.title('Original Spiral')
                    plt.legend('Training Data')
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.axis('equal')
                    
                    # Plot predictions
                    plt.subplot(122)
                    pred = pred.cpu()
                    colors = ['r', 'g', 'b', 'c', 'm']
                    for i in range(n_viz):
                        plt.plot(pred[:, i, 0], pred[:, i, 1], f'{colors[i]}-', 
                                label=f'Trajectory {i+1}', alpha=0.7)
                        plt.scatter(test_x[i, 0].cpu(), test_x[i, 1].cpu(), 
                                c=colors[i], marker='o', s=100, label=f'Start {i+1}')
                    
                    plt.title('Predicted Trajectories')
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.legend()
                    # set plot extent range to be equal to the spiral plot

                    plt.axis('equal')
                    
                    plt.tight_layout()
                    plt.savefig('best_model.png')
                    plt.close()
            epoch_pbar.set_postfix(loss=avg_loss)

    # Visualization code
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    model.eval()


    with torch.no_grad():
        n_viz = 5
        test_x = x[:n_viz]
        time_steps_viz = torch.linspace(0, 4*torch.pi, 200).to(DEVICE)
        pred, _, _ = model(test_x, time_steps_viz)
        
        plt.figure(figsize=(15, 6))
        
        # Plot original spiral
        plt.subplot(121)
        plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), alpha=0.5, s=1, label='Training Data')
        plt.title('Original Spiral')
        plt.legend()
        plt.axis('equal')
        
        # Plot predictions
        plt.subplot(122)
        pred = pred.cpu()
        colors = ['r', 'g', 'b', 'c', 'm']
        for i in range(n_viz):
            plt.plot(pred[:, i, 0], pred[:, i, 1], f'{colors[i]}-', 
                    label=f'Trajectory {i+1}', alpha=0.7)
            plt.scatter(test_x[i, 0].cpu(), test_x[i, 1].cpu(), 
                       c=colors[i], marker='o', s=100, label=f'Start {i+1}')
        
        plt.title('Predicted Trajectories')
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    train_example()
"""
A Hamiltonian network learns to model the behavior of 
ships across the global ocean as if they were particles 
moving through an abstract energy landscape where common 
shipping lanes are like gravitational valleys (low-energy, 
natural paths), while restricted areas or unusual routes 
are like hills (high-energy, unnatural paths), and this 
network can be trained on our irregularly-sampled 
historical ais data to learn what "normal" 
energy-efficient trajectories look like, even with gaps 
of hours or days between observations. Once trained, the 
network can then predict the most physically plausible 
paths between any two observed positions by following 
these learned energy-preserving dynamics (similar to 
how a ball naturally rolls down a valley rather than 
up a hill), which lets you identify suspicious behavior 
when ships take "energy-inefficient" paths that deviate 
from these natural trajectories - like a ship metaphorically 
rolling uphill or teleporting between positions in ways that 
violate the learned physics of normal maritime traffic 
patterns, even when working with very sparse observations 
across the world's oceans.
"""


"""
##############################################
Top Level Python Requirements:
    
    pip install torch pandas geopandas tqdm

##############################################
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import geopandas as gpd 
from pathlib import Path

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm


# Classical Hamiltonian with kinetic energy
class ClassicalHamiltonian(nn.Module):
    def energy(self, state):
        return (self.kinetic_energy(velocity) + 
                self.potential_energy(state)) # Total energy is conserved

# Statistical mechanics version relating to entropy
class StatMechHamiltonian(nn.Module):
    def energy(self, state):
        return (self.energy_net(state) - 
                self.entropy_net(state))  # Free energy is still conserved!

# Potential fields version (prob my favorite)
class PotentialFieldHamiltonian(nn.Module):
    def energy(self, state):
        return (self.spatial_potential(pos) + 
                self.directional_potential(heading))  # Total field energy is conserved

class MaritimeHamNet(nn.Module):
    def __init__(self):
        super().__init__()
        # State space: [lon, lat, speed, heading]
        # Output: scalar energy value
        self.potential_energy = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Kinetic energy component based on speed
        # This captures the "cost" of speed changes
        self.kinetic_energy = lambda v: 0.5 * v**2
        
    def energy(self, state):
        position = state[:2]  # lon, lat
        velocity = state[2:]  # speed, heading
        
        # Total energy is kinetic + potential
        return (self.kinetic_energy(velocity) + 
                self.potential_energy(state))

class MaritimeTrajectoryAnalyzer:
    def __init__(self):
        # Trained Hamiltonian network for ship type
        self.hamiltonian_net = MaritimeHamNet()
        
    def analyze_trajectory(self, observations):
        """
        observations: list of (timestamp, lon, lat, speed, heading) tuples
        returns: anomaly scores for each observation
        """
        # Get the most recent n observations
        recent_obs = observations[-5:]  # Example: last 5 points
        
        # For each new observation
        anomaly_scores = []
        for i in range(1, len(recent_obs)):
            # Previous state
            t_prev, lon_prev, lat_prev, spd_prev, hdg_prev = recent_obs[i-1]
            # Current state
            t_curr, lon_curr, lat_curr, spd_curr, hdg_curr = recent_obs[i]
            
            # Time delta between observations
            dt = t_curr - t_prev
            
            # Use Hamiltonian to predict where ship SHOULD be
            predicted_state = self.hamiltonian_net.predict_trajectory(
                start_state=[lon_prev, lat_prev, spd_prev, hdg_prev],
                duration=dt
            )
            
            # Calculate deviation metrics
            position_error = haversine_distance(
                (lon_curr, lat_curr), 
                (predicted_state[0], predicted_state[1])
            )
            
            speed_error = abs(spd_curr - predicted_state[2])
            heading_error = angle_difference(hdg_curr, predicted_state[3])
            
            # Combine into anomaly score (could use different weights)
            anomaly_score = (
                position_error * POSITION_WEIGHT +
                speed_error * SPEED_WEIGHT +
                heading_error * HEADING_WEIGHT
            )
            
            anomaly_scores.append(anomaly_score)
            
        return anomaly_scores

    def is_behavior_anomalous(self, recent_trajectory, threshold=None):
        """
        Determines if latest behavior is anomalous
        """
        if threshold is None:
            # Could dynamically set based on ship type & historical patterns
            threshold = self.compute_dynamic_threshold(recent_trajectory)
            
        latest_score = self.analyze_trajectory(recent_trajectory)[-1]
        return latest_score > threshold

class HamiltonianNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Neural network for potential energy
        self.potential_net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [lon, lat, speed, heading]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(self._device)
        
        # Learned parameters for kinetic energy
        self.mass = nn.Parameter(torch.ones(1))
        self.speed_weight = nn.Parameter(torch.ones(1))
        
    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
    
    def kinetic_energy(self, speed: torch.Tensor) -> torch.Tensor:
        speed = speed.to(self._device)
        return 0.5 * self.mass * self.speed_weight * speed**2
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate total energy for given state
        state: [lon, lat, speed, heading]
        """
        state = state.to(self._device)
        potential = self.potential_net(state)
        kinetic = self.kinetic_energy(state[:, 2])  # speed component
        return potential + kinetic

    def save(self):
        ham_net_root = Path('hamiltonian_models')
        if not Path(f'{ham_net_root}/runs').exists():
            Path(f'{ham_net_root}/runs').mkdir(parents=True)
        runs_count = len(list(Path(f'{ham_net_root}/runs').iterdir())) + 1
        run_path = ham_net_root / 'runs' / f'run{runs_count}'
        run_path.mkdir(parents=True, exist_ok=True)
        model_path = run_path / 'model.pt'
        torch.save(self.state_dict(), model_path)
        return str(model_path)

class MaritimePredictor:
    def __init__(self, model: HamiltonianNet):
        self.model = model
    
    def normalize_state(self, state: np.ndarray, grid_size: Tuple[float, float]) -> torch.Tensor:
        """Normalize state values to [0, 1] range"""
        norm_state = state.copy()
        norm_state[0] /= grid_size[0]  # lon
        norm_state[1] /= grid_size[1]  # lat
        norm_state[2] /= 20.0  # speed (assuming max speed of 20)
        norm_state[3] /= 360.0  # heading
        return torch.FloatTensor(norm_state)
    
    def denormalize_state(self, 
                         norm_state: torch.Tensor, 
                         grid_size: Tuple[float, float]) -> np.ndarray:
        """Convert normalized state back to original scale"""
        state = norm_state.detach().numpy()
        state[0] *= grid_size[0]
        state[1] *= grid_size[1]
        state[2] *= 20.0
        state[3] *= 360.0
        return state
    
    def calculate_anomaly_score(self, 
                              observed_state: np.ndarray,
                              predicted_state: np.ndarray,
                              grid_size: Tuple[float, float],
                              w_pos: float=0.5,
                              w_speed: float=0.3,
                              w_heading: float=0.2) -> float:
        """Calculate standardized anomaly score based on prediction error"""
        # Normalize states
        norm_observed = self.normalize_state(observed_state, grid_size)
        norm_predicted = self.normalize_state(predicted_state, grid_size)
        
        # Calculate error components
        position_error = torch.norm(norm_observed[:2] - norm_predicted[:2])
        speed_error = abs(norm_observed[2] - norm_predicted[2])
        heading_error = abs(norm_observed[3] - norm_predicted[3])
        
        # Combine errors with weights
        total_error = (w_pos * position_error + 
                      w_speed * speed_error + 
                      w_heading * heading_error)
        
        return float(total_error)

    def integrate_state(self,
                    initial_state: np.ndarray,
                    time_delta: float,
                    grid_size: Tuple[float, float],
                    step_size: float = 3600.0) -> np.ndarray:  # default 1 hour steps
        """
        Integrate Hamilton's equations to predict future state
        Parameters:
        initial_state: [lon, lat, speed, heading]
        time_delta: time difference in seconds
        grid_size: for normalization
        step_size: integration step size in seconds (default: 1 hour)
        """
        device = self.model.device
        current_state = self.normalize_state(initial_state, grid_size).to(device)
        n_steps = max(1, int(time_delta / step_size))  # at least 1 step
        
        with tqdm(total=n_steps, desc="Integrating State", leave=False) as pbar:
            for _ in range(n_steps):
                
                current_state.requires_grad_(True)
                energy = self.model(current_state.unsqueeze(0).to(device))
                grad = torch.autograd.grad(energy, current_state, create_graph=True)[0]
                
                # Update state using Hamilton's equations and keep on GPU
                current_state = (current_state + step_size * grad).detach().to(device)
                pbar.update(1)
        
        # Only move to CPU at the very end
        return self.denormalize_state(current_state.cpu(), grid_size)

    def predict_future_trajectory(self, 
                                recent_observations: List[Dict],
                                prediction_times: List[datetime],
                                grid_size: Tuple[float, float],
                                ship_type: str = None) -> List[Dict]:
        """
        Predict future states at specified times
        
        Args:
            recent_observations: list of dicts with keys ['timestamp', 'lon', 'lat', 'speed', 'heading']
            prediction_times: list of future timestamps to predict for
            grid_size: for normalization which avoids nuke-level gradient explosion
            ship_type: vessel type (currently unused but could be used for type-specific predictions)

        Returns:
            list of dicts with keys ['timestamp', 'state', 'uncertainty'] where 'state'
                is [lon, lat, speed, heading] and 'uncertainty' is a dict with 
                keys 'position', 'speed', 'heading'
        """
        predictions = []
        
        
        # Get initial state from most recent observation
        last_obs = recent_observations[-1]
        initial_state = np.array([
            last_obs['lon'], 
            last_obs['lat'], 
            last_obs['speed'], 
            last_obs['heading']
        ])
        
        last_time = pd.to_datetime(last_obs['timestamp'])


        for target_time in tqdm(prediction_times, desc="Predicting Future Trajectory"):
            target_time = pd.to_datetime(target_time)
            time_delta = (target_time - last_time).total_seconds()


            # Predict next state
            predicted_state = self.integrate_state(
                initial_state=initial_state,
                time_delta=time_delta,
                grid_size=grid_size
            )
            # Estimate uncertainty based on prediction time
            uncertainty = {
                'position': 0.001 * time_delta,  # Grows linearly with time
                'speed': 0.0005 * time_delta,
                'heading': 0.001 * time_delta
            }
            
            predictions.append({
                'timestamp': target_time,
                'state': predicted_state,
                'uncertainty': uncertainty
            })
            
            # Update initial state for next prediction
            initial_state = predicted_state
            last_time = target_time
            
        return predictions

    def estimate_uncertainty(self, state: np.ndarray, time_delta: float) -> Dict:
        """Estimate prediction uncertainty based on time horizon"""
        return {
            'position': 0.001 * time_delta,  # Growing circle/ellipse
            'speed': 0.0005 * time_delta,    # Min/max bounds
            'heading': 0.001 * time_delta    # Angular uncertainty
        }

class ShipDataFactory:
    def __init__(self, 
                 start_time: datetime = datetime(2024, 1, 1),
                 grid_size: Tuple[float, float] = (100.0, 100.0),
                 duration_days: int = 30):
        self.start_time = start_time
        self.grid_size = grid_size
        self.duration_days = duration_days
        

    def generate_sine_wave_ship(self, 
                              mmsi: int = 1, 
                              frequency: float = 0.05,
                              base_speed: float = 10.0) -> pd.DataFrame:
        """Generate ship following sine wave pattern bottom to top"""
        times = []
        positions = []
        speeds = []
        headings = []
        
        current_time = self.start_time
        while current_time < self.start_time + timedelta(days=self.duration_days):
            # Random time gap between 1-100 minutes
            time_gap = np.random.randint(1, 100)
            current_time += timedelta(minutes=time_gap)
            
            # Calculate position along sine wave
            t = (current_time - self.start_time).total_seconds() / 86400  # Convert to days
            x = t * (self.grid_size[0] / self.duration_days)
            y = (np.sin(2 * np.pi * frequency * t) + 1) * self.grid_size[1] / 2
            # add some noise to position
            x += np.random.normal(0, 0.5)


            # Calculate heading based on trajectory
            next_t = t + 0.01
            next_x = next_t * (self.grid_size[0] / self.duration_days)
            next_y = (np.sin(2 * np.pi * frequency * next_t) + 1) * self.grid_size[1] / 2
            heading = np.degrees(np.arctan2(next_y - y, next_x - x)) % 360
            # add some noise to heading
            heading += np.random.normal(0, 3)

            times.append(current_time)
            positions.append((x, y))
            speeds.append(base_speed + np.random.normal(0, 1))
            headings.append(heading)
        
        return pd.DataFrame({
            'timestamp': times,
            'mmsi': mmsi,
            'lon': [p[0] for p in positions],
            'lat': [p[1] for p in positions],
            'speed': speeds,
            'heading': headings
        })
    
    def generate_diagonal_ship(self, 
                             mmsi: int = 2,
                             base_speed: float = 12.0) -> pd.DataFrame:
        """Generate ship moving diagonally across grid"""
        times = []
        positions = []
        speeds = []
        headings = []
        
        current_time = self.start_time
        while current_time < self.start_time + timedelta(days=self.duration_days):
            time_gap = np.random.randint(1, 100)
            current_time += timedelta(minutes=time_gap)
            
            t = (current_time - self.start_time).total_seconds() / (86400 * self.duration_days)
            x = t * self.grid_size[0]
            y = t * self.grid_size[1]
            
            heading = 45 + np.random.normal(-3,3)  # Diagonal movement
            
            times.append(current_time)
            positions.append((x, y))
            speeds.append(base_speed + np.random.normal(0, 0.5))
            headings.append(heading)
            
        return pd.DataFrame({
            'timestamp': times,
            'mmsi': mmsi,
            'lon': [p[0] for p in positions],
            'lat': [p[1] for p in positions],
            'speed': speeds,
            'heading': headings
        })
    
    def generate_horizontal_ship(self,
                               mmsi: int = 3,
                               base_speed: float = 8.0,
                               y_position: float = 50.0) -> pd.DataFrame:
        """Generate ship moving horizontally"""
        times = []
        positions = []
        speeds = []
        headings = []
        
        current_time = self.start_time
        while current_time < self.start_time + timedelta(days=self.duration_days):
            time_gap = np.random.randint(1, 100)
            current_time += timedelta(minutes=time_gap)
            
            t = (current_time - self.start_time).total_seconds() / (86400 * self.duration_days)
            x = t * self.grid_size[0]
            y = y_position
            
            heading = 90  # Eastward movement
            
            times.append(current_time)
            positions.append((x, y))
            speeds.append(base_speed + np.random.normal(0, 0.5))
            headings.append(heading)
            
        return pd.DataFrame({
            'timestamp': times,
            'mmsi': mmsi,
            'lon': [p[0] for p in positions],
            'lat': [p[1] for p in positions],
            'speed': speeds,
            'heading': headings
        })
    
    def generate_all_ships(self) -> pd.DataFrame:
        """Generate data for all three ships"""
        ship1 = self.generate_sine_wave_ship()
        ship2 = self.generate_diagonal_ship()
        ship3 = self.generate_horizontal_ship()
        # plot ship1
        
        return pd.concat([ship1, ship2, ship3], ignore_index=True)

def train_model(data: pd.DataFrame, 
                model: HamiltonianNet,
                grid_size: Tuple[float, float],
                epochs: int = 100,
                batch_size: int = 256 ) -> List[float]:
    """Train the Hamiltonian network on ship trajectory data"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    # Prepare training data
    states = torch.FloatTensor(data[['lon', 'lat', 'speed', 'heading']].values)
    states[:, 0] /= grid_size[0]
    states[:, 1] /= grid_size[1]
    states[:, 2] /= 20.0
    states[:, 3] /= 360.0
    
    # put appropriate data on device
    states = states.to(DEVICE)
    dataloader = torch.utils.data.DataLoader(states, batch_size=batch_size, shuffle=True)
    
    
    # Training loop
    print("Training Hamiltonian network...")
    with tqdm(range(epochs), desc="Epochs") as pbar_epoch:
        for epoch in pbar_epoch:
            optimizer.zero_grad()
            with tqdm(dataloader, desc="Batches", leave=False) as pbar_batch:
                for state in pbar_batch:
                    energy = model(state)
                    energy_diff = torch.diff(energy.squeeze())
                    loss = torch.mean(energy_diff**2)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss))
                    pbar_batch.set_postfix(loss=loss.item())
            pbar_epoch.set_postfix(epoch_loss=loss.item())
    print("Training complete.")
    return losses

def plot_trajectory_predictions(data: pd.DataFrame, 
                              model: HamiltonianNet,
                              predictor: MaritimePredictor,
                              grid_size: Tuple[float, float],
                              n_test_ships: int = 5,
                              n_points_to_use: int = 10):
    """
    ## Not actually working just yet. kinda wonky still. 
    Plot actual vs predicted trajectories for test ships
    - Takes n_points_to_use points to predict from 
    - Compares against actual next positions
    """
    # Get next batch of most frequent MMSIs not in original plot
    all_mmsis = data['mmsi'].value_counts()
    test_mmsis = all_mmsis[20:20+n_test_ships].index
    
    plt.figure(figsize=(15, 10))
    
    for mmsi in test_mmsis:
        ship_data = data[data['mmsi'] == mmsi].sort_values('timestamp')
        
        # Take first n_points_to_use points
        history = ship_data.iloc[:n_points_to_use]
        # Get actual future points to compare against
        future_actual = ship_data.iloc[n_points_to_use:n_points_to_use+5]
        
        # Use last few points to predict next positions
        last_state = history.iloc[-1][['lon', 'lat', 'speed', 'heading']].values
        last_state = [x.item() for x in last_state]
        
        # Get timestamps we want to predict for
        pred_times = future_actual['timestamp'].tolist()
        
        # Get predictions
        predictions = predictor.predict_future_trajectory(
            recent_observations=history.to_dict('records'),
            prediction_times=pred_times,
            ship_type='unknown',# for now
            grid_size=grid_size
        )
        """ 
        recent_observations: List[Dict],
        prediction_times: List[datetime],
        ship_type: str,
        grid_size: Tuple[float, float]) -> List[Dict]:

        
        """
        
        # Plot actual trajectory
        plt.plot(history['lon'], history['lat'], 
                'b-', label=f'History Ship {mmsi}')
        plt.plot(future_actual['lon'], future_actual['lat'], 
                'g-', label=f'Actual Future Ship {mmsi}')
        
        # Plot predicted positions
        pred_lons = [p['state'][0] for p in predictions]
        pred_lats = [p['state'][1] for p in predictions]
        plt.plot(pred_lons, pred_lats, 
                'r--', label=f'Predicted Ship {mmsi}')
        
        # Plot uncertainty ellipses if available
        for pred in predictions:
            if 'uncertainty' in pred:
                unc = pred['uncertainty']['position']
                plt.gca().add_patch(plt.Circle(
                    (pred['state'][0], pred['state'][1]),
                    unc,
                    fill=False,
                    linestyle='--',
                    color='r',
                    alpha=0.3
                ))
    
    plt.title('Trajectory Predictions vs Actual')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()


def main():

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NOTE:Generate test data if you dont have any on hand. 
    # we need the cols 'timestamp', 'lon', 'lat', 'speed', 'heading'
    # I have one on hand so I can use that, but if you want the same data you can pull it from :
    # GCP Bucket location:
    #       gs://dev-bucket-001/filtered_ais_data.parquet
    # or via command line:
    #       "gsutil cp gs://dev-bucket-001/filtered_ais_data.parquet ."
    
    # data = factory.generate_all_ships()
    if  Path("filtered_ais_data.parquet").exists(): 
        data = gpd.read_parquet("filtered_ais_data.parquet")
        
        data['_timestamp'] = pd.to_datetime(data['timestamp'])
        duration_days = (data['_timestamp'].max() - data['_timestamp'].min()).days
        grid_size = int(data['lon'].max() - data['lon'].min()), int(data['lat'].max() - data['lat'].min())
        del data['_timestamp'] # temp   
        data.rename(columns={'SOG': 'speed', 'COG': 'heading', 'MMSI':'mmsi'}, inplace=True)
        factory = ShipDataFactory(
            start_time=datetime(2024, 1, 1),
            grid_size=grid_size,
            duration_days=duration_days
        )
    else:
        grid_size = (100.0, 100.0)
        duration_days = 30
        factory = ShipDataFactory(
            start_time=datetime(2024, 1, 1),
            grid_size=grid_size,
            duration_days=duration_days
        )
        data = factory.generate_all_ships() # a ship data factory that generates 3 ships with different trajectories        

    
    

    
    # Initialize and train model
    model = HamiltonianNet().to(DEVICE)
    losses = train_model(data, model, factory.grid_size, epochs=12, batch_size=256)
    model_path = model.save()
    print(f"Model saved to {model_path}")
    # Create predictor

    # load the prior first model from hamiltonian_models/runs/run1/model.pt
    # it was one of those early runs where shit actually still worked. 
    # model.load_state_dict(torch.load('hamiltonian_models/runs/run1/model.pt'))
    predictor = MaritimePredictor(model)
    
    # Plot results
    plt.figure(figsize=(30, 10))
    
    # Plot 1: Training Loss
    plt.subplot(131)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Sample Batch')
    plt.ylabel('Loss')
    
    # Plot 2: Ship Trajectories
    
    # instead of plotting all mmsis which could be a few thousand, lets pick random 20 mmsis
    # and lets get the mmsis sorted by the number of observations and use the most common 20
    # mmsi_selections = np.random.choice(data['mmsi'].unique(), 20, replace=False)
    mmsi_selections = data['mmsi'].value_counts().index[:5]
    plt.subplot(132)
    for mmsi in tqdm(mmsi_selections, desc="Plotting 20 Random Ships' Trajectories"):
        ship_data = data[data['mmsi'] == mmsi]
        plt.plot(ship_data['lon'], ship_data['lat'], 'o-', label=f'Ship {mmsi}')

    plt.title('Ship Trajectories')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    # Plot 3: Anomaly Scores. How far off are the predictions from the actuals. Error rate sorta.
    plt.subplot(133)
    anomaly_scores = []

    for mmsi in mmsi_selections:
        ship_data = data[data['mmsi'] == mmsi]
        scores = []
        for i in range(1, len(ship_data)):
            prev_state = ship_data.iloc[i-1][['lon', 'lat', 'speed', 'heading']].values
            curr_state = ship_data.iloc[i][['lon', 'lat', 'speed', 'heading']].values
            curr_state = [c.item() for c in curr_state]
            prev_state = [p.item() for p in prev_state]
            score = predictor.calculate_anomaly_score(curr_state, prev_state, factory.grid_size)
            scores.append(score)
        plt.plot(scores, label=f'Ship {mmsi}')
    plt.title('Anomaly Scores')
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    prediction_fig = plot_trajectory_predictions(
        data, model, predictor, factory.grid_size
    )
    prediction_fig.show()
    prediction_fig.savefig("prediction_fig.png")
    print("Prediction figure saved to prediction_fig.png")



if __name__ == "__main__":
    main()


































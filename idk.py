# Import required libraries
import math
import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyproj import Proj, transform

# ----------------------- Data Loading ------------------------

# Load location.csv
data = pd.read_csv('location.csv')

# Create synthetic time column (assuming 1 Hz sampling)
data['Time'] = np.arange(len(data))

print("Data columns:", data.columns.tolist())
print(f"Total samples: {len(data)}")

# ----------------------- GPS Processing ------------------------

# Convert lat/lon to UTM (using WGS84)
proj_utm = Proj(proj="utm", zone=33, ellps="WGS84")  # Change 'zone=33' if you’re not in Europe
easting, northing = proj_utm(data['lon'].values, data['lat'].values)

data['utm_easting'] = easting
data['utm_northing'] = northing

# Compute velocities
time_diff = np.diff(data['Time'])
easting_diff = np.diff(data['utm_easting'])
northing_diff = np.diff(data['utm_northing'])
velocity_east = easting_diff / time_diff
velocity_north = northing_diff / time_diff
velocity_mag = np.sqrt(velocity_east**2 + velocity_north**2)

# ----------------------- IMU Normalization ------------------------

for col in ['gyrX', 'gyrY', 'gyrZ', 'accX', 'accY', 'accZ']:
    data[col] -= np.mean(data[col])

# ----------------------- Yaw Calculation ------------------------

# Integrate gyroscope (z-axis) for yaw
dt = np.mean(np.diff(data['Time']))
gyro_yaw_rate = data['gyrZ'] - np.mean(data['gyrZ'])
integrated_yaw = cumtrapz(gyro_yaw_rate, initial=0) * np.rad2deg(dt)

# Low-pass filter yaw (magnetometer not available)
def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

fs = 1  # Sampling rate (Hz)
cutoff_freq_low = 0.1
filtered_yaw = butter_lowpass_filter(integrated_yaw, cutoff_freq_low, fs)

# ----------------------- Particle Filter ------------------------

def transition(particles, dt, acc_x, gyro_z):
    noise = np.random.normal(0, 0.5, particles.shape)
    particles[:, 0] += particles[:, 3] * dt * np.cos(particles[:, 2]) + noise[:, 0]
    particles[:, 1] += particles[:, 3] * dt * np.sin(particles[:, 2]) + noise[:, 1]
    particles[:, 2] += gyro_z * dt + noise[:, 2]  # Orientation
    particles[:, 3] += acc_x * dt + noise[:, 3]   # Velocity
    return particles

def update_weights(particles, gps_easting, gps_northing):
    distances = np.sqrt((particles[:, 0] - gps_easting)**2 + (particles[:, 1] - gps_northing)**2)
    weights = np.exp(-distances / 10.0)
    weights /= np.sum(weights)
    return weights

def resample(particles, weights):
    indices = np.arange(len(weights))
    resampled_indices = np.random.choice(indices, size=len(weights), p=weights)
    return particles[resampled_indices]

# Initialize particles
num_particles = 5000
particles = np.empty((num_particles, 4))
particles[:, 0] = data['utm_easting'].iloc[0] + np.random.normal(0, 1, num_particles)
particles[:, 1] = data['utm_northing'].iloc[0] + np.random.normal(0, 1, num_particles)
particles[:, 2] = np.random.normal(0, 0.1, num_particles)  # orientation
particles[:, 3] = np.random.normal(0, 0.1, num_particles)  # velocity

weights = np.ones(num_particles) / num_particles

predicted_states_pf = []

for i in range(len(data) - 1):
    dt = data['Time'].iloc[i + 1] - data['Time'].iloc[i]
    acc_x = data['accX'].iloc[i]
    gyro_z = data['gyrZ'].iloc[i]

    particles = transition(particles, dt, acc_x, gyro_z)
    weights = update_weights(particles, data['utm_easting'].iloc[i], data['utm_northing'].iloc[i])
    particles = resample(particles, weights)

    predicted_states_pf.append(np.mean(particles, axis=0))

predicted_states_pf = np.array(predicted_states_pf)

# ----------------------- Smoothing ------------------------

smoothed_states_pf = uniform_filter1d(predicted_states_pf[:, :2], size=5, axis=0)

# ----------------------- Metrics ------------------------

def compute_metrics(gt_easting, gt_northing, pred_easting, pred_northing):
    rmse_e = np.sqrt(mean_squared_error(gt_easting, pred_easting))
    rmse_n = np.sqrt(mean_squared_error(gt_northing, pred_northing))
    mae_e = mean_absolute_error(gt_easting, pred_easting)
    mae_n = mean_absolute_error(gt_northing, pred_northing)
    return (rmse_e, rmse_n), (mae_e, mae_n)

min_len = min(len(data['utm_easting']), len(smoothed_states_pf))
particle_filter_rmse, particle_filter_mae = compute_metrics(
    data['utm_easting'][:min_len],
    data['utm_northing'][:min_len],
    smoothed_states_pf[:min_len, 0],
    smoothed_states_pf[:min_len, 1]
)

print(f"Particle Filter RMSE (Easting): {particle_filter_rmse[0]:.2f} m")
print(f"Particle Filter RMSE (Northing): {particle_filter_rmse[1]:.2f} m")
print(f"Particle Filter MAE (Easting): {particle_filter_mae[0]:.2f} m")
print(f"Particle Filter MAE (Northing): {particle_filter_mae[1]:.2f} m")

# ----------------------- Visualization ------------------------

plt.figure(figsize=(12, 8))
plt.plot(data['utm_easting'], data['utm_northing'], label='GPS Ground Truth Path', linewidth=2)
plt.plot(smoothed_states_pf[:, 0], smoothed_states_pf[:, 1], label='Particle Filter Path', linewidth=2)
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Location Estimation using Particle Filter (from location.csv)')
plt.legend()
plt.grid(True)
plt.show()
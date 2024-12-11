# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the data
file_path = 'airline6.csv'
data_frame = pd.read_csv(file_path)
data_frame['FlightDate'] = pd.to_datetime(data_frame['Date'])

# Extract relevant columns
passenger_counts = data_frame['Number'].values
total_revenue = data_frame['Revenue'].values
flight_dates = data_frame['FlightDate']

# Define month labels
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Calculate monthly average passenger count
average_passenger_by_month = [
    data_frame[data_frame['FlightDate'].dt.month == month]['Number'].mean() for month in range(1, 13)
]

# Apply Fourier Transform to passenger data
fourier_transformed = np.fft.fft(passenger_counts)
frequency_values = np.fft.fftfreq(len(passenger_counts))

# Approximate the signal using a limited Fourier series
num_terms = 8
fourier_approximation = np.zeros_like(passenger_counts, dtype=float)

for term in range(num_terms):
    real_component = np.real(fourier_transformed[term])
    imaginary_component = np.imag(fourier_transformed[term])
    fourier_approximation += (
        2 / len(passenger_counts) * (
            real_component * np.cos(2 * np.pi * term * np.arange(len(passenger_counts)) / len(passenger_counts)) -
            imaginary_component * np.sin(2 * np.pi * term * np.arange(len(passenger_counts)) / len(passenger_counts))
        )
    )

# Compute the power spectrum and extract meaningful periods
power_values = np.abs(fourier_transformed) ** 2
valid_frequencies = frequency_values[frequency_values > 0]
valid_power = power_values[frequency_values > 0]
detected_periods = 1 / valid_frequencies

# Filter periods within a specific range (7 days to 365 days)
period_filter = (detected_periods >= 7) & (detected_periods <= 365)
filtered_periods = detected_periods[period_filter]
filtered_power_values = valid_power[period_filter]

# Calculate average ticket price and identify Main period
avg_ticket_price = total_revenue.sum() / passenger_counts.sum()
Main_power_index = np.argmax(filtered_power_values)
Main_cycle_period = filtered_periods[Main_power_index]

# Monthly averages and Fourier approximation
plt.figure(figsize=(10, 6))
plt.bar(range(1, 13), average_passenger_by_month, color='lightblue', label='Monthly Passenger Averages')
plt.plot(
    np.linspace(1, 12, len(fourier_approximation)), 
    fourier_approximation,
    color='red', linestyle='--', label='Fourier Approximation (8 terms)'
)
plt.title('Monthly Passenger Trends with Fourier Approximation \n Student ID: 23122661')
plt.xlabel('Month')
plt.ylabel('Average Number of Passengers')
plt.xticks(range(1, 13), months)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Power spectrum analysis with updated color scheme
plt.figure(figsize=(10, 6))
plt.plot(filtered_periods, filtered_power_values, label='Power Spectrum', marker='o', linestyle='-', linewidth=3, color='orange')
plt.axvline(x=Main_cycle_period, color='black', linestyle='--', label=f'Main Period: {Main_cycle_period:.2f} days')
plt.axhline(y=avg_ticket_price, color='green', linestyle='--', label=f'Avg Ticket Price: ${avg_ticket_price:.2f}')
plt.title('Power Spectrum and Periodicity Analysis')
plt.xlabel('Period (Days)')
plt.ylabel('Power')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Output calculated metrics
print(f"Average Ticket Price: ${avg_ticket_price:.2f}")
print(f"Main Period: {Main_cycle_period:.2f} days")

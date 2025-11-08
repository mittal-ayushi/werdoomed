import requests
import time
import os
import sys
import csv

# -----------------------------------------------------------------
# V V V V  YOU MUST EDIT THIS URL  V V V V
# -----------------------------------------------------------------
# Set this to the URL you see in your Phyphox app (Remote Access)
PHONE_URL = "http://172.16.88.240:8080"
# -----------------------------------------------------------------

# What data do you want to get?
# For "Accelerometer (with g)" → accX, accY, accZ
# For "Gyroscope" → gyrX, gyrY, gyrZ
# For "Location" → lat, lon, altitude
SENSORS_TO_GET = [
    "accX", "accY", "accZ",
    "gyrX", "gyrY", "gyrZ",
    "lat", "lon", "altitude"
]

# Build the URL for data request
query_string = "&".join(SENSORS_TO_GET)
FULL_REQUEST_URL = f"{PHONE_URL}/get?{query_string}"

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_sensor_data():
    """Fetches the latest sensor data from Phyphox."""
    try:
        response = requests.get(FULL_REQUEST_URL, timeout=2)
        response.raise_for_status()
        json_data = response.json()

        latest_data = {}
        for sensor in SENSORS_TO_GET:
            try:
                # Try the correct data path: ['buffer'][sensor]['buffer'][-1]
                value = json_data["buffer"][sensor]["buffer"][-1]
                latest_data[sensor] = value
            except (KeyError, IndexError, TypeError):
                latest_data[sensor] = None
        return latest_data

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {PHONE_URL}")
        print("→ Is your PC on the same Wi-Fi or hotspot as your phone?")
        print("→ Did you enable 'Allow remote access' in Phyphox?")
        return None
    except requests.exceptions.Timeout:
        print("\n⚠️ Connection timed out (phone not responding).")
        return None
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
        return None

def save_to_csv(data):
    """Saves the sensor data to a CSV file."""
    # Path to the CSV file
    file_path = 'location.csv'

    # Check if the file exists to write headers only once
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        fieldnames = SENSORS_TO_GET
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            # Write the header if the file does not exist
            writer.writeheader()

        # Write the data row
        writer.writerow(data)

# --- Main Program Loop ---
if __name__ == "__main__":
    clear_screen()
    print(f"--- Phyphox Real-time Data ---")
    print(f"Connecting to: {FULL_REQUEST_URL}\n")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            data = get_sensor_data()

            if data:
                clear_screen()
                print("--- Phyphox Real-time Data (Connected) ---")
                print("Press Ctrl+C to stop.\n")

                # Display data in the terminal
                for sensor_name, value in data.items():
                    if value is not None:
                        print(f"{sensor_name:<10}: {value:.3f}")
                    else:
                        print(f"{sensor_name:<10}: (No data)")

                # Save the data to the CSV file
                save_to_csv(data)

            time.sleep(1)  # Poll about 10 times per second

    except KeyboardInterrupt:
        print("\n🛑 Stopping script.")

import argparse
from pythonosc import dispatcher
from pythonosc import osc_server

def print_sensor_data(address, *args):
    """
    This function gets called every time a message is received.
    'address' is the OSC address (e.g., '/acceleration')
    'args' is a list of values (e.g., [x, y, z])
    """
    # Use a carriage return \r to overwrite the same line in the terminal
    # The end="" prevents adding a new line
    print(f"\r{address}: {args}", end="")

if _name_ == "_main_":
    # --- Server Setup ---
    # We listen on "0.0.0.0" which means "all available network interfaces"
    # This is important so your phone can find your computer.
    # Do NOT use "127.0.0.1" or "localhost" here.
    listen_ip = "0.0.0.0" 
    listen_port = 9000  # You can change this port if you want

    # Create a dispatcher to route incoming messages
    sensor_dispatcher = dispatcher.Dispatcher()

    # Map all incoming messages to our print_sensor_data function
    # The "*" is a wildcard that matches any sensor address
    # (e.g., /acceleration, /gyro, /magnetometer)
    sensor_dispatcher.map("*", print_sensor_data)

    # --- Start the Server ---
    server = osc_server.BlockingOSCUDPServer(
        (listen_ip, listen_port), sensor_dispatcher)
    
    print(f"--- Sensor Server Started ---")
    print(f"Listening for OSC data on port {listen_port}")
    print("Now, configure your phone app to send data to this computer.")
    print("Press Ctrl+C to stop.")
    
    try:
        server.serve_forever()  # This will block and run the server
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
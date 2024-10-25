import subprocess
import time

import psutil


def get_cpu_utilization():
    return psutil.cpu_percent(interval=1)

def get_ram_utilization():
    mem = psutil.virtual_memory()
    return mem.percent

def get_cpu_temperature_rpi():
    # For Raspberry Pi
    temp = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
    return temp.stdout.strip()

def get_cpu_temperature_jetson():
    # For Jetson Nano
    temp = subprocess.run(['sudo', 'tegra-stats'], capture_output=True, text=True)
    # Parse temperature from the output of tegra-stats (adjust parsing as needed)
    temp_str = [line for line in temp.stdout.splitlines() if 'CPU' in line][0]
    return temp_str

def monitor_system():
    while True:
        cpu = get_cpu_utilization()
        ram = get_ram_utilization()
        temp = None

        # Determine if it's a Pi or Jetson (this could also be set via a config)
        try:
            temp = get_cpu_temperature_rpi()
        except FileNotFoundError:
            temp = get_cpu_temperature_jetson()

        print(f"CPU: {cpu}%, RAM: {ram}%, Temp: {temp}")
        time.sleep(5)  # Adjust the sleep interval as needed

if __name__ == "__main__":
    monitor_system()
if __name__ == "__main__":
    monitor_system()

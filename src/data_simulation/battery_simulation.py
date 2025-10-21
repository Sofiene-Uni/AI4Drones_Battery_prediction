import pandas as pd
import pybamm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils.config import get_value

number_cells_series = 12
number_cells_parallel = 1
    
def load_test_data(file_id, prepared_dir):
    """
    Load predicted current (labels) and measured voltage/current for a given file ID.
    Uses cumulative_flight_time_s as the time base.
    """
    predictions_path = prepared_dir / "test" / "predictions" / f"{file_id}.csv"
    simulation_path = prepared_dir / "test" / "simulation" / f"{file_id}.csv"

    if not predictions_path.exists() or not simulation_path.exists():
        raise FileNotFoundError(f"Missing files for {file_id}")

    # Load predicted current
    df_predictions = pd.read_csv(predictions_path)
    predicted_current = df_predictions["current_a"].values

    # Load simulation data (time, temp, voltage, original current)
    df_sim = pd.read_csv(simulation_path)
    time_sec = df_sim["cumulative_flight_time_s"].values
    measured_voltage = df_sim["battery_status_voltage_v"].values
    measured_current = df_sim["battery_status_current_a"].values
    temperature = df_sim["sensor_baro_temperature"].values

    return time_sec, predicted_current, measured_voltage, measured_current, temperature


def run_battery_simulation(initial_voltage, current_values, cumulative_time, temperatures):
    """
    Run PyBaMM simulation using cumulative flight time.
    Delta time between points is rounded to the nearest second.
    Starts from initial_voltage.
    """
    # Calculate durations between consecutive points
    durations = np.diff(cumulative_time, prepend=cumulative_time[0])
    durations = np.round(durations).astype(int)  # round to whole seconds
    durations[0] = max(durations[0], 1)  # ensure first step is at least 1s

    # Build experiment steps
    experiment_steps = [
        f"Discharge at {current:.3f} A for {duration} s"
        for current, duration in zip(current_values, durations)
    ]
    
    print(experiment_steps)
    experiment = pybamm.Experiment(experiment_steps)

    # Initialize model and parameters
    model = pybamm.lithium_ion.DFN()
    param = pybamm.ParameterValues("Chen2020")


    param['Nominal cell capacity [A.h]'] = 42.0
    param['Number of cells connected in series to make a battery'] = number_cells_series
    param['Number of electrodes connected in parallel to make a cell'] = number_cells_parallel
    param['Ambient temperature [K]'] = 273.15 + 25  # Kelvin (25°C)
    

    # Initialize simulation
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    sim.solve(initial_soc=0.9)

    # Collect results
    solution = sim.solution
    results = pd.DataFrame({
        "Time [s]": solution["Time [s]"].entries,
        "Voltage [V]": solution["Voltage [V]"].entries,
        "Current [A]": solution["Current [A]"].entries
    })


    return results

def compare_voltage_and_current(time_sec, measured_voltage, measured_current, simulation_results):
    """
    Plot predicted voltage vs measured voltage, and predicted current vs measured current.
    Interpolates measured signals to simulation time if lengths differ.
    Scales PyBaMM cell-level outputs to full battery pack.
    
    Args:
        time_sec: array-like, measured time points
        measured_voltage: array-like, measured battery voltage
        measured_current: array-like, measured battery current
        simulation_results: DataFrame from PyBaMM simulation (cell-level)
        N_series: int, number of cells in series
        N_parallel: int, number of cells in parallel
    """
    sim_time = simulation_results["Time [s]"].values
    # Scale cell-level results to battery pack
    predicted_voltage = simulation_results["Voltage [V]"].values * number_cells_series
    predicted_current = simulation_results["Current [A]"].values 

    # Interpolate measured signals to simulation time grid
    measured_voltage_interp = np.interp(sim_time, time_sec, measured_voltage)
    measured_current_interp = np.interp(sim_time, time_sec, measured_current)

    fig, ax1 = plt.subplots(figsize=(14, 9))
    ax1.plot(sim_time, predicted_voltage, label="Predicted Voltage [V]", color="red", linewidth=2)
    ax1.plot(sim_time, measured_voltage_interp, label="Measured Voltage [V]", color="blue", linewidth=2)
    ax1.set_xlabel("Time [s]", fontsize=14)
    ax1.set_ylabel("Voltage [V]", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(sim_time, predicted_current, label="Predicted Current [A]", color="green", alpha=0.7)
    ax2.plot(sim_time, measured_current_interp, label="Measured Current [A]", color="orange", alpha=0.7)
    ax2.set_ylabel("Current [A]", fontsize=14)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Battery Voltage & Current Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()



def run():
    prepared_dir = Path(get_value("paths.prepared_dir", "data/prepared"))
    
    
    print("here")
    


    # Loop through all test files
    test_labels_dir = prepared_dir / "test" / "labels"
    for file_path in test_labels_dir.glob("*.csv"):
        file_id = file_path.stem
        print(f"🔹 Simulating file {file_id}")

        # Load data
        time_sec, predicted_current, measured_voltage, measured_current, temperatures = load_test_data(file_id, prepared_dir)

        # Run PyBaMM simulation with predicted current
        simulation_results = run_battery_simulation(measured_voltage[0],predicted_current,  time_sec, temperatures)

        # Compare predicted vs measured voltage & current
        compare_voltage_and_current(time_sec, measured_voltage, measured_current, simulation_results)

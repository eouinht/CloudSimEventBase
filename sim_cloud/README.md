## Project title
SimCloud: A Cloud-Based Simulation Platform
## Overview
SimCloud is a cloud-based platform designed to facilitate the execution and management of large-scale simulations. It provides users with the ability to run simulations on distributed computing resources, enabling faster processing times and scalability.
## Structure
The project is organized into the following main directories:
- 'core': Contains the core functionalities of the SimCloud platform, including: pm, vm and sate.
- 'datasource': Contains online_trace for load event from online data sources.
- 'engine': Contains the dispatcher modules to manage simulation tasks.
- 'events': Contains vm_event definitions.
- 'scheduler': Contains scheduling algorithms for resource allocation.
- 'models': Handle the vm changes and pm scenarios (2-100pm).
- 'run.py': The main entry point for running simulations.
## Note:
- Make sure models work correctly with the simulation engine (Not done yet)
- Syncronize time between different modules (Not done yet)
- Add more scheduling algorithms (In progress)
## How to run it
1. Create a virtual environment using Python 3.11 or higher.
2. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```
3. Navigate to the project directory and run the simulation using:
   ```
   python run.py
   ```
## Let debug and make it better with deep RL!

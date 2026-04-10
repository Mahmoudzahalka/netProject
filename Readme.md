in order to run the simulation : 

- Install Containernet, don't install any dependencies manually.
- commands to run : 
  from inside containernet directory :
  
  1. python3 -m venv venv
  2. source venv/bin/activate
  3. sudo -E env PATH=$PATH python3 examples/whateversimfile.py (assuming the simulation file is in examples directory)
  
  to close containernet and virtual env:
  
  1. exit command in containernet CLI
  2. sudo -E env PATH=$PATH mn -c
  3. deactivate
  
  
  - Before running:
  -  you need to build the given Dockerfile in a seperate directory :
    
    using the command:
    docker build -t network-multitool-python:latest .


  - To Run the simulation with all the gradients run python3 <selected_sweep_script>.py
    - you can alter what gradients the script runs in the designated field in the script
  
  

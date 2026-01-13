in order to run the simulation : 

- Install Containernet, don't install any dependencies manually.
- commands to run : 
  from inside containernet directory :
  python3 -m venv venv
  source venv/bin/activate
  sudo -E env PATH=$PATH python3 examples/<filename>.py
  
  to close containernet:
  - exit command in containernet CLI
  - sudo -E env PATH=$PATH mn -c
  - deactivate
  
  
  - Before running: you need to build the given Dockerfile in a seperate directory :
    using the command: docker build -t network-multitool-python:latest .
  
  

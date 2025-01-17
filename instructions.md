# Run the service from withint the /home/ollama folder:
./ollama-linux-amd64/bin/ollama serve&

# Kill the process after work:
## Check the processes
ps aux | grep serve
## Kill the process using the PID
kill <PID>

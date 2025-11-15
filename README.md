# Running the Code
On Georgia, run the following command:
```
uvicorn server_openvla:app --host 0.0.0.0 --port 8000 --workers 1
```

On the Jetson, run the following command: 
```
python3 remote_stream_controller.py   --server http://164.67.195.205:8000/   --prompt "pick up the red pen"   --gripper-cam 0   --external-cam -1   --hz 5   --duration 60   --scale 12   --comp-h 720
```
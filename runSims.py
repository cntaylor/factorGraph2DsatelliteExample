import subprocess

N = 10
for i in [4,6]:
    cmd = ['python','OrbitSim.py']
    process = subprocess.Popen(cmd)
    process.wait()
    cmd = ['cp','orbit_run_noisy.npz',f'run_{i}.npz']
    process = subprocess.Popen(cmd)
    process.wait()

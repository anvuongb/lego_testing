import subprocess

subprocess.run('''/usr/bin/leocad -i test.png -w 400 -h 400 --camera-angles 30 30 test.ldr''', shell=True)

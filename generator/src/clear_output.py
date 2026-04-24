import os 
import sys


number_of_files = sys.argv[1]
output_path = "output/"

for i in range(int(number_of_files)):
    try: os.remove(output_path +f"flight_{i}.out")
    except FileNotFoundError: print(f"File not found")
    try: os.remove(output_path +f"flight_{i}_sensors.out")
    except FileNotFoundError: print(f"File not found")


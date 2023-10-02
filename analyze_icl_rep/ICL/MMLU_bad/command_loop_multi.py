import subprocess
from multiprocessing import Pool
from argparse import ArgumentParser



def run_command(command):
    subprocess.run(command, shell=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--num_processes", type=int, default=1
    )
    args = parser.parse_args()

    # Read in the commands from the text file
    with open('commands.txt', 'r') as f:
        commands = f.read().split("\n\n")
        commands = [c.strip() for c in commands if c.strip() ]
        print(len(commands))

    # Set the number of commands to run in parallel
    num_processes = args.num_processes

    # Create a process pool with the specified number of processes
    with Pool(num_processes) as pool:
        # Map the run_command function to the list of commands
        # This will start running the commands in parallel
        pool.map(run_command, commands)

if __name__ == "__main__":
    main()

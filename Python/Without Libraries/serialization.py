import pickle
import os
import sys

import matplotlib.pyplot as plt

# Serialization
def save_model(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def save_training_info(losses, accuracy, filename):
    with open(filename, 'wb') as f:
        pickle.dump(losses, f)
        pickle.dump(accuracy, f)

# Save chart
def save_loss_chart(loss, filename):
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(filename)


def load_model(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    return params

# Output directory creation
def create_output_dir(dir_name):
    parent_dir = '../Without Libraries/output/'
    path = os.path.join(parent_dir, dir_name)
    os.mkdir(path)

    return str(path)

# Run count (for file name creation)
def get_run_count():
    file_name = "../Without Libraries/output/run_count.txt"

    try:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                run_count = int(f.read())
        else:
            run_count = 0

        run_count += 1

        with open(file_name, 'w') as f:
            f.write(str(run_count))

        return run_count
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


PREDICTION_WINDOW = 19 * 7


def main():
    fig = plt.figure(figsize=(12, 6))
    target = np.genfromtxt("target.csv", delimiter=",")
    learned_model = np.genfromtxt("prediction.csv", delimiter=",")
    counter_1_5 = np.genfromtxt("prediction_1_5.csv", delimiter=",")
    counter_2_5 = np.genfromtxt("prediction_2_5.csv", delimiter=",")
    counter_1_10 = np.genfromtxt("prediction_1_10.csv", delimiter=",")
    counter_2_10 = np.genfromtxt("prediction_2_10.csv", delimiter=",")
    
    death_learned_model = int(sum(learned_model[-PREDICTION_WINDOW:]))
    death_counter_1_5 = int(sum(counter_1_5[-PREDICTION_WINDOW:]))
    death_counter_2_5 = int(sum(counter_2_5[-PREDICTION_WINDOW:]))
    death_counter_1_10 = int(sum(counter_1_10[-PREDICTION_WINDOW:]))
    death_counter_2_10 = int(sum(counter_2_10[-PREDICTION_WINDOW:]))

    plt.plot(target[:])
    plt.plot(learned_model[:])

    time = list(range(len(counter_1_5)))

    plt.plot(time[-PREDICTION_WINDOW:], counter_1_5[-PREDICTION_WINDOW:])
    plt.plot(time[-PREDICTION_WINDOW:], counter_2_5[-PREDICTION_WINDOW:])
    plt.plot(time[-PREDICTION_WINDOW:], counter_1_10[-PREDICTION_WINDOW:])
    plt.plot(time[-PREDICTION_WINDOW:], counter_2_10[-PREDICTION_WINDOW:])

    plt.xlabel("TimeStamp")
    plt.ylabel("# Infections")
    plt.legend(["Ground-truth", 
        f"Learned model ({death_learned_model} projected cases)" 
        ,f"Counter factual \u0394=1, x=5% ({death_counter_1_5})"
        ,f"Counter factual \u0394=2, x=5% ({death_counter_2_5})"
        ,f"Counter factual \u0394=1, x=10% ({death_counter_1_10})"
        ,f"Counter factual \u0394=2, x=10% ({death_counter_2_10})"
        ])
    
    fig.savefig(os.path.join("counter_factual.png"))

    
if __name__ == "__main__":
    sys.exit(main())
import numpy as np
import matplotlib.pyplot as plt

# Number of recommendations
k = 5

items = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]

# True reward probabilities
true_rewards = [0.2, 0.5, 0.8, 0.3, 0.6]

# Estimated values
Q = np.zeros(k)

# Count of selections
N = np.zeros(k)

epsilon = 0.1
steps = 1000

rewards = []

for step in range(steps):

    # Exploration
    if np.random.rand() < epsilon:
        action = np.random.randint(k)

    # Exploitation
    else:
        action = np.argmax(Q)

    # Simulated reward
    reward = 1 if np.random.rand() < true_rewards[action] else 0

    # Update count
    N[action] += 1

    # Update estimated value
    Q[action] += (1 / N[action]) * (reward - Q[action])

    rewards.append(reward)

# Average reward
avg_rewards = np.cumsum(rewards) / (np.arange(steps) + 1)

# Plot
plt.plot(avg_rewards)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("AI Recommendation System using K-Bandit")
plt.show()

# Best recommendation
best_item = items[np.argmax(Q)]

print("Best Recommended Item:", best_item)
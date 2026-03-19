import matplotlib.pyplot as plt

voltages_logged = [0.98, 0.99, 0.97, 0.92, 0.96, 0.98] # Example data from your loop
plt.plot(voltages_logged, label='AEG Controlled Voltage')
plt.axhline(y=0.95, color='r', linestyle='--', label='Critical Limit')
plt.title("Grid Resilience Test (Sub-zero Storm Simulation)")
plt.ylabel("Voltage (pu)")
plt.xlabel("Time Steps")
plt.legend()
plt.savefig("resilience_test.png")
plt.show()
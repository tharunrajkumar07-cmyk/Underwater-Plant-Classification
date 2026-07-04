import re
from collections import Counter
import matplotlib.pyplot as plt

# Read text file
with open("Classification Results.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Extract classes
classes = re.findall(r"Predicted Class:\s*(\S+)", text)

# Count classes
counts = Counter(classes)

print("Counts:", counts)

# Plot histogram
plt.figure(figsize=(10,6))
plt.bar(counts.keys(), counts.values())

plt.xlabel("Plant Class")
plt.ylabel("Number of Frames")
plt.title("Underwater Plant Classification Histogram")

# Display values on bars
for i, v in enumerate(counts.values()):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.show()


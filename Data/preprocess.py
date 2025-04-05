import matplotlib.pyplot as plt
from collections import Counter

def inspect_class_distribution(dataset, title):
    labels = dataset.targets.numpy()  # Extract labels as NumPy array
    class_counts = Counter(labels)  

    # Print counts
    print(f"Class distribution in {title}:")
    for digit, count in sorted(class_counts.items()):
        print(f"Digit {digit}: {count}")

    # Plot distribution
    plt.bar(class_counts.keys(), class_counts.values(), tick_label=range(10))
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title(f"Class Distribution in {title}")
    plt.show()
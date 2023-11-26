import numpy as np

k = 2
# Assuming you have a NumPy array with shape [100]
original_array = np.array(
    [
        [
            5,
            1,
            2,
            2,
            3,
            4,
            5,
        ],
        [
            1,
            2,
            3,
            3,
            9,
            8,
            7,
        ],
    ]
)  # Replace this with your array

# Use np.partition to get the five largest elements
partitioned_array = np.partition(original_array, -k)

# Get the five largest elements
largest_elements = partitioned_array[:, -k:]

# Get the other 95 smaller elements
smaller_elements = partitioned_array[:, :-k]

print("Original Array:", original_array)
print("Five Largest Elements:", largest_elements)
print(f"types:{type(largest_elements)}, {type(smaller_elements)}")
print("Other 95 Smaller Elements:", smaller_elements)
print(f"mean: {np.mean(largest_elements, axis=1)}")

def get_valid_integer(prompt):
    """Helper function to get a valid integer input."""
    while True:
        value = input(prompt)
        if value.isdigit():  # Check if the input is a valid positive integer
            return int(value)
        else:
            print("Invalid input. Please enter a valid integer.")
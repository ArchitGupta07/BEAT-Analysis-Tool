import clustering_algo
import clustering_algo_Regular
import clustering_algo_Split
import existing_beat

# ANSI escape codes for colors
class TextColor:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"  # Reset to default color


def get_valid_integer(prompt):
    """Helper function to get a valid integer input."""
    while True:
        value = input(prompt)
        if value.isdigit():
            return int(value)
        else:
            print(f"{TextColor.RED}Invalid input. Please enter a valid integer.{TextColor.RESET}")
            return -1


def handle_clustering_option():
    """Handles clustering options."""
    while True:
        print("\nWhat Coverage Type Clustering would you like to see?")
        print("Option 1: Split Coverage")
        print("Option 2: Regular Coverage")
        print("Option 3: Both Split + Regular")
        user_input_2 = input("Select the option number and press Enter: ")

        if user_input_2 == "1":
            print("\nRunning Split Coverage Clustering...")
            sizes = handle_cluster_sizing(10)
            if sizes:
                clustering_algo_Split.clustering_algo_split_main(sizes[0])
            else:
                clustering_algo_Split.clustering_algo_split_main()

            # clustering_algo_Split.clustering_algo_split_main()
            break

        elif user_input_2 == "2":
            # handle_regular_clustering()
            sizes = handle_cluster_sizing(20)
            if sizes:
                clustering_algo_Regular.clustering_algo_regular_main(sizes[0])
            else:
                clustering_algo_Regular.clustering_algo_regular_main()
            break

        elif user_input_2 == "3":
            print("\nRunning Both Split and Regular Coverage Clustering...")

            sizes = handle_multiple_max_cluster_sizes()
            clustering_algo.clustering_algo_main(sizes[0], sizes[1])
            break

        else:
            print(f"{TextColor.RED}Invalid input for Coverage Type. Please try again.{TextColor.RESET}")


def handle_cluster_sizing(min_limit):
    """Handles the regular clustering option with size validation."""
    while True:
        cluster_size_option = input("Do you want to Change cluster size values: \n Default size Regular -> 40 & Split -> 20 \n Please enter (yes/no)").lower()
        if cluster_size_option == "yes":
            while True:
                max_cluster_size = get_valid_integer("Please Input The max limit: ")
                # min_cluster_size = get_valid_integer("Please Input The min limit: ")

                if  max_cluster_size==-1:
                    continue
                if min_limit > max_cluster_size:
                    print(f"{TextColor.RED}Error: Minimum cluster size cannot be greater than maximum cluster size.{TextColor.RESET}")
                    print("Please try again from the size input step.")
                else:
                    # clustering_algo_Regular.clustering_algo_regular_main(max_cluster_size, min_cluster_size)
                    return [max_cluster_size]
        elif cluster_size_option == "no":
            print("\nRunning Regular Coverage Clustering...")
            # clustering_algo_Regular.clustering_algo_regular_main()
            return []
        else:
            print(f"{TextColor.RED}Invalid choice. Please enter 'yes' or 'no'.{TextColor.RESET}")
            return []

def handle_multiple_max_cluster_sizes():
    
    while True:
        cluster_size_option = input("Do you want to change maximum cluster size values for Split and Regular (Yes/No): ").lower()
        if cluster_size_option == "yes":
            while True:
                max_split_size = get_valid_integer("Please Input the maximum cluster size for Split Coverage: ")
                if max_split_size == -1:
                    continue

                max_regular_size = get_valid_integer("Please Input the maximum cluster size for Regular Coverage: ")
                if max_regular_size == -1:
                    continue

                print(f"{TextColor.GREEN}You have set the maximum cluster sizes: Split = {max_split_size}, Regular = {max_regular_size}.{TextColor.RESET}")
                return [max_split_size, max_regular_size]

        elif cluster_size_option == "no":
            print("\nProceeding with default maximum cluster sizes...")
            return []

        else:
            print(f"{TextColor.RED}Invalid choice. Please enter 'yes' or 'no'.{TextColor.RESET}")



def main():
    while True:
        print(f"{TextColor.GREEN}Welcome to Mapping & Clustering Tool. Select from options below:{TextColor.RESET}")
        print("Option 1: Existing Beat - See routewise salesmen daily beat mapping")
        print("Option 2: New Clustering - See system generated daily beat mapping")
        user_input = input("Select by typing the option no. (1/2) & hit enter: ")

        if user_input == "2":
            handle_clustering_option()
            print(f"{TextColor.YELLOW}Your request is completed. Thank You!!!{TextColor.RESET}")
            print(f"{TextColor.YELLOW}For any queries contact: saket.nayal@in.nestle.com{TextColor.RESET}")
            break

        elif user_input == "1":
            print("\nFetching Existing Salesman Beat Mapping...")
            existing_beat.existing_beat_main()
            print(f"{TextColor.YELLOW}Your request is completed. Thank You!!!{TextColor.RESET}")
            print(f"{TextColor.YELLOW}For any queries contact: saket.nayal@in.nestle.com{TextColor.RESET}")
            break

        else:
            print(f"{TextColor.RED}Invalid input. Please try again.{TextColor.RESET}")


if __name__ == "__main__":
    main()
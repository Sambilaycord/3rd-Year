import threading
import time

# Create a semaphore with an initial value of 3, allowing up to three threads to access the shared resource at a time.
semaphore = threading.Semaphore(3)

# This is the shared resource that all threads will access and modify.
shared_resource = 0

# Function that each thread will run to access and modify the shared resource.
def access_resource():
    global shared_resource  # Access the shared variable
    with semaphore:         # Acquire the semaphore (enters a "critical section")
        shared_resource += 1  # Modify the shared resource
        time.sleep(0.1)  # Add delay to simulate concurrent access
        print(f"Shared resource: {shared_resource}")  # Print the updated value

# List to keep track of all threads created
threads = []

# Create and start 10 threads
for _ in range(10):
    thread = threading.Thread(target=access_resource)  # Create a thread that runs access_resource
    threads.append(thread)  # Add the thread to the list
    thread.start()          # Start the thread, which runs access_resource

# Wait for all threads to complete
for thread in threads:
    thread.join()  # Ensures the main program doesn’t exit until all threads have finished accessing and updating

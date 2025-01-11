import os
import psutil
import multiprocessing

def get_optimal_cpu_count(max_percent=80):
    """
    Get the optimal number of CPU cores to use based on system state
    
    Args:
        max_percent (int): Maximum percentage of total CPUs to use (default: 80%)
    
    Returns:
        int: Recommended number of CPU cores to use
    """
    # Get different CPU count metrics
    total_cpus = multiprocessing.cpu_count()
    physical_cpus = psutil.cpu_count(logical=False)  # Physical cores only
    available_cpus = len(psutil.Process().cpu_affinity())  # Available to this process
    
    # Get current CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory information
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Print system information
    # print("\nSystem CPU Information:")
    # print(f"Total CPU cores (logical): {total_cpus}")
    # print(f"Physical CPU cores: {physical_cpus}")
    # print(f"Available CPU cores: {available_cpus}")
    # print(f"Current CPU usage: {cpu_percent}%")
    # print(f"Current memory usage: {memory_percent}%")
    
    # Calculate recommended number of cores
    recommended_cpus = min(
        available_cpus,
        max(1, int(available_cpus * max_percent / 100))
    )
    
    # Adjust based on current system load
    if cpu_percent > 70:  # High CPU load
        recommended_cpus = max(1, recommended_cpus // 2)
    
    print(f"\nRecommended number of CPU cores to use: {recommended_cpus}")
    
    return recommended_cpus

def monitor_cpu_usage(process_func):
    """
    Decorator to monitor CPU usage during function execution
    
    Usage:
        @monitor_cpu_usage
        def your_function():
            ...
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_cpu_percent = process.cpu_percent()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # print("\nStarting process monitoring...")
        # print(f"Initial CPU usage: {start_cpu_percent}%")
        # print(f"Initial memory usage: {start_memory:.2f} MB")
        
        result = process_func(*args, **kwargs)
        
        end_cpu_percent = process.cpu_percent()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # print("\nProcess monitoring results:")
        # print(f"Final CPU usage: {end_cpu_percent}%")
        # print(f"Final memory usage: {end_memory:.2f} MB")
        # print(f"Memory change: {end_memory - start_memory:.2f} MB")
        
        return result
    
    return wrapper


# # Test the utilities
# if __name__ == "__main__":
#     # Get recommended CPU count
#     n_jobs = get_optimal_cpu_count(max_percent=80)
    
#     # Example of monitoring a simple parallel task
#     @monitor_cpu_usage
#     def test_parallel_processing():
#         with multiprocessing.Pool(n_jobs) as pool:
#             result = pool.map(lambda x: x*x, range(1000000))
#         return result
    
#     test_parallel_processing()
import time
from functools import wraps
from typing import Dict, List, Callable, Any
import statistics


class PerformanceMonitor:
    """Utility class for monitoring and reporting performance metrics"""
    
    def __init__(self):
        self.function_times = {}  # Stores execution times for functions
        self.frame_times = []  # Stores frame processing times
        self.max_history = 100  # Maximum number of entries to keep
        
    def track_function(self, func: Callable) -> Callable:
        """Decorator to track function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Store execution time
            func_name = func.__name__
            if func_name not in self.function_times:
                self.function_times[func_name] = []
                
            self.function_times[func_name].append(end_time - start_time)
            
            # Limit history length
            if len(self.function_times[func_name]) > self.max_history:
                self.function_times[func_name].pop(0)
                
            return result
        return wrapper
        
    def track_frame_time(self, frame_time: float) -> None:
        """Track frame processing time"""
        self.frame_times.append(frame_time)
        
        # Limit history length
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
            
    def get_function_stats(self, func_name: str = None) -> Dict[str, Any]:
        """Get performance statistics for a function or all functions"""
        if func_name is not None:
            if func_name not in self.function_times:
                return {}
            return self._calculate_stats(self.function_times[func_name])
            
        # Get stats for all functions
        stats = {}
        for name, times in self.function_times.items():
            stats[name] = self._calculate_stats(times)
        return stats
        
    def get_frame_stats(self) -> Dict[str, float]:
        """Get frame processing statistics"""
        if not self.frame_times:
            return {"fps": 0, "avg_time": 0, "min_time": 0, "max_time": 0}
            
        return self._calculate_stats(self.frame_times)
        
    def _calculate_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate statistics from a list of times"""
        if not times:
            return {}
            
        avg_time = statistics.mean(times)
        return {
            "avg_time": avg_time,
            "min_time": min(times),
            "max_time": max(times),
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
        }
        
    def reset(self) -> None:
        """Reset all performance metrics"""
        self.function_times = {}
        self.frame_times = []


# Create a global instance for easy import
performance_monitor = PerformanceMonitor()


# Decorator for easy function tracking
def track_performance(func):
    """Decorator to track function performance"""
    return performance_monitor.track_function(func)

import time
import sqlite3
import os
import json
import hashlib
import random
import threading
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


class Tool:
    def __init__(
        self, 
        name: str, 
        func, 
        cost: float, 
        base_ttl: int, 
        deterministic: bool = True, 
        category: str = "api",
        error_rate: float = 0.0,
        timeout: int = 30     ):
        self.name = name
        self.func = func
        self.cost = cost 
        self.base_ttl = base_ttl  
        self.deterministic = deterministic
        self.category = category
        self.error_rate = error_rate
        self.timeout = timeout 
        self.reads_from = []
        self.writes_to = []


class FailureInjector:
    
    def __init__(self):
        self._enabled = False
        self._rate = 0.0
        self._lock = threading.Lock()
    
    def enable(self, rate: float = 0.05):
        with self._lock:
            self._enabled = True
            self._rate = rate
        logger.warning(f"Failure injection ENABLED at {rate*100}% rate")
    
    def disable(self):
        with self._lock:
            self._enabled = False
            self._rate = 0.0
        logger.info("Failure injection DISABLED")
    
    def maybe_fail(self, tool_name: str):
        with self._lock:
            if self._enabled and random.random() < self._rate:
                logger.error(f"INJECTED FAILURE in {tool_name}")
                raise RuntimeError(f"Simulated failure in {tool_name}")


_failure_injector = FailureInjector()


def set_failure_mode(enabled: bool, rate: float = 0.05):
    if enabled:
        _failure_injector.enable(rate)
    else:
        _failure_injector.disable()


def maybe_inject_failure(tool_name: str):
    _failure_injector.maybe_fail(tool_name)

def run_tool_with_timeout(tool_func, tool_name, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool_func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            logger.error(f" TIMEOUT: {tool_name} exceeded {timeout}s timeout")
            future.cancel()  # Attempt to cancel (may not work if already running)
            return {"error": f"Tool execution timed out after {timeout}s", "timed_out": True}
        except Exception as e:
            logger.error(f" TOOL ERROR in {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
        finally:
            # Ensure executor shuts down cleanly
            pass

def get_user_location(user_id: int) -> Dict[str, Any]:
    maybe_inject_failure("get_user_location")
    time.sleep(0.1)
    
    locations = {
        1: {"city": "New York", "lat": 40.7128, "lon": -74.0060},
        2: {"city": "London", "lat": 51.5074, "lon": -0.1278},
        3: {"city": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        4: {"city": "Paris", "lat": 48.8566, "lon": 2.3522},
        5: {"city": "Berlin", "lat": 52.5200, "lon": 13.4050},
        6: {"city": "Sydney", "lat": -33.8688, "lon": 151.2093},
        7: {"city": "Toronto", "lat": 43.6532, "lon": -79.3832},
        8: {"city": "Singapore", "lat": 1.3521, "lon": 103.8198},
        9: {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        10: {"city": "Dubai", "lat": 25.2048, "lon": 55.2708},
    }
    
    if user_id in locations:
        return {"user_id": user_id, **locations[user_id]}

    lat = 30 + (hash(f"lat{user_id}") % 40)
    lon = -120 + (hash(f"lon{user_id}") % 240)
    city_names = ["Boston", "Seattle", "Austin", "Denver", "Portland", "Atlanta", "Phoenix"]
    city = city_names[user_id % len(city_names)]
    
    return {"user_id": user_id, "city": city, "lat": lat, "lon": lon}


def _get_weather_impl(city: str) -> Dict[str, Any]:
    time.sleep(1.5) 

    temp = 15 + (hash(city) % 20)
    conditions = ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy"]
    condition = conditions[hash(city) % len(conditions)]
    
    return {
        "city": city,
        "temperature": temp,
        "condition": condition,
        "humidity": 50 + (hash(city + "humidity") % 40),
        "wind_speed": 5 + (hash(city + "wind") % 20)
    }


def get_weather(city: str) -> Dict[str, Any]:
    maybe_inject_failure("get_weather")
    return run_tool_with_timeout(_get_weather_impl, "get_weather", 10, city)


def update_user_location(user_id: int, city: str, lat: float, lon: float) -> Dict[str, Any]:
    maybe_inject_failure("update_user_location")
    time.sleep(0.2)
    logger.info(f"Updated user {user_id} location to {city}")
    return {"user_id": user_id, "city": city, "lat": lat, "lon": lon, "status": "updated"}


def get_coordinates(city: str) -> Dict[str, Any]:
    maybe_inject_failure("get_coordinates")
    time.sleep(0.15)
    
    return {
        "city": city,
        "lat": 40.0 + (hash(city) % 40),
        "lon": -120.0 + (hash(city) % 240),
        "country": "Unknown"
    }


def get_timezone(city: str) -> Dict[str, Any]:
    maybe_inject_failure("get_timezone")
    time.sleep(0.12)
    
    timezones = ["UTC-8", "UTC-5", "UTC+0", "UTC+1", "UTC+9"]
    tz = timezones[hash(city) % len(timezones)]
    
    return {"city": city, "timezone": tz, "offset_hours": int(tz.split("UTC")[1])}


def get_currency_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    maybe_inject_failure("get_currency_rate")
    time.sleep(0.2)
    
    rate = 1.0 + (hash(f"{from_currency}{to_currency}") % 100) / 100.0
    
    return {
        "from": from_currency,
        "to": to_currency,
        "rate": round(rate, 4),
        "timestamp": time.time()
    }

DB_PATH = "cache_test.db"
_db_lock = threading.RLock()


def _init_database():
    conn = None
    try:
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    age INTEGER,
                    department TEXT,
                    salary REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT,
                    price REAL,
                    category TEXT,
                    stock INTEGER,
                    rating REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    total_price REAL,
                    timestamp TEXT
                )
            """)
            
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                departments = ["Engineering", "Sales", "Marketing", "Support", "Finance"]
                users = [(i, f"User{i}", f"user{i}@test.com", 20 + i % 50, 
                         departments[i % len(departments)], 50000 + i * 1000) 
                         for i in range(1, 201)]
                cursor.executemany("INSERT INTO users VALUES (?,?,?,?,?,?)", users)
                
                categories = ["Electronics", "Clothing", "Food", "Books", "Home"]
                products = [(i, f"Product{i}", 10.0 + i * 5, 
                            categories[i % len(categories)], 
                            100 + i * 10,
                            3.0 + (i % 20) / 10.0) for i in range(1, 101)]
                cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()


_init_database()


def db_query_user(user_id: int) -> Dict[str, Any]:
    maybe_inject_failure("db_query_user")
    time.sleep(0.05)
    
    try:
        with _db_lock:
            with sqlite3.connect(DB_PATH) as conn:  # Context manager ensures closure
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        "user_id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "age": row[3],
                        "department": row[4],
                        "salary": row[5]
                    }
                return {"error": "User not found"}
                
    except Exception as e:
        logger.error(f"Database error in db_query_user: {e}")
        return {"error": f"Database error: {str(e)}"}

def db_query_products_by_category(category: str) -> Dict[str, Any]:
    maybe_inject_failure("db_query_products_by_category")
    time.sleep(0.08)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        with _db_lock:            
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE category = ? LIMIT 20", (category,))
            rows = cursor.fetchall()
            
            products = [{"id": r[0], "name": r[1], "price": r[2], "stock": r[4], "rating": r[5]} 
                        for r in rows]
            return {"category": category, "products": products, "count": len(products)}
            
    except Exception as e:
        logger.error(f"Database error in db_query_products_by_category: {e}")
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            conn.close()


def db_complex_aggregation(department: str) -> Dict[str, Any]:
    maybe_inject_failure("db_complex_aggregation")
    time.sleep(0.15)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        with _db_lock:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT department, COUNT(*) as count, AVG(age) as avg_age, 
                       AVG(salary) as avg_salary, MAX(salary) as max_salary
                FROM users 
                WHERE department = ?
                GROUP BY department
            """, (department,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "department": row[0],
                    "employee_count": row[1],
                    "average_age": round(row[2], 1),
                    "average_salary": round(row[3], 2),
                    "max_salary": row[4]
                }
            return {"error": "Department not found"}
            
    except Exception as e:
        logger.error(f"Database error in db_complex_aggregation: {e}")
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            conn.close()


def db_join_query(user_id: int) -> Dict[str, Any]:
    maybe_inject_failure("db_join_query")
    time.sleep(0.2)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        with _db_lock:
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT o.order_id, p.name, o.quantity, o.total_price, o.timestamp
                FROM orders o 
                JOIN products p ON o.product_id = p.product_id 
                WHERE o.user_id = ?
                ORDER BY o.timestamp DESC
                LIMIT 10
            """, (user_id,))
            rows = cursor.fetchall()
            
            orders = [{"order_id": r[0], "product": r[1], "quantity": r[2], 
                       "total": r[3], "timestamp": r[4]} for r in rows]
            return {"user_id": user_id, "orders": orders, "total_orders": len(orders)}
            
    except Exception as e:
        logger.error(f"Database error in db_join_query: {e}")
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            conn.close()


def db_write_order(user_id: int, product_id: int, quantity: int) -> Dict[str, Any]:
    maybe_inject_failure("db_write_order")
    time.sleep(0.1)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        with _db_lock:
            
            cursor = conn.cursor()
            
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            result = cursor.fetchone()
            if not result:
                return {"error": "Product not found"}
            
            price = result[0]
            total_price = price * quantity
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute(
                "INSERT INTO orders (user_id, product_id, quantity, total_price, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, product_id, quantity, total_price, timestamp)
            )
            order_id = cursor.lastrowid
            conn.commit()
            
            return {
                "order_id": order_id,
                "user_id": user_id,
                "product_id": product_id,
                "quantity": quantity,
                "total_price": total_price,
                "status": "created"
            }
            
    except Exception as e:
        logger.error(f"Database error in db_write_order: {e}")
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            conn.close()


CACHE_DIR = Path("cache_test_files")
CACHE_DIR.mkdir(exist_ok=True)
_fs_lock = threading.RLock()


def fs_read_file(filename: str) -> Dict[str, Any]:
    maybe_inject_failure("fs_read_file")
    time.sleep(0.03)
    
    with _fs_lock:
        filepath = CACHE_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                return {"filename": filename, "content": content, "size": len(content)}
            except Exception as e:
                logger.error(f"File read error: {e}")
                return {"error": f"File read error: {str(e)}"}
        
        content = f"Content of {filename}"
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            return {"filename": filename, "content": content, "size": len(content)}
        except Exception as e:
            logger.error(f"File write error: {e}")
            return {"error": f"File write error: {str(e)}"}


def fs_write_file(filename: str, content: str) -> Dict[str, Any]:
    maybe_inject_failure("fs_write_file")
    time.sleep(0.05)
    
    with _fs_lock:
        filepath = CACHE_DIR / filename
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            return {"filename": filename, "size": len(content), "status": "written"}
        except Exception as e:
            logger.error(f"File write error: {e}")
            return {"error": f"File write error: {str(e)}"}


def fs_read_json_config(config_name: str) -> Dict[str, Any]:
    maybe_inject_failure("fs_read_json_config")
    time.sleep(0.04)
    
    with _fs_lock:
        filepath = CACHE_DIR / f"{config_name}.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return {"config_name": config_name, "config": data}
            except Exception as e:
                logger.error(f"JSON read error: {e}")
                return {"error": f"JSON read error: {str(e)}"}
        
        default_config = {
            "version": "1.0",
            "settings": {
                "timeout": 30,
                "retry": 3,
                "enabled": True
            }
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(default_config, f)
            return {"config_name": config_name, "config": default_config}
        except Exception as e:
            logger.error(f"JSON write error: {e}")
            return {"error": f"JSON write error: {str(e)}"}


def fs_list_directory(directory: str = ".") -> Dict[str, Any]:
    maybe_inject_failure("fs_list_directory")
    time.sleep(0.02)
    
    with _fs_lock:
        path = CACHE_DIR / directory
        if path.exists() and path.is_dir():
            try:
                files = [f.name for f in path.iterdir()]
                return {"directory": directory, "files": files, "count": len(files)}
            except Exception as e:
                logger.error(f"Directory list error: {e}")
                return {"error": f"Directory list error: {str(e)}"}
    
    return {"directory": directory, "files": [], "count": 0}


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, Any]:
    maybe_inject_failure("calculate_distance")
    time.sleep(0.05)
    
    from math import radians, sin, cos, sqrt, atan2
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = 6371 * c
    
    return {"distance_km": round(distance_km, 2), "distance_miles": round(distance_km * 0.621371, 2)}


def _compute_fibonacci_impl(n: int) -> Dict[str, Any]:
    def fib(x):
        if x <= 1:
            return x
        a, b = 0, 1
        for _ in range(2, x + 1):
            a, b = b, a + b
        return b
    
    n = min(abs(n), 100)
    result = fib(n)
    return {"n": n, "result": result}


def compute_fibonacci(n: int) -> Dict[str, Any]:
    maybe_inject_failure("compute_fibonacci")
    return run_tool_with_timeout(_compute_fibonacci_impl, "compute_fibonacci", 10, n)


def compute_hash_expensive(data: str, iterations: int = 10000) -> Dict[str, Any]:
    maybe_inject_failure("compute_hash_expensive")
    time.sleep(0.3)
    
    result = data.encode()
    for _ in range(min(iterations, 10000)):
        result = hashlib.sha256(result).digest()
    
    return {
        "data_hash": result.hex()[:16],
        "iterations": iterations,
        "algorithm": "SHA256"
    }


def compute_statistics(numbers: list) -> Dict[str, Any]:
    maybe_inject_failure("compute_statistics")
    time.sleep(0.1)
    
    if not numbers:
        return {"error": "Empty list"}
    
    try:
        arr = np.array(numbers)
        return {
            "count": len(numbers),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    except Exception as e:
        logger.error(f"Statistics computation error: {e}")
        return {"error": f"Statistics computation failed: {str(e)}"}


def external_fast_api(query: str) -> Dict[str, Any]:
    maybe_inject_failure("external_fast_api")
    time.sleep(0.05)
    return {
        "query": query,
        "result": f"fast_result_{hash(query) % 1000}",
        "latency_ms": 50,
        "service": "fast-api"
    }


def external_medium_api(query: str) -> Dict[str, Any]:
    maybe_inject_failure("external_medium_api")
    time.sleep(0.5)
    return {
        "query": query,
        "result": f"medium_result_{hash(query) % 1000}",
        "latency_ms": 500,
        "service": "medium-api"
    }


def _external_slow_api_impl(query: str) -> Dict[str, Any]:
    time.sleep(2.0)
    return {
        "query": query,
        "result": f"slow_result_{hash(query) % 1000}",
        "latency_ms": 2000,
        "service": "slow-api"
    }


def external_slow_api(query: str) -> Dict[str, Any]:
    maybe_inject_failure("external_slow_api")

    return run_tool_with_timeout(_external_slow_api_impl, "external_slow_api", 10, query)

TOOLS = {

    "get_user_location": Tool("get_user_location", get_user_location, 100, 600, True, "api", timeout=5),
    "get_weather": Tool("get_weather", get_weather, 1500, 300, True, "api", timeout=10),
    "update_user_location": Tool("update_user_location", update_user_location, 200, 0, True, "api", timeout=5),
    "get_coordinates": Tool("get_coordinates", get_coordinates, 150, 600, True, "api", timeout=5),
    "get_timezone": Tool("get_timezone", get_timezone, 120, 1800, True, "api", timeout=5),
    "get_currency_rate": Tool("get_currency_rate", get_currency_rate, 200, 300, True, "api", timeout=5),

    "db_query_user": Tool("db_query_user", db_query_user, 50, 300, True, "database", timeout=5),
    "db_query_products": Tool("db_query_products_by_category", db_query_products_by_category, 80, 600, True, "database", timeout=5),
    "db_aggregation": Tool("db_complex_aggregation", db_complex_aggregation, 150, 300, True, "database", timeout=5),
    "db_join_query": Tool("db_join_query", db_join_query, 200, 300, True, "database", timeout=5),
    "db_write_order": Tool("db_write_order", db_write_order, 100, 0, True, "database", timeout=5),

    "fs_read_file": Tool("fs_read_file", fs_read_file, 30, 600, True, "filesystem", timeout=5),
    "fs_write_file": Tool("fs_write_file", fs_write_file, 50, 0, True, "filesystem", timeout=5),
    "fs_read_config": Tool("fs_read_json_config", fs_read_json_config, 40, 1800, True, "filesystem", timeout=5),
    "fs_list_dir": Tool("fs_list_directory", fs_list_directory, 20, 300, True, "filesystem", timeout=5),

    "calculate_distance": Tool("calculate_distance", calculate_distance, 50, 3600, True, "compute", timeout=5),
    "compute_fibonacci": Tool("compute_fibonacci", compute_fibonacci, 500, 3600, True, "compute", timeout=10),
    "compute_hash": Tool("compute_hash_expensive", compute_hash_expensive, 300, 1800, True, "compute", timeout=5),
    "compute_statistics": Tool("compute_statistics", compute_statistics, 100, 1800, True, "compute", timeout=5),
    

    "external_fast": Tool("external_fast_api", external_fast_api, 50, 300, True, "external", timeout=5),
    "external_medium": Tool("external_medium_api", external_medium_api, 500, 450, True, "external", timeout=5),
    "external_slow": Tool("external_slow_api", external_slow_api, 2000, 600, True, "external", timeout=10),
}

TOOLS["get_user_location"].reads_from = ["user_db"]
TOOLS["update_user_location"].writes_to = ["user_db"]
TOOLS["db_query_user"].reads_from = ["database"]
TOOLS["db_query_products"].reads_from = ["database"]
TOOLS["db_aggregation"].reads_from = ["database"]
TOOLS["db_join_query"].reads_from = ["database"]
TOOLS["db_write_order"].writes_to = ["database"]
TOOLS["fs_read_file"].reads_from = ["filesystem"]
TOOLS["fs_write_file"].writes_to = ["filesystem"]
TOOLS["fs_read_config"].reads_from = ["filesystem"]


def get_tools_by_category(category: str):
    return {name: tool for name, tool in TOOLS.items() if tool.category == category}

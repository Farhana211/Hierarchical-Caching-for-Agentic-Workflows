import random
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
import logging

logger = logging.getLogger(__name__)

class EnhancedBenchmarkLoader:
    
    def __init__(self, seed=42, write_ratio=0.04, num_questions=10000, distribution="zipfian", zipf_alpha=1.5):
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        self.write_ratio = write_ratio
        self.num_questions = num_questions
        self.distribution = distribution
        self.zipf_alpha = zipf_alpha

        self.cities = [
            "New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", 
            "Toronto", "Moscow", "Beijing", "Mumbai", "Dubai", "Singapore",
            "Los Angeles", "Chicago", "San Francisco", "Boston", "Seattle",
            "Miami", "Houston", "Atlanta", "Madrid", "Rome", "Barcelona",
            "Amsterdam", "Stockholm", "Copenhagen", "Vienna", "Prague",
            "Warsaw", "Budapest", "Istanbul", "Cairo", "Rio", "Mexico City"
        ]
        
        self.popular_cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney"]
        self.popular_users = list(range(1, 51))
        
        self.departments = ["Engineering", "Sales", "Marketing", "Support", "Finance", "HR"]
        self.product_categories = ["Electronics", "Clothing", "Food", "Books", "Home"]
        self.currencies = ["USD", "EUR", "GBP", "JPY", "CNY", "AUD"]
        
        self._init_query_templates()
    
    def _init_query_templates(self):
        
        self.weather_templates = [
            "What's the weather in {city}?",
            "Tell me about the weather conditions in {city}",
            "How's the weather in {city} today?",
            "What's the temperature in {city}?",
            "Is it raining in {city}?",
            "What's the forecast for {city}?",
            "How cold is it in {city}?",
            "Weather report for {city}",
            "Current conditions in {city}?",
            "Compare weather in {city1} and {city2}",
            "Which is warmer, {city1} or {city2}?",
            "Tell me the weather for {city1}, {city2}, and {city3}"
        ]
        
        self.location_templates = [
            "Where is user {user_id} located?",
            "What's the location of user {user_id}?",
            "Tell me where user {user_id} is",
            "Get coordinates for user {user_id}",
            "Find user {user_id}'s location",
            "Where is user {user_id} right now?"
        ]
        
        self.distance_templates = [
            "What's the distance between user {user1} and user {user2}?",
            "How far is user {user1} from user {user2}?",
            "Calculate distance from user {user1} to user {user2}",
            "Distance between users {user1} and {user2}"
        ]

        self.db_user_templates = [
            "Get information for user {user_id}",
            "Show me user {user_id}'s profile",
            "What are the details of user {user_id}?",
            "Lookup user {user_id}"
        ]
        
        self.db_product_templates = [
            "Show me all {category} products",
            "List products in {category} category",
            "What {category} items are available?",
            "Get {category} products"
        ]
        
        self.db_aggregation_templates = [
            "Show statistics for {department} department",
            "What are the stats for {department}?",
            "Aggregate data for {department} employees",
            "Get {department} department summary"
        ]
        
        self.db_order_templates = [
            "Show orders for user {user_id}",
            "What has user {user_id} ordered?",
            "Get purchase history for user {user_id}",
            "List user {user_id}'s orders"
        ]
        
        self.fs_read_templates = [
            "Read file {filename}",
            "Show me contents of {filename}",
            "Get file {filename}",
            "What's in {filename}?"
        ]
        
        self.fs_config_templates = [
            "Load {config_name} configuration",
            "Get {config_name} settings",
            "Read {config_name} config",
            "Show {config_name} configuration"
        ]
        
        self.compute_templates = [
            "Calculate fibonacci of {n}",
            "Compute fibonacci number {n}",
            "What's the {n}th fibonacci number?",
            "Get statistics for numbers {numbers}",
            "Analyze data: {numbers}",
            "Calculate stats for {numbers}"
        ]
        
        self.external_templates = [
            "Query external service with {query}",
            "Search for {query}",
            "Look up {query} externally",
            "Get external data for {query}"
        ]

        self.workflow_templates = [
            "Where is user {user_id} and what's the weather there?",
            "Get user {user_id}'s location and current weather",
            "What's the weather where user {user_id} is?",
            "Show me user {user_id}'s profile and recent orders",
            "Get user {user_id} info and their department statistics"
        ]
    
    def _zipf_choice(self, items: List, alpha: float = None) -> Any:
        """Select item from list using Zipfian distribution"""
        if alpha is None:
            alpha = self.zipf_alpha
        weights = np.array([1.0 / (i + 1) ** alpha for i in range(len(items))])
        weights /= weights.sum()
        return np.random.choice(items, p=weights)
    
    def _uniform_choice(self, items: List) -> Any:
        """Select item from list using uniform distribution"""
        return random.choice(items)
    
    def _bimodal_choice(self, items: List) -> Any:
        """Select item using bimodal distribution: 80% from top 20%, 20% from rest"""
        if random.random() < 0.8:
            # 80% of time: choose from top 20% of items (concentrated)
            top_n = max(1, len(items) // 5)
            return random.choice(items[:top_n])
        else:
            # 20% of time: choose from any item (exploratory)
            return random.choice(items)
    
    def _distribution_choice(self, items: List) -> Any:
        """Select item based on configured distribution"""
        if self.distribution == "uniform":
            return self._uniform_choice(items)
        elif self.distribution == "bimodal":
            return self._bimodal_choice(items)
        else:  # zipfian (default)
            return self._zipf_choice(items)
    
    def generate_weather_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        for i in range(int(num_samples * 0.6)):
            city = self._distribution_choice(self.popular_cities)
            template = random.choice([t for t in self.weather_templates 
                                     if "{city}" in t and "{city1}" not in t])
            questions.append({
                "id": f"weather_{i}",
                "question": template.format(city=city),
                "query_type": "weather_single",
                "category": "api",
                "entities": [city],
                "expected_tools": ["get_weather"],
                "expected_cache_behavior": "likely_hit" if i > 100 else "cold_start"
            })
        
        for i in range(int(num_samples * 0.2)):
            cities = random.sample(self.popular_cities, min(2, len(self.popular_cities)))
            template = random.choice([t for t in self.weather_templates if "{city1}" in t])
            
            if "{city3}" in template:
                cities = random.sample(self.popular_cities, min(3, len(self.popular_cities)))
                question_text = template.format(city1=cities[0], city2=cities[1], city3=cities[2])
            else:
                question_text = template.format(city1=cities[0], city2=cities[1])
            
            questions.append({
                "id": f"weather_comp_{i}",
                "question": question_text,
                "query_type": "weather_comparison",
                "category": "api",
                "entities": cities,
                "expected_tools": ["get_weather"] * len(cities),
                "expected_cache_behavior": "partial_hit"
            })
        
        for i in range(num_samples - len(questions)):
            city = random.choice(self.cities)
            template = random.choice([t for t in self.weather_templates 
                                     if "{city}" in t and "{city1}" not in t])
            questions.append({
                "id": f"weather_diverse_{i}",
                "question": template.format(city=city),
                "query_type": "weather_diverse",
                "category": "api",
                "entities": [city],
                "expected_tools": ["get_weather"],
                "expected_cache_behavior": "likely_miss"
            })
        
        return questions
    
    def generate_location_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        for i in range(int(num_samples * 0.7)):
            user_id = self._distribution_choice(self.popular_users)
            template = random.choice(self.location_templates)
            questions.append({
                "id": f"location_{i}",
                "question": template.format(user_id=user_id),
                "query_type": "location_single",
                "category": "api",
                "entities": [f"user_{user_id}"],
                "expected_tools": ["get_user_location"],
                "expected_cache_behavior": "likely_hit" if i > 200 else "cold_start"
            })

        for i in range(num_samples - len(questions)):
            user1, user2 = random.sample(self.popular_users, 2)
            template = random.choice(self.distance_templates)
            questions.append({
                "id": f"distance_{i}",
                "question": template.format(user1=user1, user2=user2),
                "query_type": "distance_calculation",
                "category": "compute",
                "entities": [f"user_{user1}", f"user_{user2}"],
                "expected_tools": ["get_user_location", "get_user_location", "calculate_distance"],
                "expected_cache_behavior": "workflow_cacheable"
            })
        
        return questions
    
    def generate_database_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []

        for i in range(int(num_samples * 0.4)):
            user_id = random.randint(1, 200)
            template = random.choice(self.db_user_templates)
            questions.append({
                "id": f"db_user_{i}",
                "question": template.format(user_id=user_id),
                "query_type": "db_user_query",
                "category": "database",
                "entities": [f"user_{user_id}"],
                "expected_tools": ["db_query_user"],
                "expected_cache_behavior": "likely_hit"
            })
        
        for i in range(int(num_samples * 0.25)):
            category = random.choice(self.product_categories)
            template = random.choice(self.db_product_templates)
            questions.append({
                "id": f"db_product_{i}",
                "question": template.format(category=category),
                "query_type": "db_product_query",
                "category": "database",
                "entities": [category],
                "expected_tools": ["db_query_products"],
                "expected_cache_behavior": "likely_hit"
            })
        
        for i in range(int(num_samples * 0.2)):
            dept = random.choice(self.departments)
            template = random.choice(self.db_aggregation_templates)
            questions.append({
                "id": f"db_agg_{i}",
                "question": template.format(department=dept),
                "query_type": "db_aggregation",
                "category": "database",
                "entities": [dept],
                "expected_tools": ["db_aggregation"],
                "expected_cache_behavior": "likely_hit"
            })
        
        for i in range(num_samples - len(questions)):
            user_id = random.randint(1, 200)
            template = random.choice(self.db_order_templates)
            questions.append({
                "id": f"db_orders_{i}",
                "question": template.format(user_id=user_id),
                "query_type": "db_order_history",
                "category": "database",
                "entities": [f"user_{user_id}"],
                "expected_tools": ["db_join_query"],
                "expected_cache_behavior": "mixed"
            })
        
        return questions
    
    def generate_filesystem_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        filenames = [f"data_{i}.txt" for i in range(20)] + \
                   [f"log_{i}.txt" for i in range(10)] + \
                   ["config.txt", "settings.txt", "data.json"]
        
        config_names = ["app_config", "db_config", "api_config", "system_config", "user_prefs"]

        for i in range(int(num_samples * 0.6)):
            filename = self._distribution_choice(filenames)
            template = random.choice(self.fs_read_templates)
            questions.append({
                "id": f"fs_read_{i}",
                "question": template.format(filename=filename),
                "query_type": "fs_read",
                "category": "filesystem",
                "entities": [filename],
                "expected_tools": ["fs_read_file"],
                "expected_cache_behavior": "likely_hit"
            })

        for i in range(num_samples - len(questions)):
            config = self._distribution_choice(config_names)
            template = random.choice(self.fs_config_templates)
            questions.append({
                "id": f"fs_config_{i}",
                "question": template.format(config_name=config),
                "query_type": "fs_config",
                "category": "filesystem",
                "entities": [config],
                "expected_tools": ["fs_read_config"],
                "expected_cache_behavior": "likely_hit"
            })
        
        return questions
    
    def generate_compute_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []

        for i in range(int(num_samples * 0.4)):
            n = random.randint(5, 50)
            template = random.choice([t for t in self.compute_templates if "fibonacci" in t])
            questions.append({
                "id": f"compute_fib_{i}",
                "question": template.format(n=n),
                "query_type": "compute_fibonacci",
                "category": "compute",
                "entities": [n],
                "expected_tools": ["compute_fibonacci"],
                "expected_cache_behavior": "likely_hit"
            })

        for i in range(num_samples - len(questions)):
            numbers = [random.randint(1, 100) for _ in range(random.randint(5, 15))]
            numbers_str = str(numbers)
            template = random.choice([t for t in self.compute_templates if "statistics" in t or "stats" in t])
            questions.append({
                "id": f"compute_stats_{i}",
                "question": template.format(numbers=numbers_str),
                "query_type": "compute_statistics",
                "category": "compute",
                "entities": numbers,
                "expected_tools": ["compute_statistics"],
                "expected_cache_behavior": "mixed"
            })
        
        return questions
    
    def generate_external_api_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        api_types = ["fast", "medium", "slow"]
        query_terms = ["weather", "news", "stock", "crypto", "sports", "traffic", "events"]
        
        for i in range(num_samples):
            api_type = random.choices(api_types, weights=[0.5, 0.3, 0.2])[0]
            query = self._distribution_choice(query_terms)
            template = random.choice(self.external_templates)
            
            questions.append({
                "id": f"external_{api_type}_{i}",
                "question": template.format(query=query),
                "query_type": f"external_{api_type}",
                "category": "external",
                "entities": [query],
                "expected_tools": [f"external_{api_type}"],
                "expected_cache_behavior": "likely_hit" if api_type == "slow" else "mixed"
            })
        
        return questions
    
    def generate_workflow_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        for i in range(num_samples):
            user_id = self._distribution_choice(self.popular_users)
            template = random.choice(self.workflow_templates)
            
            if "weather" in template:
                expected_tools = ["get_user_location", "get_weather"]
                query_type = "workflow_location_weather"
            else:
                expected_tools = ["db_query_user", "db_join_query"]
                query_type = "workflow_user_orders"
            
            questions.append({
                "id": f"workflow_{i}",
                "question": template.format(user_id=user_id),
                "query_type": query_type,
                "category": "workflow",
                "entities": [f"user_{user_id}"],
                "expected_tools": expected_tools,
                "expected_cache_behavior": "workflow_cacheable"
            })
        
        return questions
    
    def generate_full_benchmark(self, total_samples: int = 10000) -> List[Dict[str, Any]]:
        """Generate complete benchmark with proper distribution support"""
        
        if total_samples is None:
            total_samples = getattr(self, 'num_questions', 10000)
        
        write_ratio = getattr(self, 'write_ratio', 0.04)
        
        # Calculate category allocations
        weather_location_count = int(total_samples * 0.23)
        database_count = int(total_samples * 0.23)
        filesystem_count = int(total_samples * 0.13)
        compute_count = int(total_samples * 0.13)
        external_count = int(total_samples * 0.08)
        workflow_count = int(total_samples * 0.10)
        write_count = int(total_samples * write_ratio)
        
        allocated_sum = (weather_location_count + database_count + filesystem_count + 
                        compute_count + external_count + workflow_count + write_count)
        staleness_count = max(0, total_samples - allocated_sum)
        
        if staleness_count == 0 and allocated_sum < total_samples:
            diff = total_samples - allocated_sum
            weather_location_count += diff
            logger.debug(f"Adjusted allocation by {diff} questions to reach {total_samples}")
        
        # Initialize questions list
        all_questions = []
        
        # Generate all question types
        all_questions.extend(self.generate_weather_queries(int(weather_location_count * 0.6)))
        all_questions.extend(self.generate_location_queries(int(weather_location_count * 0.4)))
        all_questions.extend(self.generate_database_queries(database_count))
        all_questions.extend(self.generate_filesystem_queries(filesystem_count))
        all_questions.extend(self.generate_compute_queries(compute_count))
        all_questions.extend(self.generate_external_api_queries(external_count))
        all_questions.extend(self.generate_workflow_queries(workflow_count))
        all_questions.extend(self.generate_write_operations(write_count))
        all_questions.extend(self.generate_staleness_pattern_queries(staleness_count))
        
        # Shuffle and add metadata
        random.shuffle(all_questions)
        for i, q in enumerate(all_questions):
            q["sequence_id"] = i
            q["timestamp"] = (datetime.now() + timedelta(seconds=i)).isoformat()
        
        return all_questions
        

    def generate_write_operations(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []
        
        write_templates = {
            "update_location": [
                "Update user {user_id} location to {city}",
                "Move user {user_id} to {city}",
                "Set user {user_id}'s location as {city}",
            ],
            "create_order": [
                "User {user_id} orders product {product_id}, quantity {quantity}",
                "Create order: user {user_id}, product {product_id}, qty {quantity}",
                "Place order for user {user_id}: {quantity}x product {product_id}",
            ],
            "write_file": [
                "Write to file {filename}: {content}",
                "Save {content} to {filename}",
                "Update {filename} with {content}",
            ]
        }
        
        for i in range(num_samples):
            operation = random.choice(["update_location", "create_order", "write_file"])
            
            if operation == "update_location":
                user_id = self._distribution_choice(self.popular_users)
                city = random.choice(self.cities)
                template = random.choice(write_templates["update_location"])
                
                questions.append({
                    "id": f"write_location_{i}",
                    "question": template.format(user_id=user_id, city=city),
                    "query_type": "write_location",
                    "category": "api",
                    "entities": [f"user_{user_id}", city],
                    "expected_tools": ["update_user_location"],
                    "invalidates": ["get_user_location"],
                    "expected_cache_behavior": "invalidates_cache",
                    "is_write": True
                })
            
            elif operation == "create_order":
                user_id = random.randint(1, 200)
                product_id = random.randint(1, 100)
                quantity = random.randint(1, 5)
                template = random.choice(write_templates["create_order"])
                
                questions.append({
                    "id": f"write_order_{i}",
                    "question": template.format(
                        user_id=user_id, 
                        product_id=product_id,
                        quantity=quantity
                    ),
                    "query_type": "write_order",
                    "category": "database_write",
                    "entities": [f"user_{user_id}", f"product_{product_id}"],
                    "expected_tools": ["db_write_order"],
                    "invalidates": ["db_join_query", "db_query_user"],
                    "expected_cache_behavior": "invalidates_cache",
                    "is_write": True
                })
            
            elif operation == "write_file":
                filename = f"data_{random.randint(1, 20)}.txt"
                content = f"Updated content {i}"
                template = random.choice(write_templates["write_file"])
                
                questions.append({
                    "id": f"write_file_{i}",
                    "question": template.format(filename=filename, content=content),
                    "query_type": "write_file",
                    "category": "filesystem_write",
                    "entities": [filename],
                    "expected_tools": ["fs_write_file"],
                    "invalidates": ["fs_read_file"],
                    "expected_cache_behavior": "invalidates_cache",
                    "is_write": True
                })
        
        return questions
    
    def generate_staleness_pattern_queries(self, num_samples: int) -> List[Dict[str, Any]]:
        questions = []

        base_queries = [
            ("get_weather", {"city": "New York"}, "What's the weather in New York?"),
            ("get_weather", {"city": "London"}, "What's the weather in London?"),
            ("get_weather", {"city": "Tokyo"}, "What's the weather in Tokyo?"),
            ("get_user_location", {"user_id": 1}, "Where is user 1 located?"),
            ("get_user_location", {"user_id": 2}, "Where is user 2 located?"),
            ("db_query_user", {"user_id": 1}, "Get information for user 1"),
            ("db_query_user", {"user_id": 5}, "Get information for user 5"),
        ]
        
        repetition_cycle = 10
        
        for i in range(num_samples):
            if i % repetition_cycle == 0:
                tool, params, question_text = random.choice(base_queries)
                questions.append({
                    "id": f"staleness_fresh_{i}",
                    "question": question_text, 
                    "query_type": "staleness_fresh",
                    "category": "staleness_pattern",
                    "tool": tool,
                    "params": params,
                    "expected_tools": [tool],
                    "expected_cache_behavior": "fresh_or_miss",
                    "is_repetition": False
                })
            else:

                cycle_start = (i // repetition_cycle) * repetition_cycle
                if cycle_start < len(questions):
                    original = questions[cycle_start]
                    questions.append({
                        "id": f"staleness_repeat_{i}",
                        "question": original["question"],
                        "query_type": "staleness_repeat",
                        "category": "staleness_pattern",
                        "tool": original["tool"],
                        "params": original["params"],
                        "expected_tools": original["expected_tools"],
                        "expected_cache_behavior": "potentially_stale",
                        "is_repetition": True,
                        "repetition_gap": i - cycle_start
                    })
        
        return questions
        
    def save_benchmark(self, questions: List[Dict], filename: str = "benchmark_dataset.json"):

        metadata = {
            "total_questions": len(questions),
            "generation_date": datetime.now().isoformat(),
            "seed": 42,
            "query_types": {},
            "categories": {},
            "expected_cache_behaviors": {},
            "tool_distribution": {}
        }
        
        for q in questions:

            qtype = q["query_type"]
            metadata["query_types"][qtype] = metadata["query_types"].get(qtype, 0) + 1

            category = q["category"]
            metadata["categories"][category] = metadata["categories"].get(category, 0) + 1
            
            cache_behavior = q["expected_cache_behavior"]
            metadata["expected_cache_behaviors"][cache_behavior] = \
                metadata["expected_cache_behaviors"].get(cache_behavior, 0) + 1

            for tool in q["expected_tools"]:
                metadata["tool_distribution"][tool] = metadata["tool_distribution"].get(tool, 0) + 1
        
        output = {
            "metadata": metadata,
            "questions": questions
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f" Saved {len(questions)} questions to {filename}")
        print(f"\nDistribution Summary:")
        print(f"  Categories: {metadata['categories']}")
        print(f"  Cache behaviors: {metadata['expected_cache_behaviors']}")
        print(f"  Top 5 tools: {sorted(metadata['tool_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]}")


if __name__ == "__main__":
    import sys 
    loader = EnhancedBenchmarkLoader(seed=42)
    if len(sys.argv) > 1:
        num_questions = int(sys.argv[1])
    else:
        num_questions = 10000  
    print(f"Generating benchmark with {num_questions} questions...")
    questions = loader.generate_full_benchmark(num_questions)
    loader.save_benchmark(questions, f"benchmark_{num_questions}_questions.json")
    print(f"\n Benchmark generation complete!")
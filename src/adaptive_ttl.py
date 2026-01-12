import logging
from typing import Dict
from datetime import datetime
import threading
import random
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AdaptiveTTLManager:
    def __init__(
        self, 
        min_hits: int = 2,           
        high_staleness: float = 0.10, 
        low_staleness: float = 0.03,    
        min_hit_rate: float = 0.3,   
        min_ttl: int = 30,         
        max_ttl: int = 3600,        
        force_initial_adjustment: bool = True,
        initial_adjustment_factor: float = 1.2,
        aggressive_learning: bool = True
    ):
        self.tool_history: Dict[str, Dict] = {}
        self.min_hits = min_hits
        self.high_staleness = high_staleness
        self.low_staleness = low_staleness
        self.min_hit_rate = min_hit_rate
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.force_initial_adjustment = force_initial_adjustment
        self.initial_adjustment_factor = initial_adjustment_factor
        self.aggressive_learning = aggressive_learning
        self._lock = threading.RLock()
        
        logger.info(
            f"AdaptiveTTLManager PRODUCTION VERSION v2.2: "
            f"min_hits={min_hits}, "
            f"force_initial_adjustment={force_initial_adjustment}, "
            f"aggressive_learning={aggressive_learning}"
        )
    
    def record_access(
        self, 
        tool_name: str, 
        was_hit: bool, 
        was_stale: bool = False
    ):

        with self._lock:
            if tool_name not in self.tool_history:
                self.tool_history[tool_name] = {
                    "hits": 0,
                    "misses": 0,
                    "stale_hits": 0,
                    "total_accesses": 0,
                    "last_adjustment": datetime.now(),
                    "current_ttl_multiplier": 1.0,
                    "adjustment_history": [],
                    "first_adjustment_made": False
                }
            
            history = self.tool_history[tool_name]
            history["total_accesses"] += 1
            
            if was_hit:
                history["hits"] += 1
                if was_stale:
                    history["stale_hits"] += 1
                    logger.debug(f"Stale hit detected for {tool_name}")
            else:
                history["misses"] += 1
            
            if self.aggressive_learning and 3 <= history["hits"] < 10: 
                self._maybe_aggressive_adjustment(tool_name, history)
    
    def _maybe_aggressive_adjustment(self, tool_name: str, history: Dict):
        if not self.aggressive_learning:
            return
        
        total_hits = history["hits"]
        total_accesses = history["total_accesses"]
        

        staleness_rate = history["stale_hits"] / total_hits if total_hits > 0 else 0
        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
        
        current_multiplier = history["current_ttl_multiplier"]
        new_multiplier = current_multiplier
        adjustment_reason = None
        

        if staleness_rate > self.high_staleness * 1.5:
            reduction_factor = 0.7
            new_multiplier = max(0.3, current_multiplier * reduction_factor)
            adjustment_reason = "aggressive_staleness_reduction"
            
            logger.info(
                f"AGGRESSIVE TTL REDUCTION for {tool_name}: "
                f"staleness={staleness_rate:.1%}, "
                f"multiplier={current_multiplier:.2f}→{new_multiplier:.2f}"
            )
        

        elif (staleness_rate < self.low_staleness * 2 and
              hit_rate >= self.min_hit_rate):
            
            increase_factor = 1.3
            new_multiplier = min(3.0, current_multiplier * increase_factor)
            adjustment_reason = "aggressive_performance_boost"
            
            logger.info(
                f"AGGRESSIVE TTL INCREASE for {tool_name}: "
                f"staleness={staleness_rate:.1%}, hit_rate={hit_rate:.1%}, "
                f"multiplier={current_multiplier:.2f}→{new_multiplier:.2f}"
            )
        
        if adjustment_reason and new_multiplier != current_multiplier:
            history["current_ttl_multiplier"] = new_multiplier
            history["last_adjustment"] = datetime.now()
            history["adjustment_history"].append({
                "timestamp": datetime.now(),
                "action": "increase" if new_multiplier > current_multiplier else "decrease",
                "reason": adjustment_reason,
                "staleness_rate": staleness_rate,
                "hit_rate": hit_rate,
                "old_multiplier": current_multiplier,
                "new_multiplier": new_multiplier,
                "aggressive": True
            })
    
    def get_recommended_ttl(self, tool_name: str, base_ttl: int) -> int:
        with self._lock:
            if tool_name not in self.tool_history:
                
                if self.force_initial_adjustment:
                    self.tool_history[tool_name] = {
                        "hits": 0,
                        "misses": 0,
                        "stale_hits": 0,
                        "total_accesses": 0,
                        "last_adjustment": datetime.now(),
                        "current_ttl_multiplier": self.initial_adjustment_factor,  
                        "adjustment_history": [{
                            "timestamp": datetime.now(),
                            "action": "increase",
                            "reason": "initial_adjustment_on_first_cache",
                            "staleness_rate": 0.0,
                            "hit_rate": 0.0,
                            "old_multiplier": 1.0,
                            "new_multiplier": self.initial_adjustment_factor,
                            "aggressive": True
                        }],
                        "first_adjustment_made": True
                    }
                    
                    recommended = int(base_ttl * self.initial_adjustment_factor)
                    recommended = max(self.min_ttl, min(self.max_ttl, recommended))
                    logger.info(
                        f"INITIAL ADJUSTMENT for NEW tool {tool_name}: "
                        f"{base_ttl}s → {recommended}s (×{self.initial_adjustment_factor:.2f})"
                    )
                    return recommended
                else:
                    self.tool_history[tool_name] = {
                        "hits": 0,
                        "misses": 0,
                        "stale_hits": 0,
                        "total_accesses": 0,
                        "last_adjustment": datetime.now(),
                        "current_ttl_multiplier": 1.0,
                        "adjustment_history": [],
                        "first_adjustment_made": False
                    }
                    return base_ttl
            
            history = self.tool_history[tool_name]
            total_hits = history["hits"]
            total_accesses = history["total_accesses"]

            current_multiplier = history["current_ttl_multiplier"]
            

            if total_hits >= self.min_hits:

                staleness_rate = history["stale_hits"] / total_hits if total_hits > 0 else 0
                hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
                
                new_multiplier = current_multiplier
                adjustment_reason = None
                

                if staleness_rate > self.high_staleness:
                    reduction_factor = 0.7 if self.aggressive_learning else 0.8
                    new_multiplier = max(0.3, current_multiplier * reduction_factor)
                    adjustment_reason = "high_staleness"
                    
                    logger.info(
                        f"TTL REDUCTION for {tool_name}: "
                        f"staleness={staleness_rate:.1%}, "
                        f"multiplier={current_multiplier:.2f}→{new_multiplier:.2f}"
                    )
                
                elif (staleness_rate < self.low_staleness and 
                      hit_rate >= self.min_hit_rate):
                    
                    increase_factor = 1.2 if self.aggressive_learning else 1.1
                    new_multiplier = min(3.0, current_multiplier * increase_factor)
                    adjustment_reason = "low_staleness_good_hits"
                    
                    logger.info(
                        f" TTL INCREASE for {tool_name}: "
                        f"staleness={staleness_rate:.1%}, hit_rate={hit_rate:.1%}, "
                        f"multiplier={current_multiplier:.2f}→{new_multiplier:.2f}"
                    )

                elif (self.aggressive_learning and
                      staleness_rate < self.low_staleness / 2 and
                      hit_rate >= 0.2):
                    
                    increase_factor = 1.4
                    new_multiplier = min(3.5, current_multiplier * increase_factor)
                    adjustment_reason = "very_low_staleness"
                    
                    logger.info(
                        f" AGGRESSIVE TTL INCREASE for {tool_name}: "
                        f"staleness={staleness_rate:.1%}, hit_rate={hit_rate:.1%}, "
                        f"multiplier={current_multiplier:.2f}→{new_multiplier:.2f}"
                    )
                

                if adjustment_reason and new_multiplier != current_multiplier:
                    history["current_ttl_multiplier"] = new_multiplier
                    history["last_adjustment"] = datetime.now()
                    history["adjustment_history"].append({
                        "timestamp": datetime.now(),
                        "action": "increase" if new_multiplier > current_multiplier else "decrease",
                        "reason": adjustment_reason,
                        "staleness_rate": staleness_rate,
                        "hit_rate": hit_rate,
                        "old_multiplier": current_multiplier,
                        "new_multiplier": new_multiplier,
                        "aggressive": "aggressive" in adjustment_reason
                    })
                    current_multiplier = new_multiplier
            
            recommended = int(base_ttl * current_multiplier)
            recommended = max(self.min_ttl, min(self.max_ttl, recommended))
            
            if recommended != base_ttl:
                logger.debug(
                    f"Adaptive TTL for {tool_name}: "
                    f"base={base_ttl}s → recommended={recommended}s "
                    f"(multiplier: {current_multiplier:.2f}x)"
                )
            
            return recommended
    
    def get_stats(self) -> Dict:
        with self._lock:
            tool_history_snapshot = {
                name: dict(history) 
                for name, history in self.tool_history.items()
            }
        
        stats = {}
        tools_with_initial_adjustments = 0
        tools_with_aggressive_adjustments = 0
        
        for tool_name, history in tool_history_snapshot.items():
            total = history["total_accesses"]
            hits = history["hits"]
            staleness_rate = history["stale_hits"] / hits if hits > 0 else 0
            hit_rate = hits / total if total > 0 else 0
            
            adjustments = len(history["adjustment_history"])
            current_multiplier = history["current_ttl_multiplier"]
            ttl_change_percent = (current_multiplier - 1.0) * 100
            
            initial_adjustments = sum(1 for adj in history["adjustment_history"] 
                                    if "initial_adjustment" in adj.get("reason", ""))
            aggressive_adjustments = sum(1 for adj in history["adjustment_history"] 
                                       if adj.get("aggressive", False))
            
            if initial_adjustments > 0:
                tools_with_initial_adjustments += 1
            if aggressive_adjustments > 0:
                tools_with_aggressive_adjustments += 1
            
            stats[tool_name] = {
                "total_accesses": total,
                "hits": hits,
                "misses": history["misses"],
                "hit_rate": f"{hit_rate*100:.1f}%",
                "stale_hits": history["stale_hits"],
                "staleness_rate": f"{staleness_rate*100:.1f}%",
                "ttl_multiplier": f"{current_multiplier:.2f}x",
                "ttl_change_percent": f"{ttl_change_percent:+.1f}%",
                "num_adjustments": adjustments,
                "initial_adjustment_made": history["first_adjustment_made"],
                "aggressive_adjustments": aggressive_adjustments,
                "has_contribution": adjustments > 0 or current_multiplier != 1.0,
                "adjustment_reasons": self._get_adjustment_summary(history["adjustment_history"])
            }
            
            if adjustments > 0:
                logger.info(
                    f" {tool_name}: {adjustments} adjustments, "
                    f"TTL multiplier={current_multiplier:.2f}x, "
                    f"initial={history['first_adjustment_made']}"
                )
        
        stats["_summary"] = {
            "total_tools_tracked": len(tool_history_snapshot),
            "tools_with_adjustments": sum(1 for h in tool_history_snapshot.values() 
                                        if h["adjustment_history"]),
            "tools_with_initial_adjustments": tools_with_initial_adjustments,
            "tools_with_aggressive_adjustments": tools_with_aggressive_adjustments,
            "total_adjustments": sum(len(h["adjustment_history"]) 
                                   for h in tool_history_snapshot.values()),
            "force_initial_adjustment": self.force_initial_adjustment,
            "aggressive_learning": self.aggressive_learning
        }
        
        return stats
    
    def _get_adjustment_summary(self, adjustments: list) -> Dict:
        if not adjustments:
            return {"no_adjustments": 1}
        
        summary = {}
        for adj in adjustments:
            reason = adj.get("reason", "unknown")
            summary[reason] = summary.get(reason, 0) + 1
        
        return summary
    
    def get_tool_history(self, tool_name: str) -> Dict:
        with self._lock:
            if tool_name not in self.tool_history:
                return {}
            return dict(self.tool_history[tool_name])
    
    def reset_tool_history(self, tool_name: str = None):
        with self._lock:
            if tool_name:
                if tool_name in self.tool_history:
                    del self.tool_history[tool_name]
            else:
                self.tool_history.clear()
    
    def export_learning_data(self) -> Dict:
        with self._lock:
            return {
                "configuration": {
                    "min_hits": self.min_hits,
                    "high_staleness": self.high_staleness,
                    "low_staleness": self.low_staleness,
                    "min_hit_rate": self.min_hit_rate,
                    "min_ttl": self.min_ttl,
                    "max_ttl": self.max_ttl,
                    "force_initial_adjustment": self.force_initial_adjustment,
                    "initial_adjustment_factor": self.initial_adjustment_factor,
                    "aggressive_learning": self.aggressive_learning
                },
                "tool_history": self.tool_history.copy(),
                "summary": {
                    "total_tools_tracked": len(self.tool_history),
                    "tools_with_adjustments": sum(
                        1 for h in self.tool_history.values() 
                        if h["adjustment_history"]
                    ),
                    "tools_with_initial_adjustments": sum(
                        1 for h in self.tool_history.values() 
                        if h["first_adjustment_made"]
                    ),
                    "total_adjustments": sum(
                        len(h["adjustment_history"]) 
                        for h in self.tool_history.values()
                    )
                }
            }
    
    def get_contribution_summary(self) -> Dict:
        stats = self.get_stats()
        
        tools_with_changes = 0
        total_adjustments = 0
        avg_ttl_change = 0.0
        tools_with_initial_boost = 0
        
        for tool_name, tool_stats in stats.items():
            if tool_name == "_summary":
                continue
                
            if tool_stats["num_adjustments"] > 0:
                tools_with_changes += 1
                total_adjustments += tool_stats["num_adjustments"]
                multiplier = float(tool_stats["ttl_multiplier"].rstrip('x'))
                avg_ttl_change += (multiplier - 1.0) * 100
            
            if tool_stats["initial_adjustment_made"]:
                tools_with_initial_boost += 1
        
        if tools_with_changes > 0:
            avg_ttl_change /= tools_with_changes
        
        return {
            "tools_tracked": len(stats) - 1,
            "tools_with_adjustments": tools_with_changes,
            "tools_with_initial_boost": tools_with_initial_boost,
            "total_adjustments": total_adjustments,
            "average_ttl_change_percent": f"{avg_ttl_change:+.1f}%",
            "contribution_score": tools_with_changes / (len(stats) - 1) if len(stats) > 1 else 0.0,
            "force_initial_adjustment_active": self.force_initial_adjustment,
            "aggressive_learning_active": self.aggressive_learning
        }
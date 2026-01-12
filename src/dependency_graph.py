from typing import Set, Dict, List
from tools import TOOLS
import logging
import threading

logger = logging.getLogger(__name__)


class ToolDependencyGraph:
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
        self._build_graph()
    
    def _build_graph(self):
        logger.info(f"Building dependency graph...")
        with self._lock:
            
            for writer_name, writer_tool in TOOLS.items():
                if not writer_tool.writes_to:
                    continue
                
                self.dependencies[writer_name] = set()
                
                for reader_name, reader_tool in TOOLS.items():
                    if reader_name == writer_name:
                        continue

                    for write_target in writer_tool.writes_to:
                        if write_target in reader_tool.reads_from:
                            self.dependencies[writer_name].add(reader_name)
                            logger.info(f"Dependency detected: {writer_name} invalidates {reader_name}")
    
    def get_invalidations(self, tool_name: str) -> Set[str]:
        with self._lock:
            return self.dependencies.get(tool_name, set()).copy()
    
    def add_manual_dependency(self, writer: str, reader: str):
        with self._lock:
            if writer not in self.dependencies:
                self.dependencies[writer] = set()
            self.dependencies[writer].add(reader)
    
    def visualize(self) -> str:
        with self._lock:
            result = "Tool Dependency Graph:\n"
            result += "=" * 50 + "\n"
            for writer, readers in self.dependencies.items():
                if readers:
                    result += f"{writer} invalidates:\n"
                    for reader in readers:
                        result += f"  → {reader}\n"
            return result
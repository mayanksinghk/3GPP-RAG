from abc import ABC, abstractmethod
import os
from pathlib import Path
import logging

# Inherits configuration if setup_logging() was called in the entry point
logger = logging.getLogger(__name__)

class BaseCorpusBuilder(ABC):
    """
    Abstract base class for all future dataset generation pipelines.
    Enforces a standard contract: download data concurrently, then process it locally.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    async def download(self):
        """Asynchronously fetch all required raw assets from remote sources."""
        pass

    @abstractmethod
    def process(self):
        """Process, normalize, and convert raw assets into RAG-ready formats."""
        pass

    async def run(self):
        """Orchestrates the complete pipeline."""
        logger.info(f"Starting corpus build pipeline in: {self.output_dir}")
        await self.download()
        self.process()
        logger.info("Corpus pipeline execution completed.")

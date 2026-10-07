import logging
import os
import argparse
import asyncio
import re

from src.create_dataset.ThreeGPPDownloader import ThreeGPPCorpusBuilder
from utility.logging_config import setup_logging

def normalize_release(release_str: str) -> str:
    """Standardizes input like '17' or 'rel-17' to 'Rel-17'."""
    match = re.search(r'\d{1,2}', str(release_str))
    if match:
        return f"Rel-{match.group(0)}"
    raise argparse.ArgumentTypeError(f"Invalid release format: '{release_str}'")

if __name__ == "__main__":
    # Configure logging for the entire module
    logger = setup_logging()
    
    # Configure basic logging as fallback
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="3GPP OOP Corpus Builder")
    parser.add_argument("--release", type=normalize_release, default="Rel-17", help="Target Release (e.g., 17)")
    parser.add_argument("--series", type=str, nargs='+', default=["38"], help="List of series (e.g., 24 33 38)")
    parser.add_argument("--output", type=str, default="./Data/3gpp_corpus", help="Base output directory")
    args = parser.parse_args()

    # Deduplicate series while preserving order
    series_list = list(dict.fromkeys(args.series))
    output_path = os.path.join(args.output, args.release)

    # Instantiate and run the pipeline
    builder = ThreeGPPCorpusBuilder(args.release, series_list, output_path)
    asyncio.run(builder.run()) 
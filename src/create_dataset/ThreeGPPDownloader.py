import asyncio
import os
import zipfile
import glob
import logging
import subprocess
import re
import uuid
from io import BytesIO
from abc import ABC, abstractmethod
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
import mammoth
from markdownify import markdownify as md
from src.create_dataset.dataset_utils import BaseCorpusBuilder

# Inherits configuration if setup_logging() was called in the entry point
logger = logging.getLogger(__name__)

class ThreeGPPCorpusBuilder(BaseCorpusBuilder):
    """
    Concrete implementation for downloading and converting 3GPP specifications.
    Utilizes concurrent downloads and a hybrid Mammoth/Markdownify conversion strategy.
    """
    BASE_URL = "https://www.3gpp.org/ftp/Specs/latest"

    def __init__(self, release: str, series_list: list, output_dir: str):
        super().__init__(output_dir)
        self.release = release
        self.series_list = series_list
        self.series_paths = {}

    async def _fetch_file_links(self, session: aiohttp.ClientSession, url: str) -> list:
        logger.info(f"Scanning remote index: {url}")
        
        # Network Retry Logic (Max 3 Attempts with Exponential Backoff)
        for attempt in range(1, 4):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    links = []
                    for a_tag in soup.find_all('a'):
                        href = a_tag.get('href')
                        if href and href.endswith('.zip'):
                            full_url = href if href.startswith('http') else f"{url.rstrip('/')}/{href}"
                            links.append(full_url)
                    return links
            except aiohttp.ClientError as e:
                if attempt == 3:
                    logger.error(f"Failed to scrape index map {url} after 3 attempts: {e}")
                    return []
                wait_time = 2 ** attempt
                logger.warning(f"Network error on {url}. Retrying {attempt}/3 in {wait_time}s... ({e})")
                await asyncio.sleep(wait_time)

    async def _download_and_extract(self, session: aiohttp.ClientSession, url: str, extract_path: str):
        filename = url.split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        
        # Check if already downloaded/extracted
        if any(os.path.exists(os.path.join(extract_path, f"{base_name}{ext}")) 
               for ext in [".doc", ".docx", ".md"]):
            logger.debug(f"Skipping download; asset checkpoint found for: {filename}")
            return

        # Network Retry Logic (Max 3 Attempts with Exponential Backoff)
        for attempt in range(1, 4):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    zip_data = BytesIO()
                    async for chunk in response.content.iter_chunked(8192):
                        zip_data.write(chunk)
                        
                    try:
                        with zipfile.ZipFile(zip_data) as zip_ref:
                            zip_ref.extractall(extract_path)
                        logger.info(f"Downloaded and extracted: {filename}")
                        return # Exit the retry loop on success
                    except zipfile.BadZipFile:
                        logger.error(f"Corrupted zip payload at {url}")
                        return # Do not retry bad zips (server-side file issue)
            except aiohttp.ClientError as e:
                if attempt == 3:
                    logger.error(f"Network stream dropped for {filename} after 3 attempts: {e}")
                    return
                wait_time = 2 ** attempt
                logger.warning(f"Network stream dropped for {filename}. Retrying {attempt}/3 in {wait_time}s... ({e})")
                await asyncio.sleep(wait_time)

    async def download(self):
        connector = aiohttp.TCPConnector(limit=15)
        async with aiohttp.ClientSession(connector=connector) as session:
            all_tasks = []
            for series in self.series_list:
                target_url = f"{self.BASE_URL}/{self.release}/{series}_series"
                series_dir = os.path.join(self.output_dir, f"{series}_series")
                os.makedirs(series_dir, exist_ok=True)
                self.series_paths[series] = series_dir
                
                zip_links = await self._fetch_file_links(session, target_url)
                if not zip_links:
                    logger.warning(f"No targets found for {series}_series")
                    continue
                
                for link in zip_links:
                    task = asyncio.create_task(self._download_and_extract(session, link, series_dir))
                    all_tasks.append(task)
                    
            if all_tasks:
                logger.info(f"Executing {len(all_tasks)} concurrent download tasks...")
                await asyncio.gather(*all_tasks)
            else:
                logger.error("No downloads scheduled.")

    def _convert_legacy_docs(self, target_dir: str):
        """Converts .doc to .docx using headless LibreOffice."""
        doc_files = glob.glob(os.path.join(target_dir, "*.doc"))
        unconverted = [f for f in doc_files if not os.path.exists(os.path.splitext(f)[0] + ".docx")]
        
        if unconverted:
            logger.info(f"Converting {len(unconverted)} legacy .doc files via LibreOffice...")
            try:
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "docx", "--outdir", target_dir] + unconverted, 
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                # Cleanup legacy .doc after successful conversion to .docx
                for doc in unconverted:
                    os.remove(doc)
            except subprocess.CalledProcessError as e:
                logger.error(f"LibreOffice conversion failed: {e}")

    def _convert_docx_to_md(self, target_dir: str):
        """Iterates over a directory to convert all .docx to Markdown."""
        docx_files = glob.glob(os.path.join(target_dir, "*.docx"))
        
        for docx_path in docx_files:
            md_path = os.path.splitext(docx_path)[0] + ".md"
            
            if os.path.exists(md_path):
                continue

            if "cover" in os.path.basename(docx_path).lower() or "annexes" in os.path.basename(docx_path).lower():
                logger.info(f"Skipping non-essential document: {os.path.basename(docx_path)}")
                continue
                
            logger.info(f"Processing Markdown conversion: {os.path.basename(docx_path)}")
            
            # Define a directory to store extracted images
            base_dir = os.path.dirname(docx_path)
            image_dir = os.path.join(base_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            def handle_image(image):
                """Writes the image to disk, converts EMF/WMF to PNG, and filters waste."""
                with image.open() as image_bytes:
                    raw_data = image_bytes.read()
                    
                    # Filter 1: Drop files smaller than 2KB
                    if len(raw_data) < 2048:
                        return {"src": ""} 
                    
                    # Filter 2: Drop tiny icons using Pillow
                    try:
                        from PIL import Image
                        from io import BytesIO
                        
                        img = Image.open(BytesIO(raw_data))
                        width, height = img.size
                        if width < 50 or height < 50:
                            return {"src": ""}
                    except Exception:
                        # Pillow will intentionally fail here on .emf/.wmf formats 
                        # because it cannot read Windows vectors. We bypass and proceed.
                        pass 

                    # Determine extension
                    ext = image.content_type.split("/")[-1].lower()
                    if ext in ["jpeg"]: ext = "jpg"
                    if ext in ["x-emf", "emf"]: ext = "emf"
                    if ext in ["x-wmf", "wmf"]: ext = "wmf"
                    
                    base_uuid = uuid.uuid4().hex[:8]
                    image_name = f"img_{base_uuid}.{ext}"
                    image_path = os.path.join(image_dir, image_name)
                    
                    # 1. Save the raw image
                    with open(image_path, "wb") as f:
                        f.write(raw_data)
                        
                    # 2. Linux EMF/WMF to PNG Conversion
                    if ext in ["emf", "wmf"]:
                        png_name = f"img_{base_uuid}.png"
                        png_path = os.path.join(image_dir, png_name)
                        
                        # Use an isolated LibreOffice profile since this runs concurrently
                        profile_dir = f"/tmp/lo_img_{uuid.uuid4().hex}"
                        import shutil
                        
                        try:
                            subprocess.run(
                                [
                                    "libreoffice", 
                                    f"-env:UserInstallation=file://{profile_dir}",
                                    "--headless", "--convert-to", "png", 
                                    "--outdir", image_dir, image_path
                                ], 
                                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                            )
                            # If successful, delete the unreadable Linux file
                            if os.path.exists(png_path):
                                os.remove(image_path)
                                image_name = png_name  # Update Markdown to point to the PNG
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Failed to convert {ext} to PNG: {e}")
                        finally:
                            shutil.rmtree(profile_dir, ignore_errors=True)
                        
                return {"src": f"images/{image_name}"}
            
            custom_style_map = (
                "p[style-name^='Heading'] => h1:fresh\n"
                "p[style-name^='3GPP_Heading'] => h1:fresh\n"
                "p[style-name='Heading 1'] => h1:fresh\n"
                "p[style-name='Heading 2'] => h2:fresh\n"
            )

            try:
                with open(docx_path, "rb") as docx_file:
                    result = mammoth.convert_to_html(
                        docx_file, 
                        style_map=custom_style_map,
                        convert_image=mammoth.images.img_element(handle_image)
                    )
                    
                markdown_content = md(
                    result.value, 
                    heading_style="ATX",
                    strip=['script', 'style'],
                    ignore_tags=['table', 'thead', 'tbody', 'tr', 'td', 'th'] 
                )
                
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_content)
            except Exception as e:
                logger.error(f"Hybrid conversion failed for {docx_path}: {e}")

    def process(self):
        logger.info("Initiating structural document transformations...")
        for series, storage_path in self.series_paths.items():
            self._convert_legacy_docs(storage_path)
            self._convert_docx_to_md(storage_path)

        # Force-kill any lingering headless instances spawned by this run
        subprocess.run(["pkill", "-f", "soffice.bin --headless"], stderr=subprocess.DEVNULL)
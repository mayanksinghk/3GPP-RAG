#!/bin/bash
set -e

if [ "$1" = "download-and-serve" ]; then
    echo "Downloading 3GPP series 38 for Rel-18..."
    uv run python app.py download --release Rel-18 --series 38 --out-dir ./output_markdowns

    # echo "Downloading all 3GPP series for Rel-18..."
    # uv run python app.py download --release Rel-18 --all-series --out-dir ./output_markdowns
    
    echo "Ingesting markdown files into Qdrant..."
    uv run python app.py ingest --dir ./output_markdowns --release Rel-18 --reset-db
    
    echo "Starting the server..."
    exec uv run python app.py serve --host 0.0.0.0 --port 8000
    
elif [ "$1" = "serve" ]; then
    echo "Starting the server..."
    exec uv run python app.py serve --host 0.0.0.0 --port 8000

else
    exec "$@"
fi
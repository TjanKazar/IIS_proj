#!/bin/bash

set -e

/home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/.venv/bin/python /home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/src/traffic_scraper.py
/home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/.venv/bin/python /home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/src/traffic_preprocess.py

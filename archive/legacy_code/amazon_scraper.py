import requests
import os
import json
from bs4 import BeautifulSoup
import re
import csv
import logging
import time

from dotenv import load_dotenv
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_product_list(products, filename, write_header=False):
    """ Function to generate format and write the product data to a csv file
    @params:
        - products: [list of dicts], contains dicts of product data
        - filename: [string], the filepath of the csv file to write to
        - write_header: [bool], the flag for writing the header row in the csv file
    """
    headers = ["title", "asin", "price", "rating", "reviews", "link", "thumbnail"]
    if not products:
        logger.info(f"No more products available - {filename}")
        return False
    
    with open(filename, 'a', newline='') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if write_header:
            writer.writeheader()

        for result in products:
            writer.writerow({
                "title": result.get("title", ""),
                "asin": result.get("asin", ""),
                "price": result.get("price", ""),
                "rating": result.get("rating", ""),
                "reviews": result.get("reviews", ""),
                "link": result.get("link_clean", ""),
                "thumbnail": result.get("thumbnail", "")
            })
    return True



def fetch_amazon_products(search_term, max_pages=20):
    """ A function designed to scrape Amazon product listings for a given search term.
    @params: 
        - search_term: [string], relevant term for searching Amazon API
        - max_pages: [int], the maximum number of pages to scrape (default: 20)
    """
    filename='./backend/data/{}_products_dataset.csv'.format(search_term.replace(" ", "_")) # Define the filepath for csv file
    
    if os.path.exists(filename):             # Clean up any old datasets with same name
        os.remove(filename)
        logger.info(f"Removed old CSV file: {filename}")

    page = 1
    total_products = 0
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "amazon",
        "k": search_term,
    }

    while page <= max_pages:
        params["page"] = page
        search = requests.get("https://serpapi.com/search", params=params)
        response = search.json()
        products = response.get("organic_results", [])

        if products:
            result = create_product_list(products, filename, write_header=(page == 1))
            total_products += len(products)
            logger.info(f"✓ Page {page}: Added {len(products)}/{total_products} products to CSV")
            page += 1
            time.sleep(15) # Wait 15 sec before starting next request
        else:
            logger.info(f"[{search_term}] - No more products available - page: {page}")
            break
    return total_products


if __name__ == "__main__":
    try:
        conditions = ["acne", "rosacea", "eczema", "psoriasis",
                      "melasma", "dark spots", "hyperpigmentation",
                      "dry skin", "oily skin", "sensitive skin"]
        start_time = time.time()
        all_products_total = 0

        for i, condition in enumerate(conditions, 1):
            logger.info(f'\n{"="*50}')
            logger.info(f"Starting fetch for {condition}")
            logger.info(f'{"="*50}\n')
            total = fetch_amazon_products(condition, max_pages=20)
            all_products_total += total

            logger.info(f"✓ Completed {condition}: {total} products")

            if i < len(conditions):
                time.sleep(5) # Buffer 5 secs before starting next condition search
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ALL COMPLETE!")
        logger.info(f"Total products: {all_products_total}")
        logger.info(f"Time taken: {elapsed/60:.2f} minutes")
        logger.info(f"{'='*50}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    

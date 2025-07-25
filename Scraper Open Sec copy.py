if __name__ == "__main__":
    import csv
    import time
    import logging
    import os

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    NAME_MAP_CSV = "name_map.csv"
    PAUSE_BETWEEN = 1.0  # seconds between scrapes

    def slug_from_filename(fn: str) -> str:
        """
        Turn 'JRutherford.jpg' -> 'jrutherford'
        Adjust if your filenames have underscores or other quirks.
        """
        name, _ = os.path.splitext(fn)
        return name.lower()

    if not os.path.exists(NAME_MAP_CSV):
        logging.error(f"{NAME_MAP_CSV} not found, aborting.")
        exit(1)

    with open(NAME_MAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename") or row.get("\\filename")
            if not fn:
                continue

            slug = slug_from_filename(fn)
            logging.info(f"Fetching top donors for '{slug}'…")
            donors = fetch_top_donors(slug)
            logging.info(f" → {len(donors)} donors for '{slug}': {donors}")

            # be polite to the server
            time.sleep(PAUSE_BETWEEN)

    # final dump of full JSON cache (in case you haven’t hit the threshold yet)
    try:
        _dump_full_json_cache()
        logging.info(f"Wrote full cache to {JSON_PATH}")
    except Exception as e:
        logging.warning(f"Error dumping JSON cache: {e}")

    # close shelve cleanly
    def close_persistent_cache():
        # Implement cache closing logic here, or pass if not needed
        pass

    close_persistent_cache()
    logging.info("Done.")


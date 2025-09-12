import os
import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# ----- CONFIGURATION -----
# Directory containing .hmm files (update with your directory)
hmm_directory = "/Users/GeorgesKanaan/Documents/Development/Methylation/OGA_MTase_Analysis/DefenseFinder_HMMs"  # <-- change this to your actual path

# CSV file to write results links
csv_file = "results_links.csv"
downloads_dir = os.path.expanduser("~/Downloads")  # Downloads directory


# ----- HELPER FUNCTION FOR RETRYING URL LOAD -----
def get_url_with_retry(driver, url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            print(f"Attempt {attempt+1} to load {url} failed: {e}")
            time.sleep(delay)
    raise Exception(f"Failed to load {url} after {retries} attempts.")


# ----- SELENIUM SETUP -----
# Make sure Safari's "Allow Remote Automation" is enabled in the Develop menu.
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 1200)  # waits up to 20 minutes per job
driver.implicitly_wait(3)
driver.command_executor.set_timeout(1000)

# ----- MAIN WORKFLOW -----
results_links = []

# List all .hmm files in the specified directory
hmm_files = [f for f in os.listdir(hmm_directory) if f.endswith(".hmm")]

for hmm_file in hmm_files:
    file_path = os.path.join(hmm_directory, hmm_file)
    print(f"Processing file: {hmm_file}")

    # Check if a file containing the HMM file name already exists in the Downloads directory
    existing_files = [fname for fname in os.listdir(downloads_dir) if hmm_file in fname]
    if existing_files:
        print(
            f"Skipping {hmm_file}: found existing file(s) in Downloads: {existing_files}"
        )
        continue

    # Use the helper function to load the page with retries.
    try:
        get_url_with_retry(
            driver, "https://tara-oceans.mio.osupytheas.fr/ocean-gene-atlas/"
        )
    except Exception as e:
        print(f"Error loading initial URL for {hmm_file}: {e}")
        continue

    try:
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "btn"))
        )
        cookie_button.click()
        print("Cookies accepted.")

    except TimeoutException:
        print("No cookie consent button found.")

    # --- FILL THE FORM ---
    try:
        # Select Dataset "OM-RGCv2+G"
        dataset_select_elem = wait.until(EC.element_to_be_clickable((By.ID, "DB")))
        select_dataset = Select(dataset_select_elem)
        select_dataset.select_by_visible_text(
            "OM-RGCv2+G - Tara Oceans Microbiome Reference Gene Catalog v2+ metaG Arctic Inside (prokaryotes)"
        )
        print("Dataset selected.")
    except Exception as e:
        print(f"Error selecting dataset for {hmm_file}: {e}")
        continue

    try:
        # Enter job title using the file name
        job_title_input = driver.find_element(By.ID, "job")
        job_title_input.clear()
        job_title_input.send_keys(hmm_file)
        print("Job title entered.")
    except Exception as e:
        print(f"Error entering job title for {hmm_file}: {e}")
        continue

    try:
        # Click the radius radio button for HMM
        radius_button = driver.find_element(By.ID, "HMM_radio")
        radius_button.click()
        print("Radius radio button clicked.")
    except Exception as e:
        print(f"Error clicking radius radio button for {hmm_file}: {e}")
        continue

    try:
        # Upload the HMM file
        file_input = driver.find_element(By.ID, "hmm_file_upload")
        # If the file input might be hidden, you can force its visibility:
        driver.execute_script("arguments[0].style.display = 'block';", file_input)
        file_input.send_keys(file_path)
        print("HMM file uploaded.")
    except Exception as e:
        print(f"Error uploading file for {hmm_file}: {e}")
        continue

    try:
        # Click the submit button
        submit_button = driver.find_element(By.ID, "submit_button")
        submit_button.click()
        print("Submit button clicked.")
    except Exception as e:
        print(f"Error clicking submit button for {hmm_file}: {e}")
        continue

    # --- WAIT FOR RESULTS ---
    try:
        # Wait until the download link appears on the results page
        download_link = wait.until(
            EC.presence_of_element_located(
                (
                    By.LINK_TEXT,
                    "Download abundances of the homologs and environmental data",
                )
            )
        )
    except Exception as e:
        print(f"Timeout or error waiting for results for {hmm_file}: {e}")
        continue

    # Save the current URL (results page) and add it to our array
    results_url = driver.current_url
    results_links.append((hmm_file, results_url))
    print(f"Results page for {hmm_file}: {results_url}")

    # --- DOWNLOAD RESULTS ---
    try:
        download_link.click()
        # Wait to allow the download to complete (adjust timing as necessary)
        time.sleep(30)
    except Exception as e:
        print(f"Error downloading results for {hmm_file}: {e}")

# ----- SAVE RESULTS TO CSV -----
try:
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["HMM File", "Results URL"])
        for row in results_links:
            writer.writerow(row)
    print(f"Results links saved to {csv_file}")
except Exception as e:
    print(f"Error writing CSV: {e}")

driver.quit()
print("Script completed.")

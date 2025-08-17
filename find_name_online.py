import os
import json
import time
import logging
from gemiwrap import GeminiWrapper
import re

# -------------------------------
# Config
# -------------------------------
FOLDER = "output/hdbscan"
SLEEP_BETWEEN_CALLS = 10  # seconds
RENAME_PAD = 3  # for numbering e.g., 001, 002, ...

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
	format="[%(asctime)s] [%(levelname)s] %(message)s",
	level=logging.INFO
)
logger = logging.getLogger("character_identifier")

# -------------------------------
# Prompt Template
# -------------------------------
PROMPT_TEMPLATE = """Please analyze the provided image and identify the character shown. 

Context: This character appears in the 2008 film "The Reader."

Question: What is this character's name in the movie?

OUTPUT FORMAT:
{
	"character_name": "<character name in the movie>",
	"portrayed_by": "<actor name who played the role>",
	"confidence_level": "<high/medium/low>"
}

Note: If information is unknown, use empty string ("")."""

# -------------------------------
# Helper Functions
# -------------------------------
def get_largest_image(folder_path: str) -> str:
	"""Return the path to the largest image file in the given folder."""
	images = [
		os.path.join(folder_path, f)
		for f in os.listdir(folder_path)
		if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
	]
	if not images:
		return None
	return max(images, key=os.path.getsize)

def rename_all_files(folder_path: str, new_base_name: str):
	"""Rename all image files in the folder to the given base name with incremental numbering."""
	files = [
		f for f in os.listdir(folder_path)
		if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
	]
	for idx, file_name in enumerate(sorted(files), start=1):
		ext = os.path.splitext(file_name)[1].lower()
		new_name = f"{new_base_name}_{str(idx).zfill(RENAME_PAD)}{ext}"
		old_path = os.path.join(folder_path, file_name)
		new_path = os.path.join(folder_path, new_name)
		os.rename(old_path, new_path)

def rename_folder(old_path: str, new_name: str):
	"""Rename the folder to the given new name (sanitized), starting suffix at _0 if conflict."""
	base_dir = os.path.dirname(old_path)
	safe_name = new_name.replace(" ", "_")
	new_path = os.path.join(base_dir, safe_name)

	# If folder exists, start numbering from _0
	counter = 0
	while os.path.exists(new_path):
		new_path = os.path.join(base_dir, f"{safe_name}_{counter}")
		counter += 1

	os.rename(old_path, new_path)
	return new_path

def sanitize_filename(name: str) -> str:
	"""
	Make a string safe for use as a file or folder name.
	Removes invalid characters for most OSes and strips trailing dots/spaces.
	"""
	# Replace invalid characters with underscores
	safe = re.sub(r'[<>:"/\\|?*]', '_', name)
	
	# Strip leading/trailing whitespace and dots
	safe = safe.strip().strip('.')
	
	# If empty after cleaning, use a placeholder
	if not safe:
		safe = ""
	
	return safe

# -------------------------------
# Main Loop
# -------------------------------
def main():
	gemini = GeminiWrapper()

	for subfolder in os.listdir(FOLDER):
		if not subfolder.startswith("person_"):
			continue

		folder_path = os.path.join(FOLDER, subfolder)
		if not os.path.isdir(folder_path):
			continue

		logger.info(f"Processing folder: {subfolder}")

		largest_img = get_largest_image(folder_path)
		if not largest_img:
			logger.warning(f"No images found in {subfolder}, skipping.")
			continue

		logger.info(f"Largest image selected: {largest_img}")

		try:
			response_list = gemini.send_message(
				user_prompt=PROMPT_TEMPLATE,
				file_path=largest_img,
				compress=False
			)
			logger.debug(f"Raw model response: {response_list}")

			parsed = json.loads(response_list[0])
			character_name = parsed.get("character_name", "").strip()
			if not character_name:
				logger.error(f"No 'character_name': {character_name} found in model response for {subfolder}")
				continue
			character_name = sanitize_filename(character_name)
			logger.info(f"Character identified: {character_name}")

			# Rename files inside folder
			rename_all_files(folder_path, character_name.replace(" ", "_"))

			# Rename the folder itself
			new_folder_path = rename_folder(folder_path, character_name)
			logger.info(f"Folder renamed to: {new_folder_path}")

		except Exception as e:
			logger.exception(f"Error processing folder {subfolder}: {e}")

		logger.info(f"Sleeping {SLEEP_BETWEEN_CALLS} seconds before next call...")
		time.sleep(SLEEP_BETWEEN_CALLS)

if __name__ == "__main__":
	main()

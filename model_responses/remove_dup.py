import os
import glob

def delete_files_with_pattern(directory, pattern):
    # Construct the search pattern for the files
    search_pattern = os.path.join(directory, f"*{pattern}*")

    # Find all files that match the pattern
    for filepath in glob.glob(search_pattern):
        # Check if the found path is a file
        if os.path.isfile(filepath):
            try:
                # Delete the file
                os.remove(filepath)
                print(f"Deleted file: {filepath}")
            except Exception as e:
                print(f"Error deleting file {filepath}: {e}")

# Example usage
directory_path = "../results/databricks__dolly-v2-12b"  # Replace with the actual directory path
pattern = "2024-05-19"  # The pattern to search for in filenames
delete_files_with_pattern(directory_path, pattern)

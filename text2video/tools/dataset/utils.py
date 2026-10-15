from pathlib import Path


def check_files(files_to_check):
    for i, current_file in enumerate(files_to_check):
        path_obj = Path(current_file)

        if not path_obj.exists():
            print(f"File <{path_obj}> does not exist.")

            original_path = str(path_obj)
            new_path = original_path.replace("../data/", "./data/")
            if not Path(new_path).exists():
                # Try to change from folder "output_files"
                new_path = new_path.replace("output_files/", "./data/")

            files_to_check[i] = new_path
            print(f"File changed: {files_to_check[i]}.")

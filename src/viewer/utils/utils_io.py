import os
import zipfile

# https://stackoverflow.com/questions/64719918/how-to-write-streamlit-uploadedfile-to-temporary-in_dir-with-original-filenam
# https://gist.github.com/benlansdell/44000c264d1b373c77497c0ea73f0ef2
# https://stackoverflow.com/questions/65612750/how-can-i-specify-the-exact-folder-in-streamlit-for-the-uploaded-file-to-be-save


def zip_folder(in_dir: str, f_out: str) -> bytes:
    """
    Zips a folder and its contents.
    """
    if os.path.exists(in_dir):
        with zipfile.ZipFile(f_out, "w") as zip_file:
            for root, dirs, files in os.walk(in_dir):
                for file in files:
                    zip_file.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), in_dir),
                    )

    with open(f_out, "rb") as f:
        out_zip = f.read()

    return out_zip


def unzip_zip_files(in_dir: str) -> None:
    """
    Unzips all ZIP files in the input dir and removes the original ZIP files.
    """
    if os.path.exists(in_dir):
        for filename in os.listdir(in_dir):
            if filename.endswith(".zip"):
                zip_path = os.path.join(in_dir, filename)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(in_dir)
                    os.remove(zip_path)

def save_uploaded_files(in_files: list, d_out: str) -> None:
    """
    Save uploaded files to the output dir and unzips zip files
    """
    # Save uploaded files
    print("Saving uploaded files")
    if in_files is not None:
        for in_file in in_files:
            f_out = os.path.join(d_out, in_file.name)
            if not os.path.exists(f_out):
                with open(os.path.join(d_out, in_file.name), "wb") as f:
                    f.write(in_file.getbuffer())
    # Unzip zip files
    print("Extracting zip files")
    if os.path.exists(d_out):
        unzip_zip_files(d_out)

def get_file_count(folder_path: str) -> int:
    count = 0
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            count += len(files)
    return count

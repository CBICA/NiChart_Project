import os
import shutil
import tkinter as tk
import zipfile
from tkinter import filedialog
from typing import Any, BinaryIO, List, Optional

# https://stackoverflow.com/questions/64719918/how-to-write-streamlit-uploadedfile-to-temporary-in_dir-with-original-filenam
# https://gist.github.com/benlansdell/44000c264d1b373c77497c0ea73f0ef2
# https://stackoverflow.com/questions/65612750/how-can-i-specify-the-exact-folder-in-streamlit-for-the-uploaded-file-to-be-save


def browse_file(path_init: str) -> Any:
    """
    File selector
    Returns the file name selected by the user and the parent folder
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_file = filedialog.askopenfilename(initialdir=path_init)
    root.destroy()
    if len(out_file) == 0:
        return None
    return out_file


def browse_folder(path_init: str) -> Any:
    """
    Folder selector
    Returns the folder name selected by the user
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir=path_init)
    root.destroy()
    if len(out_path) == 0:
        return None
    return out_path


def zip_folder(in_dir: str, f_out: str) -> Optional[bytes]:
    """
    Zips a folder and its contents.
    """
    # if os.path.exists(in_dir):
    #     with zipfile.ZipFile(f_out, "w") as zip_file:
    #         for root, dirs, files in os.walk(in_dir):
    #             for file in files:
    #                 zip_file.write(
    #                     os.path.join(root, file),
    #                     os.path.relpath(os.path.join(root, file), in_dir),
    #                 )
    #         zip_file.write(in_dir, os.path.basename(in_dir))

    if not os.path.exists(in_dir):
        return None
    else:
        shutil.make_archive(
            f_out, "zip", os.path.dirname(in_dir), os.path.basename(in_dir)
        )

        with open(f"{f_out}.zip", "rb") as f:
            dir_download = f.read()

        return dir_download


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


def copy_and_unzip_uploaded_files(in_files: list, d_out: str) -> None:
    """
    Copy uploaded files to the output dir and unzip zip files
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


def copy_uploaded_file(in_file: BinaryIO, out_file: str) -> None:
    """
    Save uploaded file to the output path
    """
    if in_file is not None:
        with open(out_file, "wb") as f:
            shutil.copyfileobj(in_file, f)


def get_file_count(folder_path: str, file_suff: str = "") -> int:
    count = 0
    if os.path.exists(folder_path):
        if file_suff == "":
            for root, dirs, files in os.walk(folder_path):
                count += len(files)
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(file_suff):
                        count += 1
    return count


def get_file_list(folder_path: str, file_suff: str = "") -> List:
    list_nifti: List[str] = []
    if not os.path.exists(folder_path):
        return list_nifti
    for f in os.listdir(folder_path):
        if f.endswith(file_suff):
            list_nifti.append(f)
    return list_nifti

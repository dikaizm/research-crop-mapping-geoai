import gdown
import os

def download_from_gdrive(file_id: str, output_path: str = None, overwrite: bool = False) -> str:
    """
    Download a file from Google Drive using its file ID.

    Parameters
    ----------
    file_id : str
        The Google Drive file ID (from the shareable link).
    output_path : str, optional
        Path to save the downloaded file. If None, uses the file's original name.
    overwrite : bool, optional
        If False and file exists, skips downloading.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    # Construct the direct download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # If no output path is given, let gdown determine filename from headers
    if output_path is None:
        output_path = gdown.download(url, quiet=False)
    else:
        if not overwrite and os.path.exists(output_path):
            print(f"✅ File already exists: {output_path}")
            return output_path
        gdown.download(url, output_path=output_path, quiet=False)

    print(f"✅ Downloaded: {output_path}")
    return output_path
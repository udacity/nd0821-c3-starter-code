import os


def find_repo_root(current_path):
    """
    Finds the root of the repository by looking for the .git directory.

    Parameters
    ----------
    current_path : str
        The current file path.

    Returns
    -------
    root_path : str
        The root path of the repository.
    """
    while current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, ".git")):
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError("Repository root with .git directory not found.")

import pytest

from src.file_util import find_repo_root


@pytest.mark.parametrize(
    "dir_structure, start_path, expected_root",
    [
        (["repo/.git", "repo/subdir/file.txt"], "repo/subdir", "repo"),
        (
            ["repo/.git", "repo/subdir1/subdir2/file.txt"],
            "repo/subdir1/subdir2",
            "repo",
        ),
        (["repo/subdir/.git", "repo/subdir/file.txt"], "repo/subdir", "repo/subdir"),
        (
            ["repo/subdir/.git", "repo/subdir/subsubdir/file.txt"],
            "repo/subdir/subsubdir",
            "repo/subdir",
        ),
    ],
)
def test_find_repo_root(tmpdir, dir_structure, start_path, expected_root):
    for path in dir_structure:
        tmpdir.join(path).ensure(dir=True)

    start_path_full = tmpdir.join(start_path)
    expected_root_full = tmpdir.join(expected_root)

    assert find_repo_root(str(start_path_full)) == str(expected_root_full)


def test_find_repo_root_not_found(tmpdir):
    # Create a directory structure without a .git directory
    dir_structure = ["repo/subdir1/subdir2/file.txt"]
    for path in dir_structure:
        tmpdir.join(path).ensure(dir=True)

    start_path_full = tmpdir.join("repo/subdir1/subdir2")

    with pytest.raises(
        FileNotFoundError, match="Repository root with .git directory not found."
    ):
        find_repo_root(str(start_path_full))

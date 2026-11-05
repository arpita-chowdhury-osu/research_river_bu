# tests

## `test_download_replace.py`: re-downloading over a corrupt image

**Problem**

`download.py` re-downloads an image whose file is present but has the wrong sha256. It used to save the verified download with `tmp.rename(dest)`. On Windows, `rename` refuses to overwrite an existing file (`FileExistsError`, WinError 183), so the download was reported as `failed`. The corrupt file stayed in place and the correct bytes were left behind in `<name>.part`, and running the command again failed the same way.

**Fix**

`tmp.replace(dest)` overwrites the old file on every OS. The order is unchanged: the hash is checked first, the `.part` file is written next, and only then is the target replaced. A download with the wrong hash still never touches the target.

**Nothing changes for users**

Commands, defaults, retries and hash checks are the same as before:

```
python download.py --split val --limit 5
```

**Run the test**

Standard library only. No network is used and no image is downloaded:

```
python tests/test_download_replace.py -v
```

The test covers four cases:
- (a) no file yet → saved;
- (b) correct file → skipped, no network call;
- (c) wrong file → replaced, then skipped on the next call;
- (d) download with the wrong hash → rejected, old file kept.

With the previous `rename`, case (c) fails on Windows (checked on Windows 11 with Python 3.12). On macOS/Linux, Python documents that `rename` already overwrites an existing file; this was not tested there.

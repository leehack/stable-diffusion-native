# Carried upstream patches

Fixes the pinned stable-diffusion.cpp lacks, carried until a pin includes them.
`CMakeLists.txt` applies each patch to build-tree copies of the upstream
sources it names and compiles the copies, so the submodule stays pristine.
Configuring fails when a patch no longer applies or changes nothing; then drop
or rebase it. Build info and the release manifest list the patches next to the
upstream commit.

A patch here needs a regression test that fails against the unpatched pin and
an upstream issue or pull request to retire it.

| Patch | Sources | Fixes | Test |
| --- | --- | --- | --- |
| `pickle_line_span.patch` | `src/model_io/pickle_io.cpp` | A legacy PyTorch checkpoint above 2 GiB is rejected: the pickle scanner narrows the rest of the file to `int` before looking for a newline ([#23](https://github.com/leehack/stable-diffusion-native/issues/23)). | `tests/test_pickle_span.py` |

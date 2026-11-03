"""Model-free causal controls for the actual carried torch ZIP reader.

Compiles unchanged zip.c/pickle_io.cpp and the parent/carried torch reader.
Fixtures are tiny tensor-only pickle byte streams, never Python-unpickled.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "third_party/stable-diffusion.cpp"
READER = Path("src/model_io/torch_zip_io.cpp")
PATCH = REPO / "patches/torch_zip_entry_lookup.patch"
NATIVE = REPO / "tests/native/torch_zip_lookup_test.cpp"
CXX = os.environ.get("CXX") or shutil.which("c++")
CC = os.environ.get("CC") or shutil.which("cc")
GIT = shutil.which("git")


def compile_source(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise AssertionError(
            f"Compiler exited {result.returncode}: {shlex.join(command)}\n"
            f"stdout (last 8192 characters):\n{result.stdout[-8192:]}\n"
            f"stderr (last 8192 characters):\n{result.stderr[-8192:]}")


def string(value):
    data = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(data)) + data


def integer(value):
    return b"J" + struct.pack("<i", value)


def tensor(name, storage, dtype="FloatStorage", elements=1, offset=0):
    return (string(name) + b"ctorch._utils\n_rebuild_tensor_v2\n(" + b"(" +
            string("storage") + b"ctorch\n" + dtype.encode() + b"\n" +
            string(storage) + string("cpu") + integer(elements) + b"tQ" +
            integer(offset) + integer(1) + b"\x85" + integer(1) + b"\x85\x89" +
            b"ccollections\nOrderedDict\n)RtR")


def pickle_data(tensors, nested=False):
    data = b"}(" + b"".join(tensors) + b"u"
    return b"\x80\x02" + (b"}" + string("state_dict") + data + b"s" if nested else data) + b"."


@unittest.skipIf(sys.platform == "win32", "uses GCC/Clang command lines")
class TorchZipLookupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = not (CC and CXX and GIT and (UPSTREAM / READER).is_file() and
                       (UPSTREAM / "ggml/include/ggml.h").is_file())
        if missing:
            message = "requires C/C++ compilers, git and initialized upstream/ggml submodules"
            if os.environ.get("CI"):
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        cls.temp = tempfile.TemporaryDirectory(prefix="torch-zip-lookup-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.patched = cls.root / "patched" / READER
        cls.patched.parent.mkdir(parents=True)
        cls.patched.write_bytes((UPSTREAM / READER).read_bytes().replace(b"\r\n", b"\n"))
        subprocess.run([GIT, "apply", str(PATCH)], cwd=cls.root / "patched", check=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(cls.root)})
        if cls.patched.read_bytes() == (UPSTREAM / READER).read_bytes():
            raise AssertionError("patch left reader unchanged")
        cls.parent = cls.build("parent", UPSTREAM / READER)
        cls.fixed = cls.build("fixed", cls.patched)

    @classmethod
    def build(cls, name, reader, flags=()):
        directory = cls.root / name
        copied = directory / READER
        copied.parent.mkdir(parents=True)
        copied.write_bytes(reader.read_bytes())
        obj = directory / "zip.o"
        # Match the library's CMake GNU C11 mode: zip.c uses POSIX/default-source
        # declarations that glibc hides when strict C11 defines __STRICT_ANSI__.
        compile_source([CC, "-std=gnu11", "-O1", "-D_FILE_OFFSET_BITS=64", *flags,
                        "-c", str(UPSTREAM / "thirdparty/zip.c"), "-o", str(obj)])
        binary = directory / "probe"
        includes = [REPO / "tests/native/torch_zip_metadata_shim", directory / "src",
                    UPSTREAM / "src", UPSTREAM / "src/model_io", UPSTREAM / "thirdparty",
                    UPSTREAM / "ggml/include"]
        compile_source([CXX, "-std=c++17", "-O1", "-g", *flags,
                        *("-I" + str(path) for path in includes), str(NATIVE),
                        str(UPSTREAM / "src/model_io/pickle_io.cpp"), str(obj), "-o", str(binary)])
        return binary

    def archive(self, name, entries):
        path = self.root / (name + ".pt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                for key, data in entries:
                    archive.writestr(key, data)
        return path

    def probe(self, binary, *archives, fault="none"):
        result = subprocess.run([str(binary), fault, *(str(p) for p in archives)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), len(archives))
        for row in rows:
            self.assertEqual(row["archives_opened"], row["archives_closed"], row)
            self.assertEqual(row["live_pickle_buffers"], 0, row)
            self.assertEqual(row["payload_reads"], 0, row)
        return rows

    def pair(self, name, entries, ok):
        archive = self.archive(name, entries)
        parent = self.probe(self.parent, archive)[0]
        fixed = self.probe(self.fixed, archive)[0]
        self.assertEqual(parent["ok"], ok, parent)
        for field in ("ok", "error", "inventory", "metadata_reads"):
            self.assertEqual(parent[field], fixed[field], (field, parent, fixed))
        self.assertLessEqual(fixed["entry_opens"], 2 * len(entries))
        return archive, fixed

    def test_linear_operation_counts_and_unpatched_causal_control(self):
        for count in (16, 32, 64, 1323):
            with self.subTest(tensors=count):
                archive = self.archive("scale-" + str(count), [
                    ("archive/data.pkl", pickle_data([tensor("w" + str(i), str(i))
                                                      for i in range(count)])),
                    *(("archive/data/" + str(i), bytes(4)) for i in range(count))])
                fixed = self.probe(self.fixed, archive)[0]
                self.assertTrue(fixed["ok"], fixed)
                self.assertEqual(len(fixed["inventory"]), count)
                self.assertEqual(fixed["entry_opens"], 2 * (count + 1))
                self.assertEqual(fixed["entry_names"], fixed["entry_opens"])
                if count <= 64:
                    parent = self.probe(self.parent, archive)[0]
                    self.assertEqual(parent["inventory"], fixed["inventory"])
                    self.assertEqual(parent["entry_opens"], (count * count + 5 * count + 2) // 2)
                    self.assertGreater(parent["entry_opens"], 2 * (count + 1),
                                       "upstream lookup is now bounded; retire/re-evaluate the patch")

    def test_first_match_paths_repeated_and_shared_storage(self):
        cases = {
            "reverse": ([tensor("b", "1"), tensor("a", "0"), tensor("again", "1")],
                        [("archive/data/0", bytes(4)), ("archive/data/1", bytes(4))], True),
            "duplicate-valid": ([tensor("a", "0")], [("archive/data/0", bytes(4)),
                                                     ("archive/data/0", b"x")], True),
            "duplicate-short": ([tensor("a", "0")], [("archive/data/0", b"x"),
                                                     ("archive/data/0", bytes(4))], False),
            "duplicate-revisit": ([tensor("a", "0"), tensor("b", "1"), tensor("again", "0")],
                                  [("archive/data/0", bytes(4)), ("archive/data/0", b"x"),
                                   ("archive/data/1", bytes(4))], True),
            "backslash": ([tensor("a", "0")], [("archive\\data\\0", bytes(4)),
                                               ("archive/data/0", b"x")], True),
            "case": ([tensor("a", "0")], [("archive/data/Zero", bytes(4))], False),
            "dot": ([tensor("a", "0")], [("archive/data/./0", bytes(4))], False),
            "long": ([tensor("a", "x" * 600)], [("archive/data/" + "x" * 600, bytes(4))], True),
            "shared": ([tensor("a", "0", elements=4), tensor("view", "0", elements=4, offset=1)],
                       [("archive/data/0", bytes(16))], True),
            "overflow": ([tensor("a", "0", elements=4, offset=4)], [("archive/data/0", bytes(16))], False),
        }
        for name, (tensors, storage, ok) in cases.items():
            with self.subTest(case=name):
                _, row = self.pair(name, [("archive/data.pkl", pickle_data(tensors)), *storage], ok)
                if name == "shared":
                    self.assertEqual([t["index"] for t in row["inventory"]], [1, 1])
                    self.assertEqual([t["offset"] for t in row["inventory"]], [0, 4])
                if name == "duplicate-revisit":
                    self.assertEqual([t["index"] for t in row["inventory"]], [1, 3, 1])

    def test_multiple_metadata_records_and_independent_reader_lifetimes(self):
        first, _ = self.pair("two-metadata", [
            ("first/data.pkl", pickle_data([tensor("first", "0")])), ("first/data/0", bytes(4)),
            ("second/data.pkl", pickle_data([tensor("second", "1")])), ("second/data/1", bytes(4))], True)
        self.pair("repeat-metadata", [("archive/data.pkl", pickle_data([tensor("first", "0")])),
                  ("archive/data/0", bytes(4)), ("archive/data.pkl", pickle_data([tensor("second", "0")]))], True)
        second, _ = self.pair("storage-first", [("archive/data/0", bytes(4)),
                              ("archive/data.pkl", pickle_data([tensor("other", "0")]))], True)
        parent = self.probe(self.parent, first, second, first)
        fixed = self.probe(self.fixed, first, second, first)
        self.assertEqual([r["inventory"] for r in parent], [r["inventory"] for r in fixed])

    def test_scalar_widths_nested_empty_and_rejected_metadata(self):
        for dtype, width in (("HalfStorage", 2), ("BFloat16Storage", 2), ("FloatStorage", 4),
                             ("IntStorage", 4), ("DoubleStorage", 8), ("LongStorage", 8)):
            _, row = self.pair(dtype, [("archive/data.pkl", pickle_data([tensor("w", "0", dtype)], True)),
                                      ("archive/data/0", bytes(width))], True)
            self.assertEqual(row["inventory"][0]["bytes"], width)
        self.pair("empty", [("archive/data.pkl", pickle_data([]))], False)
        self.pair("missing", [("archive/data.pkl", pickle_data([tensor("w", "missing")]))], False)
        self.pair("short", [("archive/data.pkl", pickle_data([tensor("w", "0", elements=2)])),
                            ("archive/data/0", bytes(4))], False)
        self.pair("bad-pickle", [("archive/data.pkl", b"\x80\x02\xff.")], False)
        self.pair("no-pickle", [("archive/data/0", bytes(4))], False)
        # The existing reader recognizes the substring, not only a basename.
        self.pair("metadata-substring", [("archive/data.pkl.suffix", pickle_data([tensor("w", "0")])),
                                         ("archive/data/0", bytes(4))], True)

    def test_open_name_and_index_range_failures_release_archive_and_pickle(self):
        archive = self.archive("faults", [("archive/data.pkl", pickle_data([tensor("w", "0")])),
                                          ("archive/data/0", bytes(4))])
        for fault in ("outer-open", "lookup-open", "outer-name", "lookup-name",
                      "negative-count", "large-count"):
            with self.subTest(fault=fault):
                failed, = self.probe(self.fixed, archive, fault=fault)
                self.assertFalse(failed["ok"], failed)
                self.assertTrue(failed["error"], failed)
                self.assertEqual(failed["archives_opened"], 1)
        self.assertTrue(self.probe(self.fixed, archive)[0]["ok"])

    def test_bounds_and_failure_cleanup_under_sanitizers(self):
        binary = self.build("sanitized", self.patched,
                            ("-fsanitize=address,undefined", "-fno-omit-frame-pointer"))
        archive = self.archive("sanitized", [
            ("archive/data.pkl", pickle_data([tensor("w", "0", elements=4, offset=4)])),
            ("archive/data/0", bytes(16))])
        for fault in ("none", "lookup-open", "lookup-name", "outer-open", "outer-name",
                      "negative-count", "large-count"):
            with self.subTest(fault=fault):
                row, = self.probe(binary, archive, fault=fault)
                self.assertFalse(row["ok"], row)

    def test_genuine_semantic_and_cleanup_mutants_are_detected(self):
        source = self.patched.read_text()
        mutants = {
            "last-match": ("lookup.entries.emplace(name, ZipEntryLookup::Entry{(int)i, entry_size});",
                           "lookup.entries.insert_or_assign(name, ZipEntryLookup::Entry{(int)i, entry_size});"),
            "truncated-name": ("std::string name = raw_name;",
                               "std::string name = raw_name; if (name.size() > 512) name.resize(512);"),
            "missing-bounds": ("if (tensor_nbytes < 0 ||\n            tensor_storage.offset > entry_size ||\n            static_cast<uint64_t>(tensor_nbytes) > entry_size - tensor_storage.offset)", "if (false)"),
            "missing-free": ("            free(pkl_data);", "            (void)pkl_data;"),
        }
        cases = {
            "last-match": [("archive/data.pkl", pickle_data([tensor("a", "0"), tensor("b", "1"), tensor("again", "0")])),
                           ("archive/data/0", bytes(4)), ("archive/data/0", b"x"), ("archive/data/1", bytes(4))],
            "truncated-name": [("archive/data.pkl", pickle_data([tensor("a", "x" * 600)])),
                               ("archive/data/" + "x" * 600, bytes(4))],
            "missing-bounds": [("archive/data.pkl", pickle_data([tensor("a", "0", elements=4, offset=4)])),
                               ("archive/data/0", bytes(16))],
            "missing-free": [("archive/data.pkl", pickle_data([tensor("a", "missing")]))],
        }
        for name, (before, after) in mutants.items():
            with self.subTest(mutant=name):
                self.assertIn(before, source)
                mutated = self.root / (name + ".cpp")
                mutated.write_text(source.replace(before, after))
                binary = self.build("mutant-" + name, mutated)
                archive = self.archive(name, cases[name])
                baseline = self.probe(self.fixed, archive)[0]
                result = subprocess.run([str(binary), "none", str(archive)], capture_output=True,
                                        text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                observed = json.loads(result.stdout)
                fields = ("ok", "error", "inventory", "live_pickle_buffers")
                self.assertNotEqual({k: baseline[k] for k in fields}, {k: observed[k] for k in fields})


class CompilerDiagnosticsTest(unittest.TestCase):
    def test_failed_compile_exposes_bounded_stdout_and_stderr(self):
        failed = subprocess.CompletedProcess(["cc", "source.c"], 1,
                                             "x" * 20000 + "stdout-marker",
                                             "y" * 20000 + "stderr-marker")
        with mock.patch.object(subprocess, "run", return_value=failed):
            with self.assertRaises(AssertionError) as caught:
                compile_source(["cc", "source.c"])
        message = str(caught.exception)
        self.assertIn("Compiler exited 1: cc source.c", message)
        self.assertIn("stdout-marker", message)
        self.assertIn("stderr-marker", message)
        self.assertLess(len(message), 17000)


if __name__ == "__main__":
    unittest.main()

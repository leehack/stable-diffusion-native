"""Narrow validation of the compiled tri_f16 rejected-triangle zero store."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

MAGIC = 0x07230203


def _validate_flow(data: bytes) -> None:
    if len(data) < 20 or len(data) % 4:
        raise ValueError('tri_f16: malformed SPIR-V size')
    words = struct.unpack(f'<{len(data) // 4}I', data)
    if words[0] != MAGIC or words[4] != 0:
        raise ValueError('tri_f16: malformed SPIR-V header')
    instructions, at = [], 5
    while at < len(words):
        count, op = words[at] >> 16, words[at] & 65535
        if count == 0 or at + count > len(words):
            raise ValueError('tri_f16: malformed SPIR-V instruction')
        instructions.append((op, words[at + 1:at + count]))
        at += count
    if instructions[-1][0] != 56 or sum(op == 54 for op, _ in instructions) != 1:
        raise ValueError('tri_f16: incomplete or ambiguous entry function')
    floats = {args[0]: args[1] for op, args in instructions if op == 22 and len(args) == 2}
    constants = {args[1]: (args[0], args[2:]) for op, args in instructions if op == 43 and len(args) >= 3}
    pointers = {args[0]: args[1:] for op, args in instructions if op == 32 and len(args) == 3}
    bindings = {args[0]: args[2] for op, args in instructions if op == 71 and len(args) == 3 and args[1] == 33}
    # The operation selects four triangle modes, then either copies binding 0
    # to binding 1 or writes zero to binding 1. Different layouts need review.
    switches = [args for op, args in instructions if op == 251 and len(args) == 10
                and set(args[2::2]) == {0, 1, 2, 3}]
    if len(switches) != 1 or sorted(bindings.values()) != [0, 1]:
        raise ValueError('tri_f16: missing or ambiguous triangle operation')
    blocks, current = {}, None
    for op, args in instructions:
        if op == 248:
            current = args[0]
            blocks[current] = []
        elif current is not None:
            blocks[current].append((op, args))
    rejected = []
    copied = []
    for label, block in blocks.items():
        conversions = {args[1]: (args[0], args[2]) for op, args in block if op == 115 and len(args) == 3}
        access = {args[1]: (args[0], args[2]) for op, args in block if op == 65 and len(args) >= 4}
        for op, args in block:
            if op != 62 or len(args) != 2 or args[1] not in conversions or args[0] not in access:
                continue
            result_type, value = conversions[args[1]]
            pointer_type, base = access[args[0]]
            if floats.get(result_type) != 16 or pointers.get(pointer_type) != (12, result_type) or bindings.get(base) != 1:
                continue
            if value in constants:
                type_id, literal = constants[value]
                if floats.get(type_id) != 32 or literal != (0,):
                    raise ValueError('tri_f16: rejected triangle does not store positive zero')
                # This constant must feed only this conversion, avoiding unused
                # zero constants masking a malformed output data path.
                uses = sum(1 for other_op, other_args in instructions
                           if other_op == 115 and len(other_args) == 3 and other_args[2] == value)
                if uses != 1:
                    raise ValueError('tri_f16: ambiguous zero conversion')
                rejected.append(label)
            else:
                # The source half is widened to float before narrowing for output.
                if value in conversions:
                    widened_type, value = conversions[value]
                    if floats.get(widened_type) != 32:
                        continue
                loads = {a[1]: a[2] for o, a in block if o == 61 and len(a) >= 3}
                if value in loads and loads[value] in access:
                    _, input_base = access[loads[value]]
                    if bindings.get(input_base) == 0:
                        copied.append(label)
    branches = [(args[1], args[2]) for op, args in instructions if op == 250 and len(args) in (3, 5)]
    if len(rejected) != 1 or len(copied) != 1 or (copied[0], rejected[0]) not in branches:
        raise ValueError('tri_f16: missing or ambiguous accepted/rejected output branches')


# Normalizing only the rejected-branch float constant admits the observed
# compiler defect for identification, while flow validation still rejects it.
KNOWN_LAYOUTS = {
    "a8691344947943aa581022ecfc95953c02ea3247f85abd0d2f46a23ad49cbe0f",
    "c38f10b1d14aa4078c2508fd3aa4ad10a799c02dd64dc2eb7f8e50f0ede4e2f7",
}
GENERATED_MODULE = Path('sd/ggml/src/ggml-vulkan/vulkan-shaders.spv/tri_f16.spv')


def normalized_identity(data: bytes) -> str | None:
    if len(data) < 20 or len(data) % 4:
        return None
    words = list(struct.unpack(f'<{len(data) // 4}I', data))
    floats, constants, conversions = {}, {}, []
    at = 5
    while at < len(words):
        count, op = words[at] >> 16, words[at] & 65535
        if not count or at + count > len(words):
            return None
        args = words[at + 1:at + count]
        if op == 22 and len(args) == 2:
            floats[args[0]] = args[1]
        elif op == 43 and len(args) == 3:
            constants[args[1]] = (args[0], at + 3)
        elif op == 115 and len(args) == 3:
            conversions.append((args[0], args[2]))
        at += count
    identities = []
    for result_type, value in conversions:
        if floats.get(result_type) == 16 and value in constants:
            source_type, offset = constants[value]
            if floats.get(source_type) == 32:
                normalized = words[:]
                normalized[offset] = 0
                digest = hashlib.sha256(struct.pack(f'<{len(words)}I', *normalized)).hexdigest()
                if digest in KNOWN_LAYOUTS:
                    identities.append(digest)
    return identities[0] if len(identities) == 1 else None


def validate_module(data: bytes) -> None:
    try:
        _validate_flow(data)
    except (IndexError, struct.error) as error:
        raise ValueError('tri_f16: malformed instruction operands') from error
    if normalized_identity(data) is None:
        raise ValueError('tri_f16: unrecognized module layout; review compiler output')


def embedded_modules(data: bytes) -> tuple[list[bytes], int]:
    found, unknown, position = [], 0, 0
    while True:
        start = data.find(b'\x03\x02\x23\x07', position)
        if start < 0:
            break
        position = start + 4
        if start + 20 > len(data):
            continue
        header = struct.unpack_from('<5I', data, start)
        if not (0x10000 <= header[1] <= 0x10600 and header[4] == 0):
            continue
        at = start + 20
        while at + 4 <= len(data) and at - start < 1000000:
            word = struct.unpack_from('<I', data, at)[0]
            count = word >> 16
            if not count or at + count * 4 > len(data):
                break
            at += count * 4
            if word & 65535 == 56:
                module = data[start:at]
                if normalized_identity(module) is not None:
                    found.append(module)
                else:
                    try:
                        _validate_flow(module)
                        unknown += 1
                    except (ValueError, IndexError, struct.error) as error:
                        if 'does not store positive zero' in str(error):
                            unknown += 1
                break
    return found, unknown


def build_record(build_root: Path) -> dict:
    path = build_root / GENERATED_MODULE
    if not path.is_file():
        raise ValueError('tri_f16: generated module is missing')
    data = path.read_bytes()
    validate_module(data)
    return {'schemaVersion': 1, 'sha256': hashlib.sha256(data).hexdigest(), 'spirv_hex': data.hex()}


def validate_artifact_bytes(binary: bytes, record: dict | None, *, allow_legacy: bool = False) -> None:
    found, unknown = embedded_modules(binary)
    if len(found) != 1 or unknown:
        raise ValueError('tri_f16: runtime module is missing, ambiguous or unrecognized')
    data = found[0]
    validate_module(data)
    if record is None and allow_legacy:
        return
    try:
        if type(record['schemaVersion']) is not int or record['schemaVersion'] != 1:
            raise ValueError('tri_f16: unsupported compiled module schema')
        captured = bytes.fromhex(record['spirv_hex'])
        expected = record['sha256']
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('tri_f16: missing compiled module provenance') from error
    if hashlib.sha256(captured).hexdigest() != expected:
        raise ValueError('tri_f16: compiled module digest mismatch')
    if captured != data:
        raise ValueError('tri_f16: captured module differs from packaged runtime')


def validate_artifact(library: Path, record: dict | None, *, allow_legacy: bool = False) -> None:
    validate_artifact_bytes(library.read_bytes(), record, allow_legacy=allow_legacy)

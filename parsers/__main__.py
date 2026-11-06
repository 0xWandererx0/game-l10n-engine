"""
Paket Stüdyosu terminal komutları.

    python -m parsers detect    English.asr
    python -m parsers extract   English.asr                 # -> output/studio/English/English.source.json + .meta.json
    python -m parsers repack    English.asr English.tr.json # -> output/studio/English/English_TR.asr
    python -m parsers roundtrip English.asr                 # bayt bayt doğrulama + uzunluk değişimi stres testi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from config import BASE_DIR
from parsers.asura import AsuraParser, entry_keys, parse_asura
from parsers.base import GameParserError, restore_glyphs
from parsers.studio import PackageStudio

ENCODINGS = {"auto": None, "utf-16-le": "utf-16-le", "utf-8": "utf-8", "cp1254": "cp1254"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _stress_text(text: str, index: int) -> str:
    """Uzunluğu değişen Türkçe benzeri metin: her kaydın boyu farklı oranda değişir."""
    if not text:
        return text
    return f"ÇĞİÖŞÜ çğıöşü #{index}: " + text + " (çevrildi)" * (index % 3)


def roundtrip(path: Path, studio: PackageStudio) -> int:
    data = path.read_bytes()
    info = studio.inspect(path)
    print(f"Dosya     : {path.name} ({len(data):,} bayt, sha256 {_sha(data)}…)")
    print(f"{info.describe()}")
    with tempfile.TemporaryDirectory() as tmp:
        temp_studio = PackageStudio(Path(tmp))
        package = temp_studio.extract(path)
        strings = json.loads(package.json_path.read_text(encoding="utf-8"))

        print("\n[1] Değişiklik yok: extract -> JSON -> repack")
        same = temp_studio.repack(path, package.json_path)
        rebuilt = same.output_path.read_bytes()
        identical = rebuilt == data
        print(f"    sha256 orijinal {_sha(data)}  |  repack {_sha(rebuilt)}  ->  "
              f"{'BİREBİR AYNI' if identical else 'FARKLI!'}")
        if not identical:
            return 1

        print("\n[2] Stres testi: her metnin uzunluğu değiştirildi (Türkçe karakterlerle)")
        changed = {label: {key: _stress_text(value, i) for i, (key, value) in enumerate(table.items())}
                   for label, table in strings.items()}
        changed_json = Path(tmp) / f"{path.stem}.tr.json"
        changed_json.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        result = temp_studio.repack(path, changed_json, meta_path=package.meta_path)
        output = result.output_path.read_bytes()
        print(f"    {result.changed} metin gömüldü, {result.unchanged} değişmedi, uyarı: {len(result.warnings)}")
        print(f"    dosya boyutu {len(data):,} -> {len(output):,} bayt")

        if info.parser_id == "asura":
            original, new = parse_asura(data), parse_asura(output)
            print(f"    {'chunk':<22}{'eski boyut':>12}{'yeni boyut':>12}{'kayıt':>8}  kontrol")
            position = 8
            for before, after in zip(original.chunks, new.chunks):
                ok = after.offset == position and after.size == 16 + len(after.body)
                position += after.size
                count = len(after.entries) if after.entries is not None else "-"
                print(f"    {(after.label or after.type.decode()):<22}{before.size:>12,}{after.size:>12,}{count:>8}  "
                      f"{'ofset+boyut tutarlı' if ok else 'HATA'}")
            print(f"    8 + Σchunk + bitiş({len(new.trailer)}) = {position + len(new.trailer):,} = dosya boyutu "
                  f"{len(output):,}  ->  {'TUTARLI' if position + len(new.trailer) == len(output) else 'HATA'}")
            # Her kaydın uzunluk alanını ham baytlardan bağımsız olarak yeniden say
            mismatches = 0
            for chunk in new.text_chunks():
                expected = changed[chunk.label]
                for key, entry in zip(entry_keys(len(chunk.entries)), chunk.entries):
                    mismatches += entry.text != restore_glyphs(expected[key])
            print(f"    geri okunan metin uyuşmazlığı: {mismatches}")
            back = AsuraParser().extract(output).strings
            restored = json.loads(json.dumps(back, ensure_ascii=False))
            print(f"    yeni dosya yeniden extract edildi: {sum(len(t) for t in restored.values())} metin, "
                  f"glif token'ları korundu: {restored == changed}")
            return 0 if mismatches == 0 and restored == changed else 1
        reread = studio.detector.get(info.parser_id)().extract(output).strings
        print(f"    yeni dosya yeniden extract edildi, eşleşme: {reread == changed}")
        return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="python -m parsers", description="Oyun Paket Stüdyosu")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("detect", "extract", "roundtrip"):
        command = sub.add_parser(name)
        command.add_argument("file", type=Path)
        command.add_argument("--format", choices=["asura", "vdf"])
        command.add_argument("--encoding", choices=list(ENCODINGS), default="auto")
    repack = sub.add_parser("repack")
    repack.add_argument("original", type=Path)
    repack.add_argument("translated_json", type=Path)
    repack.add_argument("-o", "--output", type=Path)
    repack.add_argument("--meta", type=Path)
    repack.add_argument("--latin1", action="store_true", help="ğ→g, ş→s, ı→i (fontta Türkçe glif yoksa)")
    parser.add_argument("--work-dir", type=Path, default=BASE_DIR / "output" / "studio")
    args = parser.parse_args(argv)

    studio = PackageStudio(args.work_dir)
    try:
        if args.command == "detect":
            print(studio.inspect(args.file, format_override=args.format, encoding=ENCODINGS[args.encoding]).describe())
        elif args.command == "extract":
            package = studio.extract(args.file, format_override=args.format, encoding=ENCODINGS[args.encoding])
            print(f"Algılanan Format: {package.info.describe()}")
            print(f"JSON : {package.json_path}\nMeta : {package.meta_path}\nÇeviri motoru çıktısı: {package.translated_path}")
        elif args.command == "repack":
            result = studio.repack(args.original, args.translated_json, meta_path=args.meta,
                                   output_path=args.output, latin1_fallback=args.latin1)
            print(f"{result.changed} metin gömüldü, {result.unchanged} orijinal kaldı")
            print(f"Boyut: {result.size_before:,} -> {result.size_after:,} bayt\nÇıktı: {result.output_path}")
            for warning in result.warnings:
                print(f"  ! {warning}")
        else:
            return roundtrip(args.file, studio)
    except (GameParserError, OSError) as exc:
        print(f"HATA: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--project",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--return-dir",type=Path,required=True);args=parser.parse_args()
    project=args.project.resolve();output=args.output.resolve();destination=args.return_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
    for path in destination.glob("PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part*"):path.unlink()
    source_files=[]
    tracked=subprocess.check_output(["git","-C",str(project.parent),"ls-files",project.name],text=True).splitlines()
    for relative in tracked:
        path=project.parent/relative
        if path.is_file():source_files.append({"scope":"source","path":relative,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    result_files=[]
    for path in sorted(output.rglob("*")):
        if path.is_file():result_files.append({"scope":"result","path":str(path.relative_to(output)),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    manifest={"protocol":"PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1","source_files":source_files,"result_files":result_files,"raw_c1_included":False}
    forbidden=[item["path"] for item in [*source_files,*result_files] if str(item["path"]).lower().endswith((".xlsx",".xls")) or "PRISM_SHARED_DATA_C1" in str(item["path"])]
    if forbidden:raise RuntimeError(f"raw data selected for archive: {forbidden[:3]}")
    manifest_path=output/"RELEASE_ASSET_MANIFEST.json";manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    prefix=destination/"PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part"
    tar=subprocess.Popen(["tar","--zstd","--exclude=__pycache__","--exclude=*.pyc","--exclude=.pytest_cache","-cf","-","-C",str(project.parent),project.name,"-C",str(output),"."],stdout=subprocess.PIPE)
    if tar.stdout is None:raise RuntimeError("tar stdout unavailable")
    split=subprocess.run(["split","-b","1800M","-d","-a","3","-",str(prefix)],stdin=tar.stdout,check=True);tar.stdout.close()
    if tar.wait()!=0:raise RuntimeError("tar packaging failed")
    parts=[]
    for path in sorted(destination.glob("PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part*")):parts.append({"name":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    decompressor=subprocess.Popen(["zstd","-dc"],stdin=subprocess.PIPE,stdout=subprocess.PIPE);listing=subprocess.Popen(["tar","-tf","-"],stdin=decompressor.stdout,stdout=subprocess.DEVNULL)
    if decompressor.stdin is None or decompressor.stdout is None:raise RuntimeError("archive validation pipeline unavailable")
    for part in sorted(destination.glob("PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part*")):
        with part.open("rb") as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b""):decompressor.stdin.write(chunk)
    decompressor.stdin.close();decompressor.stdout.close()
    if listing.wait()!=0 or decompressor.wait()!=0:raise RuntimeError("streamed archive validation failed")
    assets={"status":"PASS","archive_stream_validation":"PASS","raw_data_scan":"PASS","parts":parts,"reassemble":"cat PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part* > PRISM_V2_MODULAR_CPU_RESULTS.tar.zst","extract":"tar --zstd -xf PRISM_V2_MODULAR_CPU_RESULTS.tar.zst"}
    asset_path=destination/"PRISM_V2_RELEASE_PARTS.json";asset_path.write_text(json.dumps(assets,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","manifest_files":len(source_files)+len(result_files),"parts":parts,"asset_manifest":str(asset_path)},ensure_ascii=False))


if __name__=="__main__":main()

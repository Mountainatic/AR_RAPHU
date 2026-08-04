from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import zstandard


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


class MultipartWriter(io.RawIOBase):
    def __init__(self,prefix:Path,limit:int)->None:
        self.prefix=prefix;self.limit=limit;self.index=-1;self.current=None;self.size=0
    def writable(self)->bool:return True
    def _open(self)->None:
        if self.current is not None:self.current.close()
        self.index+=1;self.current=(self.prefix.parent/f"{self.prefix.name}{self.index:03d}").open("wb");self.size=0
    def write(self,value:bytes)->int:
        view=memoryview(value);written=0
        while written<len(view):
            if self.current is None or self.size>=self.limit:self._open()
            count=min(len(view)-written,self.limit-self.size);self.current.write(view[written:written+count]);self.size+=count;written+=count
        return written
    def close(self)->None:
        if self.current is not None:self.current.close();self.current=None
        super().close()


class MultipartReader(io.RawIOBase):
    def __init__(self,parts:list[Path])->None:self.parts=parts;self.index=0;self.current=parts[0].open("rb") if parts else None
    def readable(self)->bool:return True
    def readinto(self,buffer:bytearray)->int:
        if self.current is None:return 0
        total=0;view=memoryview(buffer)
        while total<len(view) and self.current is not None:
            count=self.current.readinto(view[total:])
            if count:total+=count
            else:
                self.current.close();self.index+=1;self.current=self.parts[self.index].open("rb") if self.index<len(self.parts) else None
        return total
    def close(self)->None:
        if self.current is not None:self.current.close();self.current=None
        super().close()


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
    prefix=destination/"PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part";writer=MultipartWriter(prefix,1800*1024*1024)
    compressor=zstandard.ZstdCompressor(level=6,threads=4).stream_writer(writer,closefd=False)
    def filter_entry(info:tarfile.TarInfo)->tarfile.TarInfo|None:
        parts=Path(info.name).parts
        return None if any(part in {"__pycache__",".pytest_cache",".git"} for part in parts) or info.name.endswith(".pyc") else info
    with tarfile.open(fileobj=compressor,mode="w|") as archive:
        archive.add(project,arcname=project.name,filter=filter_entry);archive.add(output,arcname="PRISM_V2_MODULAR_CPU_RESULTS",filter=filter_entry)
    compressor.close();writer.close()
    parts=[]
    for path in sorted(destination.glob("PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part*")):parts.append({"name":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    part_paths=sorted(destination.glob("PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part*"));reader=MultipartReader(part_paths);decompressor=zstandard.ZstdDecompressor().stream_reader(reader);listed=0
    with tarfile.open(fileobj=decompressor,mode="r|") as archive:
        for _ in archive:listed+=1
    decompressor.close();reader.close()
    assets={"status":"PASS","archive_stream_validation":"PASS","archive_listing_count":listed,"raw_data_scan":"PASS","parts":parts,"reassemble":"cat PRISM_V2_MODULAR_CPU_RESULTS.tar.zst.part* > PRISM_V2_MODULAR_CPU_RESULTS.tar.zst","extract":"python -m zstandard -d PRISM_V2_MODULAR_CPU_RESULTS.tar.zst -o PRISM_V2_MODULAR_CPU_RESULTS.tar && tar -xf PRISM_V2_MODULAR_CPU_RESULTS.tar"}
    asset_path=destination/"PRISM_V2_RELEASE_PARTS.json";asset_path.write_text(json.dumps(assets,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","manifest_files":len(source_files)+len(result_files),"parts":parts,"asset_manifest":str(asset_path)},ensure_ascii=False))


if __name__=="__main__":main()

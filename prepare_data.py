"""Download Xeno-canto recordings and create recording-disjoint segment manifests."""
import argparse
from pathlib import Path
from time import sleep
import numpy as np
import pandas as pd
import requests
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SPECIES_10 = {'cuculus_canorus','apus_apus','corvus_cornix','falco_tinnunculus',
              'dendrocopos_major','anas_platyrhynchos','erithacus_rubecula',
              'pica_pica','columba_livia','parus_major'}


def segment_starts(path, duration, method):
    if duration < 1:
        return []
    if duration <= 5:
        return [0.0]
    if method == 'uniform':
        return np.linspace(0, duration - 5, min(int(duration // 5), 5)).tolist()
    # Original RMS-based approach: 5-second windows, 1-second stride.
    y, sr = librosa.load(path, sr=32000, mono=True)
    if len(y) < sr:
        return []
    if len(y) <= 5*sr:
        return [0.0]
    starts = list(range(0, len(y)-5*sr+1, sr))
    scores = [np.sqrt(np.mean(y[i:i+5*sr]**2)) for i in starts]
    return sorted(starts[i]/sr for i in np.argsort(scores)[-5:])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--audio-dir',type=Path,default=Path('local/audio'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--classes',type=int,choices=[10,15],default=15)
    p.add_argument('--segmentation',choices=['uniform','rms'],default='uniform')
    p.add_argument('--download',action='store_true',help='Explicitly enable network downloads')
    p.add_argument('--max-train-per-class',type=int,default=0,help='Optional segment cap; 0 disables')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.manifest,dtype={'id':str}).drop_duplicates('id')
    if args.classes==10: df=df[df.label.isin(SPECIES_10)].copy()
    if df.label.nunique()!=args.classes: raise ValueError('Unexpected number of classes')
    rows=[]; failures=[]
    for row in tqdm(df.to_dict('records'),desc='Audio files'):
        ext=Path(str(row['file-name'])).suffix.lower()
        if ext not in {'.mp3','.flac','.wav','.ogg','.m4a'}: ext='.mp3'
        path=args.audio_dir/row['label']/f"XC{row['id']}{ext}"
        try:
            if not path.is_file():
                if not args.download: raise FileNotFoundError(path)
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+'.part')
                with requests.get(f"https://xeno-canto.org/{row['id']}/download",timeout=60,stream=True) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in response.iter_content(1024*128): f.write(chunk)
                tmp.replace(path); sleep(.5)
            duration=librosa.get_duration(path=path)
            if duration<1: raise ValueError('Shorter than one second')
            rows.append({**row,'local_path':str(path.resolve()),'duration_sec':duration})
        except Exception as exc:
            failures.append({'id':row['id'],'error':str(exc)})
    pd.DataFrame(failures,columns=['id','error']).to_csv(args.out/'download_failures.csv',index=False)
    available=pd.DataFrame(rows)
    if available.empty or available.label.nunique()!=args.classes:
        raise RuntimeError('Not all classes have readable audio. See download_failures.csv.')
    train,temp=train_test_split(available,test_size=.30,stratify=available.label,random_state=42)
    val,test=train_test_split(temp,test_size=.50,stratify=temp.label,random_state=42)
    for name,split in [('train',train),('val',val),('test',test)]:
        split.to_csv(args.out/f'{name}_recordings.csv',index=False)
        segments=[]
        for row in tqdm(split.to_dict('records'),desc=f'{name}: {args.segmentation}'):
            for i,start in enumerate(segment_starts(row['local_path'],row['duration_sec'],args.segmentation)):
                segments.append({'segment_id':f"{name}_{row['id']}_{i}",'recording_id':row['id'],
                    'label':row['label'],'local_path':row['local_path'],'start_time':float(start),'duration':5.0})
        frame=pd.DataFrame(segments)
        if name=='train' and args.max_train_per_class:
            frame=pd.concat([g.sample(min(len(g),args.max_train_per_class),random_state=42) for _,g in frame.groupby('label')],ignore_index=True)
        frame.to_csv(args.out/f'{name}_segments.csv',index=False)
        print(name,len(split),'recordings',len(frame),'segments')

if __name__=='__main__': main()

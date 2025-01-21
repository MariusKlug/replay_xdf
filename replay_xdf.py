#!/usr/bin/env python3
import sys
import time
import threading
import numpy as np
import pyxdf
from pylsl import StreamInfo, StreamOutlet

DEFAULT_FILE_PATH = "C:/projects/situation_awareness/data/0_raw-data/s1/s1_experiment.xdf"

def replay_stream(stream_data, stream_timestamps, name, stype, sfreq, uid, stop_event):
    print(f"[INFO] Starting replay for stream '{name}' (type: '{stype}', {len(stream_data)} samples).")
    stream_data = np.asarray(stream_data)
    stream_timestamps = np.asarray(stream_timestamps)
    if stream_data.ndim == 1:
        stream_data = stream_data.reshape(-1, 1)

    ch_count = stream_data.shape[1]
    if stream_data.dtype.kind in ['U', 'S', 'O']:
        data_type = 'string'
    else:
        data_type = 'float32'
        stream_data = stream_data.astype(np.float32)

    info = StreamInfo(name=name, type=stype, channel_count=ch_count,
                      nominal_srate=sfreq, channel_format=data_type, source_id=uid)
    outlet = StreamOutlet(info)

    start_time = time.time()
    t0 = stream_timestamps[0]
    for sample, ts in zip(stream_data, stream_timestamps):
        if stop_event.is_set():
            print(f"[INFO] Stopping replay of stream '{name}'.")
            break
        wait_time = (ts - t0) - (time.time() - start_time)
        if wait_time > 0:
            time.sleep(wait_time)
        if data_type == 'string':
            outlet.push_sample([str(x) for x in sample])
        else:
            outlet.push_sample(sample.tolist())

def replay_xdf(xdf_file, stop_event):
    print(f"[INFO] Loading XDF file: {xdf_file}")
    streams, _ = pyxdf.load_xdf(xdf_file)
    print(f"[INFO] Found {len(streams)} streams in '{xdf_file}'.")

    threads = []
    for s in streams:
        name = s['info'].get('name', ['Unnamed'])[0]
        stype = s['info'].get('type', ['Data'])[0]
        uid = s['info'].get('uid', [name])[0]
        sfreq = float(s['info'].get('nominal_srate', [0])[0])
        data = s['time_series']
        ts = s['time_stamps']

        if len(data) and len(ts):
            print(f"[INFO] Creating replay thread for stream '{name}'...")
            t = threading.Thread(target=replay_stream,
                                 args=(data, ts, name, stype, sfreq, uid, stop_event))
            t.start()
            threads.append(t)
        else:
            print(f"[WARN] Stream '{name}' has no data or timestamps; skipping.")

        print(f"[INFO] Successfully created replay thread for stream '{name}'...")
    input("[INFO] Press Enter to stop streaming...\n")
    print("[INFO] Stop signal received. Waiting for threads to finish.")
    stop_event.set()

    for t in threads:
        t.join()
    print("[INFO] All streams stopped. Exiting.")

def main():
    stop_event = threading.Event()
    xdf_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE_PATH
    replay_xdf(xdf_file, stop_event)

if __name__ == "__main__":
    main()

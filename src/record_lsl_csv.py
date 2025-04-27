import numpy as np
import pylsl
import time
import os
import csv
import re

class LSLRecorder:
    """
    Records arbitrary named LSL streams directly to CSV files as data arrives.
    Each session is saved to a subdirectory 'r_<n>' under the base directory, where n increments from 0.
    """
    def __init__(
        self,
        streams: dict[str, str],
        csv_dir: str,
        update_interval: float = 0.01,
    ):
        """
        :param streams: Mapping from key names to LSL stream names.
        :param csv_dir: Base directory for saving CSVs.
        :param update_interval: Sleep interval between pulls.
        """
        self.csv_dir = csv_dir
        self.update_interval = update_interval
        self._proc_flags = (
            pylsl.proc_clocksync |
            pylsl.proc_dejitter |
            pylsl.proc_monotonize
        )

        self.streams = streams

        # Ensure base directory exists
        os.makedirs(self.csv_dir, exist_ok=True)
        # Determine next session index
        existing = [d for d in os.listdir(self.csv_dir)
                    if os.path.isdir(os.path.join(self.csv_dir, d))]
        indices = []
        for d in existing:
            m = re.match(r"r_(\d+)$", d)
            if m:
                indices.append(int(m.group(1)))
        next_idx = max(indices) + 1 if indices else 0
        session_dir = os.path.join(self.csv_dir, f"r_{next_idx}")
        os.makedirs(session_dir, exist_ok=True)

        self._configs = {}

        for key, name in self.streams.items():
            # Resolve and open LSL inlet
            found = pylsl.resolve_byprop('name', name, timeout=5)
            if not found:
                raise RuntimeError(f"No LSL stream named '{name}' found.")
            inlet = pylsl.StreamInlet(
                found[0], max_buflen=1024, processing_flags=self._proc_flags
            )
            info = inlet.info()
            nch = info.channel_count()

            # Prepare CSV file and writer in the session directory
            fname = os.path.join(session_dir, f"{key}.csv")
            f = open(fname, 'w', newline='')
            writer = csv.writer(f)

            # Write header row
            desc = info.desc().child('channels').child('channel')
            labels = []
            while not desc.empty():
                labels.append(desc.child_value('label'))
                desc = desc.next_sibling()
            if not labels:
                labels = [f"{key}_{i+1}" for i in range(nch)]
            writer.writerow(['timestamp'] + labels)

            self._configs[key] = {'inlet': inlet, 'writer': writer, 'file': f}
            print(f"[{key}] writing to {fname} with channels {labels}")

    def update(self):
        """Pull new chunks and write to CSV immediately."""
        for cfg in self._configs.values():
            data, stamps = cfg['inlet'].pull_chunk(timeout=0.0)
            if data:
                arr = np.array(data)
                for sample, stamp in zip(arr, stamps):
                    ts_ms = int(round(stamp * 1000))
                    row = [ts_ms] + sample.tolist()
                    cfg['writer'].writerow(row)
                cfg['file'].flush()

    def collect_and_close(self):
        """Continuously update until interrupted, then close files."""
        print("Start recording. Press Ctrl+C to stop.")
        try:
            while True:
                self.update()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nStopping and closing files...")
            for cfg in self._configs.values():
                cfg['file'].close()
            print("All files closed.")


def main():
    recorder = LSLRecorder(
        streams={
            'exg': "raw_exg",
            'prompt': "finger_prompt"
        },
        csv_dir="data/s_test"
    )
    recorder.collect_and_close()


if __name__ == "__main__":
    main()

import pylsl
from hand_serial import HandSerial
import time
            
def client_send_serial(lsl_name:str, left_hand:bool=True):
    # ------------------ Configuration ------------------
    SERIAL_PORT = '/dev/cu.usbserial-120'  # Set to None to auto-detect
    
    hand_serial = HandSerial(SERIAL_PORT)

    # Resolve LSL streams
    print("Looking for FingerPredictions stream...")
    streams = pylsl.resolve_byprop("name", lsl_name)
    if not streams:
        print(f"No {lsl_name} stream found.")
        return

    inlet_status = pylsl.StreamInlet(streams[0], processing_flags=pylsl.proc_ALL)

    try:
        while True:
            # Pull latest samples
            samples, _ = inlet_status.pull_chunk(timeout=0.01)
            
            if samples:
                sample = samples[-1]
                if left_hand:
                    # Reverse the order of the fingers
                    sample = sample[::-1]
                hand_serial.send_serial(sample)
                print("Sent:", sample)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        hand_serial.close()

if __name__ == "__main__":
    client_send_serial(lsl_name="FingerStatus", left_hand=True)

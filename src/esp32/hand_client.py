import pylsl
from hand_serial import HandSerial
import time
            
def client_send_serial(lsl_name:str, serial_port:str=None, left_hand:bool=False, right_hand:bool=False):
    # ------------------ Configuration ------------------
    hand_serial = HandSerial(serial_port, left_hand=left_hand, right_hand=right_hand)

    # Resolve LSL streams
    print(f"Looking for {lsl_name} stream...")
    streams = pylsl.resolve_byprop("name", lsl_name)
    if not streams:
        print(f"No {lsl_name} stream found.")
        return

    inlet = pylsl.StreamInlet(streams[0], processing_flags=pylsl.proc_ALL)

    try:
        while True:
            # Pull latest samples
            samples, _ = inlet.pull_chunk(timeout=0.01)
            
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
    client_send_serial(lsl_name="finger_prompt", left_hand=True)

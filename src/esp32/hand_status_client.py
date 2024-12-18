import pylsl
from hand_serial import HandSerial
            
def mediapipe_client_send_serial():
    # ------------------ Configuration ------------------
    SERIAL_PORT = '/dev/cu.usbserial-310'  # Set to None to auto-detect

    LSL_STATUS_NAME = "FingerStatus"
    
    hand_serial = HandSerial(SERIAL_PORT)
    # ---------------------------------------------------

    # Resolve LSL streams
    print("Looking for HandLandmarks stream...")
    streams = pylsl.resolve_byprop("name", LSL_STATUS_NAME)
    if not streams:
        print(f"No {LSL_STATUS_NAME} stream found.")
        return

    inlet_status = pylsl.StreamInlet(streams[0], processing_flags=pylsl.proc_ALL)

    try:
        while True:
            # Pull latest samples
            status_samples, _ = inlet_status.pull_chunk(timeout=0.01)
            
            if status_samples:
                status_sample = status_samples[-1]
                hand_serial.send_serial(status_sample)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        hand_serial.close()

if __name__ == "__main__":
    mediapipe_client_send_serial()

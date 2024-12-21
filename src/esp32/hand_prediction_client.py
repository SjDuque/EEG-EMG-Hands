import pylsl
from hand_client import client_send_serial
import time
            
if __name__ == "__main__":
    client_send_serial(lsl_name="FingerPredictions", left_hand=True)

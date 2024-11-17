from openbci_handtracking import process_frames

def main():
    data_folder = 'data_session_lucas/EMG_Hand_Data_20241116_012044'
    camera_frames = f'{data_folder}/camera_frames'
    finger = f'{data_folder}/fingers.csv'
    
    process_frames(camera_frames, finger)
    
if __name__ == '__main__':
    main()
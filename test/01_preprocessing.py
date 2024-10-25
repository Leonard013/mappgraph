from scapy.all import rdpcap, IP, TCP, UDP, ICMP, DNS, ARP, IPv6, Raw
import os
from concurrent.futures import ProcessPoolExecutor

def pcap_to_csv(pcap_file, output_file):
    """
    Convert a PCAP (Packet Capture) file to CSV format, extracting network flow information.
    
    This function processes network packets from a PCAP file and creates a CSV file containing
    detailed information about each packet, including timestamps, stream identification, and 
    protocol information. It tracks TCP and UDP flows by assigning unique stream IDs to 
    bidirectional connections.

    Args:
        pcap_file (str): Path to the input PCAP file to be processed
        output_file (str): Path where the output CSV file will be saved

    CSV Output Format:
        - time: Packet timestamp
        - stream_id: Unique identifier for each network flow
        - protocol: Network protocol (tcp/udp)
        - source_address: Source IP address
        - source_port: Source port number
        - destination_address: Destination IP address
        - destination_port: Destination port number
        - length: Packet length in bytes

    Note:
        - Requires scapy library for PCAP processing
        - Only processes TCP and UDP packets with valid IP information
        - Bidirectional flows are assigned the same stream ID
        - Packets missing critical information are skipped

    Example:
        >>> pcap_to_csv("capture.pcap", "output.csv")
        # Creates a CSV file with network flow information

    Raises:
        Exception: Prints error message when packet processing fails
    """
    strea_dict = {}
    counter = 0
    n = 0
    packets = rdpcap(pcap_file)

    with open(output_file, 'w') as f:
        f.write(",time,stream_id,protocol,source_address,source_port,destination_address,destination_port,length\n")
        
        for i, pkt in enumerate(packets):
            try:
                # Extract the timestamp
                timestamp = pkt.time

                # Initialize default values
                src_ip = stream_id = dst_ip = src_port = dst_port = protocol = ""
                pkt_size = len(pkt)

                # Extract IP layer information
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst

                # Extract TCP layer information
                if TCP in pkt:
                    src_port = str(pkt[TCP].sport)
                    dst_port = str(pkt[TCP].dport)
                    protocol = "tcp"
                # Extract UDP layer information
                elif UDP in pkt:
                    src_port = str(pkt[UDP].sport)
                    dst_port = str(pkt[UDP].dport)
                    protocol = "udp"
                
                if not protocol or not src_port or not dst_port or not src_ip or not dst_ip:
                    continue

                
                # Assign stream ID to each packet
                direct1 = str([protocol, src_ip, src_port, dst_ip, dst_port])
                direct2 = str([protocol, dst_ip, dst_port, src_ip, src_port])

                if direct1 in strea_dict or direct2 in strea_dict:
                    stream_id = strea_dict[direct1] if direct1 in strea_dict else strea_dict[direct2]
                else:
                    strea_dict[direct1] = str(counter)
                    stream_id = strea_dict[direct1]
                    counter += 1


                # Write the extracted information to the file
                n += 1
                f.write(f"{n-1},{timestamp},{stream_id},{protocol},{src_ip},{src_port},{dst_ip},{dst_port},{pkt_size}\n")

            except Exception as e:
                # Handle exceptions for packets that do not contain expected layers
                print(f"An error occurred: {e}")  


def copy_and_process_files(src_root, dest_root):
    """
    Recursively processes PCAP files from a source directory and converts them to CSV format in a destination directory.
    
    This function walks through a directory tree, maintaining the directory structure while converting
    PCAP files to CSV format. It skips hidden files and directories (those starting with '.') and 
    processes only new files that haven't been converted yet.

    Args:
        src_root (str): Path to the source root directory containing PCAP files
        dest_root (str): Path to the destination root directory where CSV files will be saved

    Directory Processing:
        - Maintains the same directory structure from source to destination
        - Creates destination directories if they don't exist
        - Skips hidden directories (starting with '.')
        - Processes all non-hidden files in each directory

    File Processing:
        - Converts each PCAP file to CSV using pcap_to_csv function
        - Output CSV files maintain the same name as input files (with .csv extension)
        - Skips files that have already been processed (if CSV exists)
        - Skips hidden files (starting with '.')

    Progress Tracking:
        - Prints processing status for each file
        - Shows completion status with checkmark emoji
        - Visual separation between file processing outputs

    Example:
        >>> copy_and_process_files("/path/to/pcap/files", "/path/to/csv/output")
        Processing file1.pcap ... 🔄
        Processed file1.pcap ✅
        -----------------------------------
        Processing file2.pcap ... 🔄
        Processed file2.pcap ✅
        -----------------------------------

    Dependencies:
        - Requires pcap_to_csv function for file conversion
        - Uses os module for file system operations
    """
    for root, dirs, files in os.walk(src_root):
        # Crea la struttura di directory nella destinazione
        for dir_name in dirs:
            if dir_name.startswith("."):
                continue
            src_dir_path = os.path.join(root, dir_name)
            dest_dir_path = os.path.join(dest_root, os.path.relpath(src_dir_path, src_root))
            if not os.path.exists(dest_dir_path):
                os.makedirs(dest_dir_path)
            for root2, dirs2, files2 in os.walk(src_dir_path):
                for file in files2:
                    if not file.startswith("."):
                        print(f"Processing {file} ... 🔄")
                        src_file_path = os.path.join(root2, file)
                        dest_file_path = os.path.join(dest_dir_path, os.path.splitext(os.path.basename(file))[0]+".csv")
                        if not os.path.exists(dest_file_path):
                            pcap_to_csv(src_file_path, dest_file_path)
                        print(f"Processed {file} ✅\n-----------------------------------")


if __name__ == "__main__":
    
    root_path = "path" # path to the dataset folder

    src_root = root_path + "/raw_data"
    dest_root = root_path + "/sources" 
    copy_and_process_files(src_root, dest_root)
